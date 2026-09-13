package server

import (
	"database/sql"
	"encoding/json"
	"os"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
)

// deviceNamesPersistEvery bounds how often the remembered names are written when nothing about
// them changed. Devices are re-enumerated every few seconds while anything polls, so writing
// on every refresh would rewrite them constantly; LastSeen survives a restart to within
// this interval instead.
const deviceNamesPersistEvery = 10 * time.Minute

// deviceNames remembers the name each device -- keyed by PCI address -- had the last time the
// server enumerated it healthy, so a device that has since faulted can still be called what
// it was ("that was CUDA1").
//
// A faulted device leaves the enumeration, and the names are enumeration order, so they shift:
// with the first of two cards gone the survivor becomes CUDA0. Only a record taken while the
// card was healthy can say what it was called, and that record has to outlive the process,
// because a server started after the fault -- which happened on 2026-09-12, when a deploy
// restarted it with GPU1 already faulted -- never saw the card at all. It is kept in the
// server database (server_db.go); without one it lives in the process only.
type deviceNames struct {
	mu        sync.Mutex
	db        *serverDB
	byPCI     map[string]seenDevice
	persisted time.Time
	now       func() time.Time
}

type seenDevice struct {
	Name     string    `json:"name"`
	Library  string    `json:"library,omitempty"`
	LastSeen time.Time `json:"last_seen"`
}

const deviceNamesSchema = `
CREATE TABLE IF NOT EXISTS device_names (
	pci_id TEXT PRIMARY KEY,
	name TEXT NOT NULL,
	library TEXT,
	last_seen_ms INTEGER NOT NULL
);
`

// seenDeviceRow is one device's record as the database writer receives it.
type seenDeviceRow struct {
	PCIID string
	seenDevice
}

func upsertDeviceName(tx *sql.Tx, r seenDeviceRow) error {
	_, err := tx.Exec(`INSERT INTO device_names (pci_id, name, library, last_seen_ms) VALUES (?,?,?,?)
		ON CONFLICT(pci_id) DO UPDATE SET name = excluded.name, library = excluded.library, last_seen_ms = excluded.last_seen_ms`,
		r.PCIID, r.Name, nullIfEmpty(r.Library), r.LastSeen.UnixMilli())
	return err
}

// newDeviceNames loads what an earlier process remembered from db, if anything. A nil db keeps
// the memory in the process only.
func newDeviceNames(db *serverDB) *deviceNames {
	d := &deviceNames{db: db, byPCI: map[string]seenDevice{}, now: time.Now}
	if db == nil {
		return d
	}
	rows, err := db.db.Query(`SELECT pci_id, name, library, last_seen_ms FROM device_names`)
	if err != nil {
		return d
	}
	defer rows.Close()
	for rows.Next() {
		var pci, name string
		var library sql.NullString
		var seen int64
		if rows.Scan(&pci, &name, &library, &seen) == nil {
			d.byPCI[pci] = seenDevice{Name: name, Library: library.String, LastSeen: time.UnixMilli(seen)}
		}
	}
	return d
}

// importDeviceNames files the records of a device-names.json.
func (d *serverDB) importDeviceNames(path string) (int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	var byPCI map[string]seenDevice
	if err := json.Unmarshal(b, &byPCI); err != nil {
		return 0, err
	}
	tx, err := d.db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback() //nolint:errcheck
	for pci, s := range byPCI {
		if err := upsertDeviceName(tx, seenDeviceRow{PCIID: pci, seenDevice: s}); err != nil {
			return 0, err
		}
	}
	return len(byPCI), tx.Commit()
}

// observe records the devices an enumeration returned. Every one of them is healthy by
// construction -- a faulted device is not returned -- so this is exactly the record wanted.
func (d *deviceNames) observe(devices []ml.DeviceInfo) {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()

	now := d.now()
	changed := false
	for _, dev := range devices {
		if dev.PCIID == "" || dev.Name == "" {
			continue
		}
		prev, ok := d.byPCI[dev.PCIID]
		if !ok || prev.Name != dev.Name || prev.Library != dev.Library {
			changed = true
		}
		d.byPCI[dev.PCIID] = seenDevice{Name: dev.Name, Library: dev.Library, LastSeen: now}
	}
	if d.db == nil || (!changed && now.Sub(d.persisted) < deviceNamesPersistEvery) {
		return
	}
	for pci, s := range d.byPCI {
		d.db.enqueue(seenDeviceRow{PCIID: pci, seenDevice: s})
	}
	d.persisted = now
}

// lookup returns what the device at pci was last called while healthy.
func (d *deviceNames) lookup(pci string) (seenDevice, bool) {
	if d == nil || pci == "" {
		return seenDevice{}, false
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	s, ok := d.byPCI[pci]
	return s, ok
}
