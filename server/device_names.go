package server

import (
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
)

// deviceNamesPersistEvery bounds how often the remembered names are written when nothing about
// them changed. Devices are re-enumerated every few seconds while anything polls, so writing
// on every refresh would rewrite the file constantly; LastSeen survives a restart to within
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
// restarted it with GPU1 already faulted -- never saw the card at all.
type deviceNames struct {
	mu        sync.Mutex
	path      string
	byPCI     map[string]seenDevice
	persisted time.Time
	now       func() time.Time
}

type seenDevice struct {
	Name     string    `json:"name"`
	Library  string    `json:"library,omitempty"`
	LastSeen time.Time `json:"last_seen"`
}

// newDeviceNames loads what an earlier process remembered from path, if anything. An empty
// path keeps the memory in the process only.
func newDeviceNames(path string) *deviceNames {
	d := &deviceNames{path: path, byPCI: map[string]seenDevice{}, now: time.Now}
	if path == "" {
		return d
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return d
	}
	if err := json.Unmarshal(b, &d.byPCI); err != nil {
		slog.Warn("ignoring unreadable device name memory", "path", path, "error", err)
		d.byPCI = map[string]seenDevice{}
	}
	return d
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
	if d.path == "" || (!changed && now.Sub(d.persisted) < deviceNamesPersistEvery) {
		return
	}
	if err := d.writeLocked(); err != nil {
		slog.Debug("could not persist device names", "path", d.path, "error", err)
		return
	}
	d.persisted = now
}

func (d *deviceNames) writeLocked() error {
	b, err := json.MarshalIndent(d.byPCI, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(d.path), ".device-names-*")
	if err != nil {
		return err
	}
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	return os.Rename(tmp.Name(), d.path)
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
