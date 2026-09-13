package server

import (
	"database/sql"
	"encoding/json"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// The box profile in the server database: every measurement, never overwritten, so a card whose
// bandwidth drifts can be seen doing it. The profiler uses the latest per machine identity.

const profileSchema = `
CREATE TABLE IF NOT EXISTS profiles (
	id INTEGER PRIMARY KEY,
	identity TEXT NOT NULL,
	measured_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS profiles_identity ON profiles(identity, measured_at_ms);
CREATE TABLE IF NOT EXISTS profile_devices (
	profile_id INTEGER NOT NULL,
	pci_id TEXT NOT NULL,
	bandwidth_bytes_per_sec INTEGER, token_overhead_ms REAL, layer_overhead_us REAL, fit_error_pct REAL
);
CREATE TABLE IF NOT EXISTS profile_links (
	profile_id INTEGER NOT NULL,
	pci_ids TEXT NOT NULL,
	width INTEGER NOT NULL,
	us REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS profile_failures (
	profile_id INTEGER NOT NULL,
	what TEXT NOT NULL,
	pci_ids TEXT,
	error TEXT
);
`

// profileRow is one measurement of one machine, as the database writer receives it.
type profileRow struct {
	Identity string
	Profile  storedProfile
}

func insertProfile(tx *sql.Tx, r profileRow) error {
	res, err := tx.Exec(`INSERT INTO profiles (identity, measured_at_ms) VALUES (?, ?)`, r.Identity, r.Profile.MeasuredAt.UnixMilli())
	if err != nil {
		return err
	}
	id, err := res.LastInsertId()
	if err != nil {
		return err
	}
	for _, d := range r.Profile.Devices {
		if _, err := tx.Exec(`INSERT INTO profile_devices VALUES (?,?,?,?,?,?)`,
			id, d.PCIID, int64(d.BandwidthBytesPerSec), d.TokenOverheadMs, d.LayerOverheadUs, d.FitErrorPct); err != nil {
			return err
		}
	}
	for _, l := range r.Profile.Links {
		for _, red := range l.Reductions {
			if _, err := tx.Exec(`INSERT INTO profile_links VALUES (?,?,?,?)`, id, strings.Join(l.PCIIDs, ","), red.Width, red.Us); err != nil {
				return err
			}
		}
	}
	for _, f := range r.Profile.Failures {
		if _, err := tx.Exec(`INSERT INTO profile_failures VALUES (?,?,?,?)`, id, f.What, strings.Join(f.PCIIDs, ","), f.Error); err != nil {
			return err
		}
	}
	return nil
}

// loadProfiles returns the latest measurement of each machine identity.
func (d *serverDB) loadProfiles() (map[string]*storedProfile, error) {
	out := map[string]*storedProfile{}
	rows, err := d.db.Query(`SELECT id, identity, measured_at_ms FROM profiles p
		WHERE measured_at_ms = (SELECT max(measured_at_ms) FROM profiles q WHERE q.identity = p.identity)`)
	if err != nil {
		return nil, err
	}
	ids := map[int64]*storedProfile{}
	for rows.Next() {
		var id, at int64
		var identity string
		if err := rows.Scan(&id, &identity, &at); err != nil {
			rows.Close()
			return nil, err
		}
		p := &storedProfile{MeasuredAt: time.UnixMilli(at).UTC()}
		out[identity], ids[id] = p, p
	}
	rows.Close()

	for id, p := range ids {
		if err := d.loadProfileParts(id, p); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (d *serverDB) loadProfileParts(id int64, p *storedProfile) error {
	rows, err := d.db.Query(`SELECT pci_id, bandwidth_bytes_per_sec, token_overhead_ms, layer_overhead_us, fit_error_pct
		FROM profile_devices WHERE profile_id = ? ORDER BY pci_id`, id)
	if err != nil {
		return err
	}
	for rows.Next() {
		var dev api.ProfileDevice
		var bw int64
		if err := rows.Scan(&dev.PCIID, &bw, &dev.TokenOverheadMs, &dev.LayerOverheadUs, &dev.FitErrorPct); err != nil {
			rows.Close()
			return err
		}
		dev.BandwidthBytesPerSec = uint64(bw)
		p.Devices = append(p.Devices, dev)
	}
	rows.Close()

	rows, err = d.db.Query(`SELECT pci_ids, width, us FROM profile_links WHERE profile_id = ? ORDER BY rowid`, id)
	if err != nil {
		return err
	}
	byPCIs := map[string]int{}
	for rows.Next() {
		var pcis string
		var red api.LinkReduction
		if err := rows.Scan(&pcis, &red.Width, &red.Us); err != nil {
			rows.Close()
			return err
		}
		i, ok := byPCIs[pcis]
		if !ok {
			i = len(p.Links)
			byPCIs[pcis] = i
			p.Links = append(p.Links, api.ProfileLink{PCIIDs: strings.Split(pcis, ",")})
		}
		p.Links[i].Reductions = append(p.Links[i].Reductions, red)
	}
	rows.Close()

	rows, err = d.db.Query(`SELECT what, pci_ids, error FROM profile_failures WHERE profile_id = ? ORDER BY rowid`, id)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var f api.ProfileFailure
		var pcis sql.NullString
		if err := rows.Scan(&f.What, &pcis, &f.Error); err != nil {
			return err
		}
		if pcis.String != "" {
			f.PCIIDs = strings.Split(pcis.String, ",")
		}
		p.Failures = append(p.Failures, f)
	}
	return rows.Err()
}

// importProfiles files the measurements of a box-profile.json.
func (d *serverDB) importProfiles(path string) (int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	var profiles map[string]*storedProfile
	if err := json.Unmarshal(b, &profiles); err != nil {
		return 0, err
	}
	tx, err := d.db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback() //nolint:errcheck
	for id, p := range profiles {
		if p == nil {
			continue
		}
		for i := range p.Devices {
			p.Devices[i].ID = "" // ids are reported from the devices present, never stored
		}
		if err := insertProfile(tx, profileRow{Identity: id, Profile: *p}); err != nil {
			return 0, err
		}
	}
	return len(profiles), tx.Commit()
}
