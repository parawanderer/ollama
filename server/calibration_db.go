package server

import (
	"database/sql"
	"encoding/json"
	"os"
	"time"

	"github.com/ollama/ollama/llm"
)

// Memory calibration in the server database: every sample the calibration was given and every
// time a key was forgotten, in order. Replaying them rebuilds exactly the calibration the server
// had, including the cap on samples per key and a probe round retracted for yielding fewer than
// two points. Nothing is overwritten, so what a prediction had to go on can be read afterwards.
//
// The calibration itself (llm.VRAMCalibration) keeps its own JSON Load and Save, unused here.

const calibrationSchema = `
CREATE TABLE IF NOT EXISTS calibration_events (
	id INTEGER PRIMARY KEY,
	at_ms INTEGER NOT NULL,
	op TEXT NOT NULL,
	model TEXT NOT NULL,
	key TEXT NOT NULL,
	num_ctx INTEGER, vram INTEGER, source TEXT
);
CREATE INDEX IF NOT EXISTS calibration_events_key ON calibration_events(key, id);
`

// calibrationEvent is one row: op is "record" (a sample, with where it came from: "probe",
// "load" or "imported") or "forget" (every sample of the key retracted).
type calibrationEvent struct {
	At     time.Time
	Op     string
	Key    llm.CalibrationKey
	NumCtx int
	VRAM   uint64
	Source string
}

func insertCalibrationEvent(tx *sql.Tx, e calibrationEvent) error {
	key, err := json.Marshal(e.Key)
	if err != nil {
		return err
	}
	var numCtx, vram, source any
	if e.Op == "record" {
		numCtx, vram, source = e.NumCtx, int64(e.VRAM), nullIfEmpty(e.Source)
	}
	_, err = tx.Exec(`INSERT INTO calibration_events (at_ms, op, model, key, num_ctx, vram, source) VALUES (?,?,?,?,?,?,?)`,
		e.At.UnixMilli(), e.Op, e.Key.Model, string(key), numCtx, vram, source)
	return err
}

// loadCalibration replays every event into c, oldest first.
func (d *serverDB) loadCalibration(c *llm.VRAMCalibration) error {
	if d == nil {
		return nil
	}
	rows, err := d.db.Query(`SELECT op, key, num_ctx, vram FROM calibration_events ORDER BY id`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var op, keyJSON string
		var numCtx, vram sql.NullInt64
		if err := rows.Scan(&op, &keyJSON, &numCtx, &vram); err != nil {
			return err
		}
		var key llm.CalibrationKey
		if err := json.Unmarshal([]byte(keyJSON), &key); err != nil {
			continue
		}
		switch op {
		case "record":
			c.Record(key, int(numCtx.Int64), uint64(vram.Int64))
		case "forget":
			c.Forget(key)
		}
	}
	return rows.Err()
}

// importCalibration files the samples of a vram-calibration.json as records, dated when that
// file was written.
func (d *serverDB) importCalibration(path string) (int, error) {
	c := llm.NewVRAMCalibration()
	c.Load(path)
	at := time.Now()
	var file struct {
		Written time.Time `json:"written"`
	}
	if b, err := os.ReadFile(path); err == nil && json.Unmarshal(b, &file) == nil && !file.Written.IsZero() {
		at = file.Written
	}
	tx, err := d.db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback() //nolint:errcheck
	n := 0
	var ierr error
	c.Each(func(key llm.CalibrationKey, numCtx int, vram uint64) {
		if ierr == nil {
			ierr = insertCalibrationEvent(tx, calibrationEvent{At: at, Op: "record", Key: key, NumCtx: numCtx, VRAM: vram, Source: "imported"})
			n++
		}
	})
	if ierr != nil {
		return 0, ierr
	}
	return n, tx.Commit()
}

// recordCalibration gives the calibration a sample and files it. source is "probe" or "load".
func (s *Scheduler) recordCalibration(key llm.CalibrationKey, numCtx int, vram uint64, source string) {
	if numCtx <= 0 || vram == 0 {
		return
	}
	s.vramCalibration.Record(key, numCtx, vram)
	s.db.enqueue(calibrationEvent{At: time.Now(), Op: "record", Key: key, NumCtx: numCtx, VRAM: vram, Source: source})
}

// forgetCalibration retracts every sample of key, and files the retraction.
func (s *Scheduler) forgetCalibration(key llm.CalibrationKey) {
	s.vramCalibration.Forget(key)
	s.db.enqueue(calibrationEvent{At: time.Now(), Op: "forget", Key: key})
}
