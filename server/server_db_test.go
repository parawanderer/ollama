package server

import (
	"database/sql"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func testServerDB(t *testing.T, dir string) *serverDB {
	t.Helper()
	d, err := openServerDB(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.close)
	return d
}

func countRows(t *testing.T, dir, table string) int {
	t.Helper()
	db, err := sql.Open("sqlite3", filepath.Join(dir, serverDBName)+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var n int
	if err := db.QueryRow(`SELECT count(*) FROM ` + table).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// The old usage database is adopted under the new name, including rows that were still only
// in its write-ahead log. Renaming the main file alone would lose exactly those: the log is
// found by name, and it is the newest data.
func TestServerDBAdoptsTheUsageDatabaseWithItsLog(t *testing.T) {
	live := t.TempDir()
	old, err := sql.Open("sqlite3", filepath.Join(live, serverDBPrevious)+"?_journal_mode=WAL")
	if err != nil {
		t.Fatal(err)
	}
	old.SetMaxOpenConns(1)
	if _, err := old.Exec(`PRAGMA wal_autocheckpoint=0;` + firstUsageSchema +
		`INSERT INTO generations (at_ms, model) VALUES (1, 'only-in-the-log')`); err != nil {
		t.Fatal(err)
	}
	// Copy the files while the connection is open, as a server killed mid-flight leaves them.
	dir := t.TempDir()
	for _, suffix := range []string{"", "-wal"} {
		copyTestFile(t, filepath.Join(live, serverDBPrevious+suffix), filepath.Join(dir, serverDBPrevious+suffix))
	}
	old.Close()
	if fi, err := os.Stat(filepath.Join(dir, serverDBPrevious+"-wal")); err != nil || fi.Size() == 0 {
		t.Fatalf("the fixture has no log to carry over (%v); the test proves nothing", err)
	}

	testServerDB(t, dir).close()
	if _, err := os.Stat(filepath.Join(dir, serverDBPrevious)); !os.IsNotExist(err) {
		t.Error("the old database is still there under its old name")
	}
	if n := countRows(t, dir, "generations WHERE model = 'only-in-the-log'"); n != 1 {
		t.Errorf("rows from the log after the rename: %d, want 1", n)
	}
}

func copyTestFile(t *testing.T, from, to string) {
	t.Helper()
	in, err := os.Open(from)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	out, err := os.Create(to)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	if _, err := io.Copy(out, in); err != nil {
		t.Fatal(err)
	}
}

// Replaying the event log rebuilds the calibration the server had: a retracted key stays
// retracted, and a key's samples are the same ones, not every sample ever filed.
func TestCalibrationSurvivesARestartAsItWas(t *testing.T) {
	dir := t.TempDir()
	s := &Scheduler{vramCalibration: llm.NewVRAMCalibration(), db: testServerDB(t, dir)}
	kept := llm.CalibrationKey{Model: "kept", NumBatch: 512}
	retracted := llm.CalibrationKey{Model: "retracted", NumBatch: 512}
	for ctx := 1; ctx <= 10; ctx++ { // more than the calibration keeps per key
		s.recordCalibration(kept, ctx*1024, uint64(ctx)<<30, "load")
	}
	s.recordCalibration(retracted, 8192, 5<<30, "probe")
	s.forgetCalibration(retracted)
	s.db.close()

	restarted := llm.NewVRAMCalibration()
	if err := testServerDB(t, dir).loadCalibration(restarted); err != nil {
		t.Fatal(err)
	}
	if got, want := restarted.SampleCount(kept), s.vramCalibration.SampleCount(kept); got != want {
		t.Errorf("kept key: %d samples after restart, %d before", got, want)
	}
	if n := restarted.SampleCount(retracted); n != 0 {
		t.Errorf("a retracted key came back with %d samples", n)
	}
	a, _ := s.vramCalibration.Predict(kept, 3000, 0, 0)
	b, _ := restarted.Predict(kept, 3000, 0, 0)
	if a != b {
		t.Errorf("prediction %d after restart, %d before", b, a)
	}
	if n := countRows(t, dir, "calibration_events"); n != 12 {
		t.Errorf("%d events filed, want every one of the 12: history is kept, not overwritten", n)
	}
}

// Each JSON store is imported once. A second start must not file its contents again.
func TestServerDBImportsTheJSONStoresOnce(t *testing.T) {
	dir := t.TempDir()
	cal := llm.NewVRAMCalibration()
	key := llm.CalibrationKey{Model: "m", NumBatch: 2048, NumGPU: 1}
	cal.Record(key, 8192, 10<<30)
	cal.Record(key, 131072, 20<<30)
	if err := cal.Save(filepath.Join(dir, "vram-calibration.json")); err != nil {
		t.Fatal(err)
	}
	names, _ := json.Marshal(map[string]seenDevice{"0000:03:00.0": {Name: "CUDA1", Library: "CUDA", LastSeen: time.Unix(1_800_000_000, 0)}})
	os.WriteFile(filepath.Join(dir, "device-names.json"), names, 0o600)
	prof, _ := json.Marshal(map[string]*storedProfile{"abc": {
		MeasuredAt: time.Unix(1_800_000_000, 0).UTC(),
		Devices:    []api.ProfileDevice{{PCIID: "0000:01:00.0", BandwidthBytesPerSec: 1_607_000_000_000, LayerOverheadUs: 25}},
		Links:      []api.ProfileLink{{PCIIDs: []string{"0000:01:00.0", "0000:03:00.0"}, Reductions: []api.LinkReduction{{Width: 4096, Us: 11.3}, {Width: 8192, Us: 16.2}}}},
	}})
	os.WriteFile(filepath.Join(dir, "box-profile.json"), prof, 0o644)

	for range 2 {
		d := testServerDB(t, dir)
		c := llm.NewVRAMCalibration()
		if err := d.loadCalibration(c); err != nil {
			t.Fatal(err)
		}
		if c.SampleCount(key) != 2 {
			t.Errorf("calibration after import: %d samples, want 2", c.SampleCount(key))
		}
		if s, ok := newDeviceNames(d).lookup("0000:03:00.0"); !ok || s.Name != "CUDA1" {
			t.Errorf("device names after import: %+v %v", s, ok)
		}
		p := newBoxProfiler(d)
		if got := p.profiles["abc"]; got == nil || len(got.Links) != 1 || len(got.Links[0].Reductions) != 2 {
			t.Errorf("profile after import: %+v", got)
		}
		d.close()
	}
	for table, want := range map[string]int{"calibration_events": 2, "device_names": 1, "profiles": 1} {
		if n := countRows(t, dir, table); n != want {
			t.Errorf("%s: %d rows after two starts, want %d", table, n, want)
		}
	}
}

// Without a database everything still works, from memory.
func TestNoDatabaseMeansMemoryOnly(t *testing.T) {
	var db *serverDB
	s := &Scheduler{vramCalibration: llm.NewVRAMCalibration(), db: db}
	s.recordCalibration(llm.CalibrationKey{Model: "m"}, 8192, 1<<30, "load")
	if s.vramCalibration.SampleCount(llm.CalibrationKey{Model: "m"}) != 1 {
		t.Error("the calibration lost a sample because there was no database")
	}
	newDeviceNames(db).observe([]ml.DeviceInfo{cudaDev("0000:01:00.0", "CUDA0")})
	newBoxProfiler(db)
	db.recordGeneration(usageGeneration{})
	db.close()
}
