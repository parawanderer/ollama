package server

import (
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// serverDB is the one SQLite file, beside the models, holding everything the server has
// measured or recorded:
//   - usage: one row per generation and per load (usage.go)
//   - calibration: every memory sample and every retraction, in order (calibration_db.go)
//   - device names: what each card was called while healthy (device_names.go)
//   - the box profile: every measurement of the machine's speed (box_profile.go)
//
// All of it is re-measurable. If the file cannot be opened the server runs exactly the same
// from memory and measures again after a restart, so an unusable database costs time, never
// correctness.
//
// Writes go through a buffered channel to one goroutine that commits in batches, so neither a
// request nor a load ever waits on the disk. If the buffer is full a row is dropped and counted
// rather than blocking the scheduler. Reads happen once, at startup, before the writer starts.
//
// What it never holds: anything a person wrote. Usage rows have counts, timings, the caller's
// hint and a salted client hash, never a prompt or a reply.
type serverDB struct {
	db      *sql.DB
	rows    chan any
	dropped atomic.Int64
	done    chan struct{}
	closeMu sync.Once
}

// serverDBName is the file, and serverDBPrevious the name it had while it held only usage.
const (
	serverDBName     = "slop.db"
	serverDBPrevious = "usage.db"
)

const serverDBSchema = `
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
` + usageSchema + calibrationSchema + deviceNamesSchema + profileSchema

const serverDBBuffer = 4096

// openServerDB opens the database in dir, creating it if needed, and carries over what the
// server kept before it had one: the usage database under its old name, and the JSON stores,
// each imported once.
func openServerDB(dir string) (*serverDB, error) {
	path := filepath.Join(dir, serverDBName)
	if err := adoptPreviousDB(dir); err != nil {
		slog.Warn("could not rename the usage database; starting a new one", "error", err)
	}
	db, err := sql.Open("sqlite3", path+"?_journal_mode=WAL&_busy_timeout=5000&_synchronous=NORMAL")
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(serverDBSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("database schema: %w", err)
	}
	if err := addMissingColumns(db, "generations", usageAddedColumns); err != nil {
		db.Close()
		return nil, fmt.Errorf("database schema: %w", err)
	}
	d := &serverDB{db: db, rows: make(chan any, serverDBBuffer), done: make(chan struct{})}
	if salt, err := d.salt(); err == nil {
		clientSalt.Store(salt)
	}
	d.importJSON(dir)
	go d.writer()
	return d, nil
}

// adoptPreviousDB renames usage.db to slop.db when only the old one exists. Its write-ahead log
// is folded into the main file first, because the log is found by name and would be left behind
// by a rename of the main file alone.
func adoptPreviousDB(dir string) error {
	path, prev := filepath.Join(dir, serverDBName), filepath.Join(dir, serverDBPrevious)
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	if _, err := os.Stat(prev); err != nil {
		return nil
	}
	old, err := sql.Open("sqlite3", prev+"?_busy_timeout=5000")
	if err != nil {
		return err
	}
	_, err = old.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
	old.Close()
	if err != nil {
		return err
	}
	if err := os.Rename(prev, path); err != nil {
		return err
	}
	for _, suffix := range []string{"-wal", "-shm"} {
		if err := os.Remove(prev + suffix); err != nil && !errors.Is(err, os.ErrNotExist) {
			slog.Debug("could not remove the old database's side file", "file", prev+suffix, "error", err)
		}
	}
	slog.Info("the usage database now holds everything the server measures", "from", prev, "to", path)
	return nil
}

// importJSON loads each JSON store the server wrote before this database existed, once. The
// files are left where they are and never written again.
func (d *serverDB) importJSON(dir string) {
	for _, imp := range []struct {
		file string
		fn   func(path string) (int, error)
	}{
		{"vram-calibration.json", d.importCalibration},
		{"device-names.json", d.importDeviceNames},
		{"box-profile.json", d.importProfiles},
	} {
		key := "imported:" + imp.file
		var done string
		if d.db.QueryRow(`SELECT value FROM meta WHERE key = ?`, key).Scan(&done) == nil {
			continue
		}
		path := filepath.Join(dir, imp.file)
		n, err := 0, error(nil)
		if _, statErr := os.Stat(path); statErr == nil {
			n, err = imp.fn(path)
		}
		if err != nil {
			slog.Warn("could not import a store into the database; its contents will be measured again", "file", path, "error", err)
		} else if n > 0 {
			slog.Info("imported a store into the database", "file", path, "rows", n)
		}
		_, _ = d.db.Exec(`INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)`, key, time.Now().UTC().Format(time.RFC3339))
	}
}

// salt returns this server's client-hash salt, creating it once. Persisted so a client's id is
// stable across restarts; random so it means nothing on any other server.
func (d *serverDB) salt() (string, error) {
	var s string
	err := d.db.QueryRow(`SELECT value FROM meta WHERE key = 'client_salt'`).Scan(&s)
	if err == nil {
		return s, nil
	}
	s = randomHex(16)
	_, err = d.db.Exec(`INSERT OR IGNORE INTO meta (key, value) VALUES ('client_salt', ?)`, s)
	if err != nil {
		return "", err
	}
	return s, d.db.QueryRow(`SELECT value FROM meta WHERE key = 'client_salt'`).Scan(&s)
}

func (d *serverDB) enqueue(row any) {
	if d == nil {
		return
	}
	select {
	case d.rows <- row:
	default:
		if d.dropped.Add(1)%1000 == 1 {
			slog.Warn("the database is behind; dropping rows", "dropped", d.dropped.Load())
		}
	}
}

func (d *serverDB) writer() {
	defer close(d.done)
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	var batch []any
	flush := func() {
		if len(batch) == 0 {
			return
		}
		if err := d.write(batch); err != nil {
			slog.Warn("could not write to the database", "rows", len(batch), "error", err)
		}
		batch = batch[:0]
	}
	for {
		select {
		case row, ok := <-d.rows:
			if !ok {
				flush()
				return
			}
			batch = append(batch, row)
			if len(batch) >= 256 {
				flush()
			}
		case <-ticker.C:
			flush()
		}
	}
}

func (d *serverDB) write(batch []any) error {
	tx, err := d.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback() //nolint:errcheck
	for _, row := range batch {
		switch r := row.(type) {
		case usageGeneration:
			err = insertGeneration(tx, r)
		case usageLoad:
			err = insertLoad(tx, r)
		case calibrationEvent:
			err = insertCalibrationEvent(tx, r)
		case seenDeviceRow:
			err = upsertDeviceName(tx, r)
		case profileRow:
			err = insertProfile(tx, r)
		default:
			err = fmt.Errorf("no table for %T", row)
		}
		if err != nil {
			return err
		}
	}
	return tx.Commit()
}

// close flushes what is buffered and closes the database.
func (d *serverDB) close() {
	if d == nil {
		return
	}
	d.closeMu.Do(func() {
		close(d.rows)
		<-d.done
		_, _ = d.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
		d.db.Close()
	})
}

// addMissingColumns adds each column the table lacks. CREATE TABLE IF NOT EXISTS leaves an
// existing table as it was, so a database from an earlier build gains new columns here. Append
// only: a column is never renamed or dropped, so old rows stay readable.
func addMissingColumns(db *sql.DB, table string, cols []struct{ name, decl string }) error {
	rows, err := db.Query(`SELECT name FROM pragma_table_info(?)`, table)
	if err != nil {
		return err
	}
	have := map[string]bool{}
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			rows.Close()
			return err
		}
		have[name] = true
	}
	rows.Close()
	for _, c := range cols {
		if !have[c.name] {
			if _, err := db.Exec(fmt.Sprintf(`ALTER TABLE %s ADD COLUMN %s %s`, table, c.name, c.decl)); err != nil {
				return err
			}
		}
	}
	return nil
}
