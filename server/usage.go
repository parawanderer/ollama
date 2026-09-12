package server

import (
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	_ "github.com/mattn/go-sqlite3"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

// usageStore keeps one row per finished generation and one per load, in SQLite beside the
// models. This is the high-volume record: the calibration store holds at most eight samples
// per key and stays JSON, while this grows with every request, so it gets a database.
//
// What it is for: the learned keep-alive and the per-model speed store need the history of
// what was requested, how it was served and how long it took, from every client. The event
// stream carries the same facts but keeps ten minutes of them.
//
// What it never holds: anything a person wrote. A row has counts, timings, the caller's hint
// and a salted client hash, never a prompt or a reply.
//
// Writes go through a buffered channel to one goroutine that commits in batches, so a request
// never waits on the disk. If the buffer is full the row is dropped and counted rather than
// blocking the scheduler.
type usageStore struct {
	db      *sql.DB
	rows    chan any
	dropped atomic.Int64
	done    chan struct{}
	closeMu sync.Once
}

// usageGeneration is one finished generation.
type usageGeneration struct {
	At      time.Time
	Model   string
	Timings api.GenerationTimings
	Meta    *api.GenerationMeta
	// How the runner that served it was placed.
	Devices  string
	NumCtx   int
	NumBatch int
	Split    string
}

// usageLoad is one completed load.
type usageLoad struct {
	At        time.Time
	Model     string
	Estimate  *api.LoadEstimate
	Devices   string
	Split     string
	SizeVRAM  int64
	SizeTotal int64
	WeightsMs int64
	ContextMs int64
	TotalMs   int64
}

const usageSchema = `
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS generations (
	id INTEGER PRIMARY KEY,
	at_ms INTEGER NOT NULL,
	model TEXT NOT NULL,
	prompt_tokens INTEGER, prompt_tokens_cached INTEGER, prompt_ms REAL, eval_ms REAL, decoded INTEGER,
	cache_swap_ms REAL,
	hint_use TEXT, hint_session TEXT, hint_request TEXT, hint_after TEXT, hint_synthetic INTEGER,
	endpoint TEXT, surface TEXT, stream INTEGER, messages INTEGER, images INTEGER, tools INTEGER,
	format INTEGER, think TEXT,
	req_num_ctx INTEGER, req_num_gpu INTEGER, req_num_predict INTEGER, req_keep_alive_s INTEGER,
	client TEXT,
	devices TEXT, num_ctx INTEGER, num_batch INTEGER, split TEXT
);
CREATE INDEX IF NOT EXISTS generations_model_at ON generations(model, at_ms);
CREATE INDEX IF NOT EXISTS generations_session_at ON generations(hint_session, at_ms);
CREATE TABLE IF NOT EXISTS loads (
	id INTEGER PRIMARY KEY,
	at_ms INTEGER NOT NULL,
	model TEXT NOT NULL,
	predicted INTEGER, predicted_for_load INTEGER, source TEXT,
	num_ctx INTEGER, num_gpu INTEGER, num_batch INTEGER,
	devices TEXT, split TEXT,
	size_vram INTEGER, size_total INTEGER,
	weights_ms INTEGER, context_ms INTEGER, total_ms INTEGER
);
CREATE INDEX IF NOT EXISTS loads_model_at ON loads(model, at_ms);
`

const usageBuffer = 4096

// openUsageStore opens (creating if needed) the database at path. A store that cannot be
// opened is not an error for the server: usage is recorded when possible, never required.
func openUsageStore(path string) (*usageStore, error) {
	db, err := sql.Open("sqlite3", path+"?_journal_mode=WAL&_busy_timeout=5000&_synchronous=NORMAL")
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(usageSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("usage schema: %w", err)
	}
	u := &usageStore{db: db, rows: make(chan any, usageBuffer), done: make(chan struct{})}
	if salt, err := u.salt(); err == nil {
		clientSalt.Store(salt)
	}
	go u.writer()
	return u, nil
}

// salt returns this server's client-hash salt, creating it once. Persisted so a client's id is
// stable across restarts; random so it means nothing on any other server.
func (u *usageStore) salt() (string, error) {
	var s string
	err := u.db.QueryRow(`SELECT value FROM meta WHERE key = 'client_salt'`).Scan(&s)
	if err == nil {
		return s, nil
	}
	s = randomHex(16)
	_, err = u.db.Exec(`INSERT OR IGNORE INTO meta (key, value) VALUES ('client_salt', ?)`, s)
	if err != nil {
		return "", err
	}
	return s, u.db.QueryRow(`SELECT value FROM meta WHERE key = 'client_salt'`).Scan(&s)
}

func (u *usageStore) recordGeneration(g usageGeneration) { u.enqueue(g) }
func (u *usageStore) recordLoad(l usageLoad)             { u.enqueue(l) }

func (u *usageStore) enqueue(row any) {
	if u == nil {
		return
	}
	select {
	case u.rows <- row:
	default:
		if u.dropped.Add(1)%1000 == 1 {
			slog.Warn("usage store is behind; dropping rows", "dropped", u.dropped.Load())
		}
	}
}

func (u *usageStore) writer() {
	defer close(u.done)
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	var batch []any
	flush := func() {
		if len(batch) == 0 {
			return
		}
		if err := u.write(batch); err != nil {
			slog.Warn("could not record usage", "rows", len(batch), "error", err)
		}
		batch = batch[:0]
	}
	for {
		select {
		case row, ok := <-u.rows:
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

func (u *usageStore) write(batch []any) error {
	tx, err := u.db.Begin()
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
		}
		if err != nil {
			return err
		}
	}
	return tx.Commit()
}

func insertGeneration(tx *sql.Tx, g usageGeneration) error {
	var hint api.RequestHint
	var shape api.RequestShape
	if g.Meta != nil && g.Meta.Hint != nil {
		hint = *g.Meta.Hint
	}
	if g.Meta != nil && g.Meta.Shape != nil {
		shape = *g.Meta.Shape
	}
	var swapMs any
	if g.Timings.PromptCacheSwap != nil {
		swapMs = g.Timings.PromptCacheSwap.Ms
	}
	_, err := tx.Exec(`INSERT INTO generations (
		at_ms, model, prompt_tokens, prompt_tokens_cached, prompt_ms, eval_ms, decoded, cache_swap_ms,
		hint_use, hint_session, hint_request, hint_after, hint_synthetic,
		endpoint, surface, stream, messages, images, tools, format, think,
		req_num_ctx, req_num_gpu, req_num_predict, req_keep_alive_s, client,
		devices, num_ctx, num_batch, split
	) VALUES (?,?,?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?)`,
		g.At.UnixMilli(), g.Model, g.Timings.PromptTokens, g.Timings.PromptTokensCached, g.Timings.PromptMs,
		g.Timings.EvalMs, g.Timings.Decoded, swapMs,
		nullIfEmpty(hint.Use), nullIfEmpty(hint.Session), nullIfEmpty(hint.Request), nullIfEmpty(hint.After), boolInt(hint.Synthetic),
		nullIfEmpty(shape.Endpoint), nullIfEmpty(shape.Surface), boolInt(shape.Stream), shape.Messages, shape.Images, shape.Tools,
		boolInt(shape.Format), nullIfEmpty(shape.Think),
		shape.NumCtx, shape.NumGPU, shape.NumPredict, shape.KeepAliveS, nullIfEmpty(shape.Client),
		nullIfEmpty(g.Devices), g.NumCtx, g.NumBatch, nullIfEmpty(g.Split))
	return err
}

func insertLoad(tx *sql.Tx, l usageLoad) error {
	var e api.LoadEstimate
	if l.Estimate != nil {
		e = *l.Estimate
	}
	_, err := tx.Exec(`INSERT INTO loads (
		at_ms, model, predicted, predicted_for_load, source, num_ctx, num_gpu, num_batch,
		devices, split, size_vram, size_total, weights_ms, context_ms, total_ms
	) VALUES (?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?)`,
		l.At.UnixMilli(), l.Model, e.Predicted, e.PredictedForLoad, nullIfEmpty(e.Source), e.NumCtx, e.NumGPU, e.NumBatch,
		nullIfEmpty(l.Devices), nullIfEmpty(l.Split), l.SizeVRAM, l.SizeTotal, l.WeightsMs, l.ContextMs, l.TotalMs)
	return err
}

// close flushes what is buffered and closes the database.
func (u *usageStore) close() {
	if u == nil {
		return
	}
	u.closeMu.Do(func() {
		close(u.rows)
		<-u.done
		_, _ = u.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
		u.db.Close()
	})
}

func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// clientSalt salts the client hash. The usage store replaces it with a persisted value; until
// then (or without a store) it is random per process.
var clientSalt atomic.Value

func init() { clientSalt.Store(randomHex(16)) }

// clientID is a pseudonymous id for whoever sent a request: stable on this server, meaningless
// on any other, and not reversible to an address.
func clientID(c *gin.Context) string {
	if c == nil || c.Request == nil || c.Request.Header == nil {
		return ""
	}
	sum := sha256.Sum256([]byte(clientSalt.Load().(string) + "|" + c.ClientIP() + "|" + c.Request.UserAgent()))
	return hex.EncodeToString(sum[:6])
}

func surfaceOf(c *gin.Context) string {
	if c != nil && c.Request != nil && c.Request.URL != nil && strings.HasPrefix(c.Request.URL.Path, "/v1/") {
		return "openai"
	}
	return "native"
}

// requestedInt reads an option the request itself set. JSON numbers arrive as float64.
func requestedInt(opts map[string]any, key string) *int {
	v, ok := opts[key]
	if !ok {
		return nil
	}
	var n int
	switch x := v.(type) {
	case float64:
		n = int(x)
	case int:
		n = x
	case int64:
		n = int(x)
	default:
		return nil
	}
	return &n
}

func keepAliveSeconds(d *api.Duration) *int64 {
	if d == nil {
		return nil
	}
	var s int64
	if d.Duration < 0 {
		s = -1
	} else {
		s = int64(math.Round(d.Duration.Seconds()))
	}
	return &s
}

func thinkOf(t *api.ThinkValue) string {
	if t == nil || t.Value == nil {
		return ""
	}
	return fmt.Sprint(t.Value)
}

func chatMeta(c *gin.Context, req api.ChatRequest) *api.GenerationMeta {
	images := 0
	for _, m := range req.Messages {
		images += len(m.Images)
	}
	return &api.GenerationMeta{
		Hint: req.Hint.Sanitized(),
		Shape: &api.RequestShape{
			Endpoint:   "chat",
			Surface:    surfaceOf(c),
			Stream:     req.Stream == nil || *req.Stream,
			Messages:   len(req.Messages),
			Images:     images,
			Tools:      len(req.Tools),
			Format:     len(req.Format) > 0,
			Think:      thinkOf(req.Think),
			NumCtx:     requestedInt(req.Options, "num_ctx"),
			NumGPU:     requestedInt(req.Options, "num_gpu"),
			NumPredict: requestedInt(req.Options, "num_predict"),
			KeepAliveS: keepAliveSeconds(req.KeepAlive),
			Client:     clientID(c),
		},
	}
}

func generateMeta(c *gin.Context, req api.GenerateRequest) *api.GenerationMeta {
	return &api.GenerationMeta{
		Hint: req.Hint.Sanitized(),
		Shape: &api.RequestShape{
			Endpoint:   "generate",
			Surface:    surfaceOf(c),
			Stream:     req.Stream == nil || *req.Stream,
			Images:     len(req.Images),
			Format:     len(req.Format) > 0,
			Think:      thinkOf(req.Think),
			NumCtx:     requestedInt(req.Options, "num_ctx"),
			NumGPU:     requestedInt(req.Options, "num_gpu"),
			NumPredict: requestedInt(req.Options, "num_predict"),
			KeepAliveS: keepAliveSeconds(req.KeepAlive),
			Client:     clientID(c),
		},
	}
}

// usageDevices names the devices a runner was placed on, "CUDA0,CUDA1"; empty for the CPU.
func usageDevices(gpus []ml.DeviceInfo) string {
	names := make([]string, 0, len(gpus))
	for _, g := range gpus {
		names = append(names, g.Library+g.ID)
	}
	return strings.Join(names, ",")
}

// usageSplit says how a runner was split: "cpu", "none" (one device) or "layer" (ollama never
// uses tensor split).
func usageSplit(gpus []ml.DeviceInfo, opts api.Options) string {
	switch {
	case len(gpus) == 0 || opts.NumGPU == 0:
		return "cpu"
	case len(gpus) == 1:
		return "none"
	default:
		return "layer"
	}
}
