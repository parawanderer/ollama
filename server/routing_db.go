package server

import (
	"database/sql"
	"encoding/json"
	"time"

	"github.com/ollama/ollama/api"
)

// Mixture-of-experts routing in the server database: what each runner recorded of the routing it
// actually did, while it served (llama-server's GET /routing, LLAMA_ROUTING_STATS in the engine
// fork).
//
// Routing skew is the one input to a cost model for a mixture of experts that is not in a GGUF
// header. It could be measured before this only by taking the box for a benchmark, which measures
// the prompt the benchmark was given; this measures the traffic the model actually got.
//
// One row per runner life, replaced as it goes. The counts only grow over a runner's life, so the
// newest snapshot of a runner supersedes every earlier one and there is nothing to accumulate
// across rows. Writing periodically rather than only at unload is what makes it survive the way
// runners actually end here: killed, or the whole box power-cycled.
//
// What it never holds: anything about the tokens. Counts per expert per layer, and nothing that
// could say what was asked or answered.
const routingSchema = `
CREATE TABLE IF NOT EXISTS routing_stats (
	runner TEXT PRIMARY KEY,
	at_ms INTEGER NOT NULL,
	model TEXT NOT NULL,
	n_expert INTEGER NOT NULL,
	n_expert_used INTEGER NOT NULL,
	n_ubatch_seen INTEGER NOT NULL,
	prefill_tokens INTEGER NOT NULL,
	decode_tokens INTEGER NOT NULL,
	stats TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS routing_stats_model ON routing_stats(model, at_ms);
`

// routingSnapshot is one runner's record at one moment. Runner identifies a runner life, not a
// model: a model loaded twice is two rows, which is the honest unit, since the counts restart
// with the process.
type routingSnapshot struct {
	At     time.Time
	Runner string
	Model  string
	Stats  *api.RoutingStats
}

func insertRoutingSnapshot(tx *sql.Tx, s routingSnapshot) error {
	if s.Stats == nil {
		return nil
	}
	// The whole payload is kept as it arrived rather than spread over columns. The per-layer
	// summaries are what a fit consumes and the pooled counts are what a popularity profile
	// needs, and which of those matters is not settled -- so neither is thrown away, and the
	// columns carry only what is worth indexing on.
	blob, err := json.Marshal(s.Stats)
	if err != nil {
		return err
	}
	_, err = tx.Exec(`INSERT INTO routing_stats
		(runner, at_ms, model, n_expert, n_expert_used, n_ubatch_seen, prefill_tokens, decode_tokens, stats)
		VALUES (?,?,?,?,?,?,?,?,?)
		ON CONFLICT(runner) DO UPDATE SET
			at_ms = excluded.at_ms, n_ubatch_seen = excluded.n_ubatch_seen,
			prefill_tokens = excluded.prefill_tokens, decode_tokens = excluded.decode_tokens,
			stats = excluded.stats`,
		s.Runner, s.At.UnixMilli(), s.Model, s.Stats.NumExpert, s.Stats.NumExpertUsed,
		s.Stats.UBatchesSeen, s.Stats.Prefill.Tokens, s.Stats.Decode.Tokens, string(blob))
	return err
}

func (d *serverDB) recordRoutingSnapshot(s routingSnapshot) { d.enqueue(s) }
