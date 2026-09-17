package server

import (
	"context"
	"database/sql"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func routingStatsFixture(prefillTokens int64, busiest float64) *api.RoutingStats {
	return &api.RoutingStats{
		Enabled: true, NumExpert: 64, NumExpertUsed: 6,
		UBatchesSeen: 122, ReadMicros: 377,
		Prefill: api.RoutingPopulation{
			UBatchesRecorded: 4, Tokens: prefillTokens,
			Layers: []api.RoutingLayer{
				{Layer: 1, Tokens: prefillTokens, Batches: 4, Touched: 0.99,
					EffExperts: 43.7, Busiest: busiest, Counts: []int64{3, 1, 0, 9}},
				// the last layer routes only the tokens whose output is needed
				{Layer: 26, Tokens: 4, Batches: 4, Touched: 0.09,
					EffExperts: 6, Busiest: 10.67, Counts: []int64{0, 0, 4, 0}},
			},
		},
		Decode: api.RoutingPopulation{
			UBatchesRecorded: 31, Tokens: 31,
			Layers: []api.RoutingLayer{{Layer: 1, Tokens: 31, Batches: 31, Busiest: 10.67,
				Counts: []int64{5, 2, 1, 0}}},
		},
	}
}

func readRoutingRow(t *testing.T, dir, runner string) (model string, prefill int64, stats api.RoutingStats) {
	t.Helper()
	db, err := sql.Open("sqlite3", filepath.Join(dir, serverDBName)+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var blob string
	err = db.QueryRow(`SELECT model, prefill_tokens, stats FROM routing_stats WHERE runner = ?`, runner).
		Scan(&model, &prefill, &blob)
	if err != nil {
		t.Fatalf("reading routing row %q: %v", runner, err)
	}
	if err := json.Unmarshal([]byte(blob), &stats); err != nil {
		t.Fatal(err)
	}
	return model, prefill, stats
}

// A runner's counts only grow, so a later snapshot supersedes an earlier one rather than adding to
// it. If it appended instead, the table would carry a row per minute per runner and a reader
// summing them would count the same tokens many times over.
func TestRoutingSnapshotsReplaceRatherThanAccumulate(t *testing.T) {
	dir := t.TempDir()
	d := testServerDB(t, dir)

	for _, tokens := range []int64{512, 1024, 2048} {
		d.recordRoutingSnapshot(routingSnapshot{
			At: time.Now(), Runner: "4242@1000", Model: "deepseek-v2:16b",
			Stats: routingStatsFixture(tokens, 3.87),
		})
	}
	// a second life of the same model is a different runner, and its own row
	d.recordRoutingSnapshot(routingSnapshot{
		At: time.Now(), Runner: "4243@2000", Model: "deepseek-v2:16b",
		Stats: routingStatsFixture(99, 4.10),
	})
	d.close()

	if n := countRows(t, dir, "routing_stats"); n != 2 {
		t.Errorf("routing_stats has %d rows, want 2: one per runner life", n)
	}
	if _, prefill, _ := readRoutingRow(t, dir, "4242@1000"); prefill != 2048 {
		t.Errorf("prefill_tokens = %d, want the newest snapshot's 2048", prefill)
	}
}

// The payload is kept whole: the per-layer summaries a fit consumes and the pooled counts a
// popularity profile needs are both in it, and which of those matters is not settled.
func TestRoutingSnapshotKeepsTheWholePayload(t *testing.T) {
	dir := t.TempDir()
	d := testServerDB(t, dir)
	d.recordRoutingSnapshot(routingSnapshot{
		At: time.Now(), Runner: "7@7", Model: "qwen3-vl:30b", Stats: routingStatsFixture(512, 3.87),
	})
	d.close()

	model, _, got := readRoutingRow(t, dir, "7@7")
	if model != "qwen3-vl:30b" {
		t.Errorf("model = %q", model)
	}
	if got.NumExpert != 64 || got.NumExpertUsed != 6 || got.UBatchesSeen != 122 {
		t.Errorf("header fields lost: %+v", got)
	}
	if len(got.Prefill.Layers) != 2 || len(got.Decode.Layers) != 1 {
		t.Fatalf("layers lost: prefill %d, decode %d", len(got.Prefill.Layers), len(got.Decode.Layers))
	}
	first := got.Prefill.Layers[0]
	if first.Busiest != 3.87 || first.EffExperts != 43.7 || first.Touched != 0.99 {
		t.Errorf("per-micro-batch summaries lost: %+v", first)
	}
	if want := []int64{3, 1, 0, 9}; len(first.Counts) != len(want) || first.Counts[3] != 9 {
		t.Errorf("pooled counts lost: %v", first.Counts)
	}
	// The last layer's own token total is what a consumer divides by; a global one would make it
	// look like the model's least skewed layer.
	if last := got.Prefill.Layers[1]; last.Tokens != 4 {
		t.Errorf("the last layer's own token total = %d, want 4", last.Tokens)
	}
	if got.Prefill.Tokens == got.Prefill.Layers[1].Tokens {
		t.Error("the population total and the last layer's total must stay separate")
	}
}

// routingRecorder is a runner that reports routing and counts how often it is asked.
type routingRecorder struct {
	mockLlm
	stats *api.RoutingStats
	asked int
}

func (r *routingRecorder) RoutingStats(ctx context.Context) *api.RoutingStats {
	r.asked++
	return r.stats
}

func TestOnlyAMixtureOfExpertsIsAskedForRouting(t *testing.T) {
	dir := t.TempDir()
	d := testServerDB(t, dir)
	defer d.close()

	dense := &routingRecorder{stats: routingStatsFixture(512, 3.87)}
	runner := &runnerRef{db: d, llama: dense, name: "llama3.2:3b", expertCount: 0,
		routingStop: make(chan struct{})}
	runner.snapshotRouting(time.Second)
	if dense.asked != 0 {
		t.Errorf("a model that is not a mixture of experts was asked %d times", dense.asked)
	}

	moe := &routingRecorder{stats: routingStatsFixture(512, 3.87)}
	runner = &runnerRef{db: d, llama: moe, name: "deepseek-v2:16b", expertCount: 64,
		routingStop: make(chan struct{}), pid: 11, loadStarted: time.UnixMilli(22)}
	runner.snapshotRouting(time.Second)
	if moe.asked != 1 {
		t.Errorf("a mixture of experts was asked %d times, want 1", moe.asked)
	}
	if key := runner.routingKey(); key != "11@22" {
		t.Errorf("routingKey = %q, want the pid and load time", key)
	}
}

// A runner whose engine reports nothing writes nothing, rather than a row of zeroes that a reader
// could not tell from a model that genuinely routed nowhere.
func TestARunnerThatReportsNothingWritesNothing(t *testing.T) {
	dir := t.TempDir()
	d := testServerDB(t, dir)

	silent := &routingRecorder{stats: nil}
	runner := &runnerRef{db: d, llama: silent, name: "deepseek-v2:16b", expertCount: 64,
		routingStop: make(chan struct{}), pid: 1, loadStarted: time.UnixMilli(1)}
	runner.snapshotRouting(time.Second)
	d.close()

	if silent.asked != 1 {
		t.Errorf("asked %d times, want 1", silent.asked)
	}
	if n := countRows(t, dir, "routing_stats"); n != 0 {
		t.Errorf("routing_stats has %d rows, want none", n)
	}
}

// unload() must stop the poller; a goroutine per runner life that outlives its runner would leak
// one per load for as long as the server runs.
func TestUnloadStopsTheRoutingPoller(t *testing.T) {
	runner := &runnerRef{expertCount: 64, routingStop: make(chan struct{})}
	runner.unload()
	select {
	case <-runner.routingStop:
	default:
		t.Error("unload did not stop the routing poller")
	}
	runner.unload() // a second unload must not panic on an already-closed channel
}
