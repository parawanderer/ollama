package server

import (
	"database/sql"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

var (
	decodeGPU0 = ml.DeviceID{ID: "0", Library: "CUDA"}
	decodeGPU1 = ml.DeviceID{ID: "1", Library: "CUDA"}
	// The profile this machine measured on 2026-09-13.
	measuredFit = api.ProfileDevice{BandwidthBytesPerSec: 1_607_000_000_000, TokenOverheadMs: 0.058, LayerOverheadUs: 25.1}
)

func near(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %.4f, want %.4f", name, got, want)
	}
}

// qwen3:32b on one card: 19.76 GB read per token over 64 layers. The notebook's prediction from
// the synthetic fit was 13.96 ms, against 14.49 measured; this must reproduce the arithmetic.
func TestExpectedDecodeOneDevice(t *testing.T) {
	in := decodeInputs{
		gpus:     []ml.DeviceID{decodeGPU0},
		memByGPU: map[ml.DeviceID]api.MemoryBreakdown{decodeGPU0: {Weights: 19_760_000_000, KVCache: 1 << 30}},
		fits:     map[ml.DeviceID]api.ProfileDevice{decodeGPU0: measuredFit},
		layers:   64, activeWeights: 1, grantedCtxTotal: 4096,
	}
	base, perCtx, ok, why := expectedDecode(in)
	if why != "" || !ok {
		t.Fatalf("unavailable %q ok=%v", why, ok)
	}
	near(t, "base ms", base, 0.058+64*0.0251+19.76e9/1.607e9, 1e-6)
	near(t, "per-context ms", perCtx, float64(1<<30)/4096/1.607e9, 1e-12)
}

// A layer split visits its devices in turn: the byte terms add, the per-token overhead is paid
// once, and the per-layer overhead follows where the layers are.
func TestExpectedDecodeSplitAddsDevicesAndPaysTheTokenOverheadOnce(t *testing.T) {
	slow := measuredFit
	slow.TokenOverheadMs = 0.2
	in := decodeInputs{
		gpus: []ml.DeviceID{decodeGPU0, decodeGPU1},
		memByGPU: map[ml.DeviceID]api.MemoryBreakdown{
			decodeGPU0: {Weights: 30_000_000_000}, decodeGPU1: {Weights: 10_000_000_000},
		},
		fits:   map[ml.DeviceID]api.ProfileDevice{decodeGPU0: measuredFit, decodeGPU1: slow},
		layers: 80, activeWeights: 1,
	}
	base, _, ctxOK, _ := expectedDecode(in)
	near(t, "base ms", base, 0.2+80*0.0251+40e9/1.607e9, 1e-6)
	if ctxOK {
		t.Error("a context term was claimed with no granted context to divide the cache by")
	}
}

// A mixture of experts reads its active experts, and a recurrent state in full whatever the
// routing.
func TestExpectedDecodeReadsOnlyTheActiveExperts(t *testing.T) {
	in := decodeInputs{
		gpus:     []ml.DeviceID{decodeGPU0},
		memByGPU: map[ml.DeviceID]api.MemoryBreakdown{decodeGPU0: {Weights: 60_000_000_000, RecurrentState: 1_000_000_000}},
		fits:     map[ml.DeviceID]api.ProfileDevice{decodeGPU0: measuredFit},
		layers:   36, activeWeights: 0.1,
	}
	base, _, _, _ := expectedDecode(in)
	near(t, "base ms", base, 0.058+36*0.0251+(6e9+1e9)/1.607e9, 1e-6)
}

func TestExpectedDecodeSaysWhyNot(t *testing.T) {
	one := []ml.DeviceID{decodeGPU0}
	mem := map[ml.DeviceID]api.MemoryBreakdown{decodeGPU0: {Weights: 1e9}}
	fits := map[ml.DeviceID]api.ProfileDevice{decodeGPU0: measuredFit}
	for name, tc := range map[string]struct {
		in   decodeInputs
		want string
	}{
		"not measured":   {decodeInputs{gpus: one, memByGPU: mem, layers: 8}, "profile_pending"},
		"spilled":        {decodeInputs{gpus: one, memByGPU: mem, fits: fits, layers: 8, partlyOnCPU: true}, "partly_on_cpu"},
		"no weights":     {decodeInputs{gpus: one, fits: fits, layers: 8}, "memory_unknown"},
		"not on the gpu": {decodeInputs{fits: fits}, "not_on_gpu"},
	} {
		if _, _, _, why := expectedDecode(tc.in); why != tc.want {
			t.Errorf("%s: %q, want %q", name, why, tc.want)
		}
	}
	if r := expectedDecodeReport(decodeInputs{fits: fits}); r != nil {
		t.Errorf("a model on no GPU got %+v; the field should be absent", r)
	}
}

// The recorded prediction is at the context the decode ran through on average, and its basis
// label is decided from the same inputs as the number. A sliding-window model must never be
// labelled as including a cache read it does not contain.
func TestPredictedEvalMsBasis(t *testing.T) {
	in := decodeInputs{
		gpus:     []ml.DeviceID{decodeGPU0},
		memByGPU: map[ml.DeviceID]api.MemoryBreakdown{decodeGPU0: {Weights: 10e9, KVCache: 4e9}},
		fits:     map[ml.DeviceID]api.ProfileDevice{decodeGPU0: measuredFit},
		layers:   48, activeWeights: 1, grantedCtxTotal: 40960,
	}
	base, perCtx, _, _ := expectedDecode(in)
	ms, basis, ok := predictedEvalMs(in, 30000, 200)
	if !ok || basis != "profile" {
		t.Fatalf("ok=%v basis=%q", ok, basis)
	}
	near(t, "predicted ms", ms, base+perCtx*30100, 1e-9)

	in.slidingWindow = true
	ms, basis, _ = predictedEvalMs(in, 30000, 200)
	if basis != "profile_no_kv" || ms != base {
		t.Errorf("sliding window: %.4f %q, want %.4f profile_no_kv", ms, basis, base)
	}
	if _, _, ok := predictedEvalMs(in, 10, 0); ok {
		t.Error("a generation that decoded nothing got a per-token prediction")
	}
}

func TestActiveWeightsFraction(t *testing.T) {
	write := func(kv ggml.KV) *ggml.GGML {
		t.Helper()
		path := filepath.Join(t.TempDir(), "m.gguf")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		ts := []*ggml.Tensor{
			{Name: "token_embd.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{64, 64}, WriterTo: zeroTensor{}},
			{Name: "blk.0.attn_q.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{64, 64}, WriterTo: zeroTensor{}},
			{Name: "blk.0.ffn_up_exps.weight", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{64, 64, 4}, WriterTo: zeroTensor{}},
		}
		if err := ggml.WriteGGUF(f, kv, ts); err != nil {
			t.Fatal(err)
		}
		// The tensors wrote nothing; extend the file so their data regions exist, as a hole.
		if err := f.Truncate(1 << 20); err != nil {
			t.Fatal(err)
		}
		f.Close()
		r, _ := os.Open(path)
		defer r.Close()
		g, err := ggml.Decode(r, -1)
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	moe := write(ggml.KV{"general.architecture": "test", "test.expert_count": uint32(4), "test.expert_used_count": uint32(1)})
	// attn 1 unit, experts 4 units of which 1 is read; the embedding counts on neither side.
	near(t, "moe", activeWeightsFraction(moe), 2.0/5.0, 1e-9)
	dense := write(ggml.KV{"general.architecture": "test"})
	near(t, "dense", activeWeightsFraction(dense), 1, 0)
}

// firstUsageSchema is the generations table as the first usage build (2026-09-12) created it.
const firstUsageSchema = `CREATE TABLE generations (id INTEGER PRIMARY KEY, at_ms INTEGER NOT NULL, model TEXT NOT NULL,
	prompt_tokens INTEGER, prompt_tokens_cached INTEGER, prompt_ms REAL, eval_ms REAL, decoded INTEGER, cache_swap_ms REAL,
	hint_use TEXT, hint_session TEXT, hint_request TEXT, hint_after TEXT, hint_synthetic INTEGER,
	endpoint TEXT, surface TEXT, stream INTEGER, messages INTEGER, images INTEGER, tools INTEGER, format INTEGER, think TEXT,
	req_num_ctx INTEGER, req_num_gpu INTEGER, req_num_predict INTEGER, req_keep_alive_s INTEGER, client TEXT,
	devices TEXT, num_ctx INTEGER, num_batch INTEGER, split TEXT);
`

type zeroTensor struct{}

func (zeroTensor) WriteTo(io.Writer) (int64, error) { return 0, nil }

// A database from before the prediction existed gains its columns, keeps its rows, and records
// the prediction from then on.
func TestUsageStoreAddsColumnsToAnOlderDatabase(t *testing.T) {
	dir := t.TempDir()
	old, err := sql.Open("sqlite3", filepath.Join(dir, serverDBPrevious))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := old.Exec(firstUsageSchema + `INSERT INTO generations (at_ms, model) VALUES (1, 'before');`); err != nil {
		t.Fatal(err)
	}
	old.Close()

	u, err := openServerDB(dir)
	if err != nil {
		t.Fatal(err)
	}
	ms := 13.96
	u.recordGeneration(usageGeneration{At: time.Now(), Model: "after", PredictedEvalMs: &ms, PredictedBasis: "profile"})
	u.close()

	db, _ := sql.Open("sqlite3", filepath.Join(dir, serverDBName))
	defer db.Close()
	var n int
	db.QueryRow(`SELECT count(*) FROM generations`).Scan(&n)
	var got sql.NullFloat64
	var basis sql.NullString
	if err := db.QueryRow(`SELECT predicted_eval_ms_per_token, predicted_basis FROM generations WHERE model='after'`).Scan(&got, &basis); err != nil {
		t.Fatal(err)
	}
	if n != 2 || got.Float64 != 13.96 || basis.String != "profile" {
		t.Errorf("rows=%d predicted=%v basis=%v", n, got, basis)
	}
}
