package server

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

func TestSpeedCorrectionIsTheMedianOfARecentWindow(t *testing.T) {
	c := newSpeedCorrections()
	if f, n := c.factor("k"); f != 1 || n != 0 {
		t.Fatalf("empty: %v %d", f, n)
	}
	for _, r := range []float64{1.1, 5, 1.2} { // one outlier must not move a median
		c.add("k", r)
	}
	near(t, "median of three", func() float64 { f, _ := c.factor("k"); return f }(), 1.2, 1e-12)
	for i := 0; i < correctionWindow; i++ {
		c.add("k", 2)
	}
	if f, n := c.factor("k"); f != 2 || n != correctionWindow {
		t.Errorf("after the window rolled over: %v from %d samples, want 2 from %d", f, n, correctionWindow)
	}
}

// The loop the server runs on every generation: predict, correct from what was learned before,
// then learn. A model running at twice the profile's prediction earns a factor of 2, and from
// then on the corrected prediction is right.
func TestSchedulerLearnsAModelsCorrection(t *testing.T) {
	s := &Scheduler{speed: newSpeedCorrections()}
	in := decodeInputs{
		gpus:     []ml.DeviceID{decodeGPU0},
		memByGPU: map[ml.DeviceID]api.MemoryBreakdown{decodeGPU0: {Weights: 5e9}},
		fits:     map[ml.DeviceID]api.ProfileDevice{decodeGPU0: measuredFit},
		layers:   36, activeWeights: 1,
	}
	raw, _, _ := predictedEvalMs(in, 100, 200)
	gen := func(contended bool, decoded int) usageGeneration {
		row := usageGeneration{Contended: contended}
		s.predictAndLearn("m|CUDA0|none", in, api.GenerationTimings{PromptTokens: 100, Decoded: decoded, EvalMs: 2 * raw * float64(decoded)}, &row)
		return row
	}

	for i := range correctionMinSamples {
		if row := gen(false, 200); row.CorrectedEvalMs != nil {
			t.Fatalf("generation %d got a corrected prediction from %d samples", i, i)
		}
	}
	// Neither of these may teach: one shared the card, one decoded too little to time.
	gen(true, 200)
	gen(false, correctionMinDecoded-1)
	row := gen(false, 200)
	if row.CorrectedEvalMs == nil || row.CorrectionSamples != correctionMinSamples {
		t.Fatalf("corrected=%v samples=%d, want a correction from exactly %d clean samples", row.CorrectedEvalMs, row.CorrectionSamples, correctionMinSamples)
	}
	near(t, "corrected ms", *row.CorrectedEvalMs, 2*raw, 1e-9)
	near(t, "raw ms still recorded", *row.PredictedEvalMs, raw, 1e-9)
}

// A generation shares its card's bandwidth with any other runner there working at the same
// time; a runner on another card, or one that finished before the generation began, does not
// count.
func TestContention(t *testing.T) {
	s := &Scheduler{}
	me := &runnerUsage{key: "me", gpus: []ml.DeviceID{decodeGPU0}}
	neighbour := &runnerUsage{key: "neighbour", gpus: []ml.DeviceID{decodeGPU0, decodeGPU1}}
	elsewhere := &runnerUsage{key: "elsewhere", gpus: []ml.DeviceID{decodeGPU1}}
	for _, u := range []*runnerUsage{me, neighbour, elsewhere} {
		s.usages.add(u)
	}
	now := time.Now().UnixMilli()
	since := now - 1000

	elsewhere.noteRequests(1)
	neighbour.idleSinceMs.Store(since - 5000)
	if s.contended(me, since) {
		t.Error("contended by a runner on another card, or one idle since before the generation")
	}
	neighbour.idleSinceMs.Store(since + 10)
	if !s.contended(me, since) {
		t.Error("a neighbour that worked during the generation was not counted")
	}
	neighbour.idleSinceMs.Store(since - 5000)
	neighbour.noteRequests(1)
	if !s.contended(me, since) {
		t.Error("a neighbour working now was not counted")
	}

	// An unloaded runner is forgotten once it has been idle long enough.
	neighbour.noteRequests(0)
	neighbour.idleSinceMs.Store(time.Now().Add(-2 * usageForgetAfter).UnixMilli())
	neighbour.unloaded.Store(true)
	s.contended(me, since)
	if len(s.usages.others(me)) != 1 {
		t.Error("a runner unloaded long ago was kept")
	}
}

// The corrections survive a restart: they are rebuilt from the generations recorded, leaving out
// the ones that taught nothing at the time.
func TestSpeedCorrectionsAreRebuiltFromTheDatabase(t *testing.T) {
	dir := t.TempDir()
	d := testServerDB(t, dir)
	for i, row := range []struct {
		decoded   int
		ratio     float64
		contended bool
	}{{200, 1.5, false}, {200, 1.5, false}, {200, 9, true}, {4, 9, false}, {200, 1.5, false}} {
		ms := 10.0
		d.recordGeneration(usageGeneration{
			At: time.Now().Add(time.Duration(i) * time.Millisecond), Model: "m", Devices: "CUDA0", Split: "none",
			Timings:         api.GenerationTimings{Decoded: row.decoded, EvalMs: ms * row.ratio * float64(row.decoded)},
			PredictedEvalMs: &ms, Contended: row.contended,
		})
	}
	d.close()

	c := newSpeedCorrections()
	if err := testServerDB(t, dir).loadSpeedCorrections(c); err != nil {
		t.Fatal(err)
	}
	f, n := c.factor(speedKey("m", "CUDA0", "none"))
	if n != 3 || f != 1.5 {
		t.Errorf("rebuilt %v from %d samples, want 1.5 from 3 (the contended and the short one left out)", f, n)
	}
	var contended int
	db, _ := sql.Open("sqlite3", filepath.Join(dir, serverDBName)+"?mode=ro")
	defer db.Close()
	db.QueryRow(`SELECT count(*) FROM generations WHERE contended = 1`).Scan(&contended)
	if contended != 1 {
		t.Errorf("%d rows marked contended, want 1", contended)
	}
}

func TestExpectedDecodeReportAppliesTheCorrection(t *testing.T) {
	in := decodeInputs{
		gpus:     []ml.DeviceID{decodeGPU0},
		memByGPU: map[ml.DeviceID]api.MemoryBreakdown{decodeGPU0: {Weights: 5e9, KVCache: 1 << 30}},
		fits:     map[ml.DeviceID]api.ProfileDevice{decodeGPU0: measuredFit},
		layers:   36, activeWeights: 1, grantedCtxTotal: 8192,
	}
	plain := expectedDecodeReport(in, 2, correctionMinSamples-1)
	if plain.Basis != "profile" || plain.CorrectionFactor != 0 {
		t.Errorf("a correction from too few samples was applied: %+v", plain)
	}
	r := expectedDecodeReport(in, 2, correctionMinSamples)
	if r.Basis != "profile_corrected" || r.ProfileTokensPerSec != plain.TokensPerSec || r.CorrectionFactor != 2 {
		t.Fatalf("report = %+v", r)
	}
	near(t, "corrected tok/s", r.TokensPerSec, plain.TokensPerSec/2, 0.01)
	near(t, "corrected context slope", r.MsPerTokenPer1kContext, plain.MsPerTokenPer1kContext*2, 0.01)
}

// The scheduler keeps a runner's usage in step with its requests: working while one is in
// flight, idle with a time once it finishes. If the decrement were not recorded, every later
// generation on a shared card would count as contended and nothing would ever be learned.
func TestRunnerUsageFollowsItsRequests(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 2*time.Second)
	defer done()
	s := InitScheduler(ctx)
	s.getGpuFn = getGpuFn
	s.getSystemInfoFn = getSystemInfoFn
	a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: time.Minute}, nil)
	s.newServerFn = a.newServer
	s.pendingReqCh <- a.req
	s.Run(ctx)
	var runner *runnerRef
	select {
	case runner = <-a.req.successCh:
	case err := <-a.req.errCh:
		t.Fatal(err)
	case <-ctx.Done():
		t.Fatal("timeout")
	}
	if runner.usage == nil {
		t.Fatal("the runner has no usage, so contention can never be seen")
	}
	if got := runner.usage.inFlight.Load(); got != 1 {
		t.Fatalf("in flight while serving = %d, want 1", got)
	}
	if others := s.usages.others(nil); len(others) != 1 || others[0] != runner.usage {
		t.Fatalf("registry = %v, want the one runner", others)
	}

	before := time.Now().UnixMilli()
	a.ctxDone() // the request finishes
	for runner.usage.inFlight.Load() != 0 {
		if ctx.Err() != nil {
			t.Fatal("the finished request was never recorded")
		}
		time.Sleep(5 * time.Millisecond)
	}
	if runner.usage.idleSinceMs.Load() < before {
		t.Error("the runner went idle without recording when")
	}
}
