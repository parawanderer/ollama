package llm

import (
	"os"
	"path/filepath"
	"testing"
)

const (
	mib = 1024 * 1024
	gib = 1024 * mib
)

func testKey() CalibrationKey {
	return CalibrationKey{Model: "/models/test.gguf", ModelSize: 1234, NumParallel: 1, NumGPU: 1}
}

// withinMiB keeps the assertions readable: these are memory sizes, and agreement to the
// byte is neither achievable from a least-squares fit nor required by the caller.
func withinMiB(got, want uint64, mibs float64) bool {
	diff := float64(got) - float64(want)
	if diff < 0 {
		diff = -diff
	}
	return diff <= mibs*mib
}

func TestCalibrationFallsBackToPriorWithoutSamples(t *testing.T) {
	c := NewVRAMCalibration()

	got, calibrated := c.Predict(testKey(), 8192, 10*gib, 1024)
	if calibrated {
		t.Error("reported a calibrated prediction with no samples recorded")
	}
	if want := uint64(10*gib + 8192*1024); got != want {
		t.Errorf("prior not returned unchanged: got=%d want=%d", got, want)
	}
}

func TestCalibrationFitsTwoSamples(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()

	// 20 GiB of weights plus exactly 2048 bytes per token.
	c.Record(key, 8192, 20*gib+8192*2048)
	c.Record(key, 32768, 20*gib+32768*2048)

	for _, numCtx := range []int{4096, 16384, 65536, 262144} {
		got, calibrated := c.Predict(key, numCtx, 999*gib, 1)
		if !calibrated {
			t.Fatalf("ctx %d: fell back to the prior despite two samples", numCtx)
		}
		want := uint64(20*gib + numCtx*2048)
		if !withinMiB(got, want, 1) {
			t.Errorf("ctx %d: got=%d want=%d", numCtx, got, want)
		}
	}
}

// The prior is deliberately absurd in the test above and here: a calibrated prediction
// must not be influenced by it at all.
func TestCalibrationSingleSampleUsesPriorSlopeThroughMeasuredPoint(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()
	c.Record(key, 8192, 30*gib)

	// One sample fixes the intercept; the slope has to come from the prior.
	got, calibrated := c.Predict(key, 16384, 999*gib, 4096)
	if !calibrated {
		t.Fatal("a single sample should still beat the prior")
	}
	if want := uint64(30*gib + 8192*4096); got != want {
		t.Errorf("got=%d want=%d", got, want)
	}

	// And it must extrapolate downwards as well, without underflowing.
	got, _ = c.Predict(key, 1024, 999*gib, 4096)
	if want := uint64(30*gib - 7168*4096); got != want {
		t.Errorf("downward: got=%d want=%d", got, want)
	}
	if got, _ := c.Predict(key, 1, 0, 1<<62); got != 0 {
		t.Errorf("expected underflow to clamp at 0, got %d", got)
	}
}

func TestCalibrationKeyChangeInvalidates(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()
	c.Record(key, 8192, 20*gib+8192*2048)
	c.Record(key, 32768, 20*gib+32768*2048)

	// Each of these describes a load whose memory use is not comparable, so each must miss
	// the samples above rather than silently applying them.
	changed := map[string]func(*CalibrationKey){
		"model":           func(k *CalibrationKey) { k.Model = "/models/other.gguf" },
		"model size":      func(k *CalibrationKey) { k.ModelSize = 5678 },
		"projector":       func(k *CalibrationKey) { k.Projectors = "/blobs/sha256-abc" },
		"kv cache type":   func(k *CalibrationKey) { k.KVCacheType = "q8_0" },
		"flash attention": func(k *CalibrationKey) { k.FlashAttention = true },
		"num batch":       func(k *CalibrationKey) { k.NumBatch = 512 },
		"num parallel":    func(k *CalibrationKey) { k.NumParallel = 4 },
		"num gpu":         func(k *CalibrationKey) { k.NumGPU = 2 },
	}

	for name, mutate := range changed {
		t.Run(name, func(t *testing.T) {
			other := testKey()
			mutate(&other)
			if _, calibrated := c.Predict(other, 8192, 10*gib, 1024); calibrated {
				t.Errorf("%s changed but the old samples were still applied", name)
			}
		})
	}

	if _, calibrated := c.Predict(key, 8192, 10*gib, 1024); !calibrated {
		t.Error("the original key stopped resolving to its own samples")
	}
}

func TestCalibrationRecordReplacesSameContext(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()

	c.Record(key, 8192, 40*gib)
	c.Record(key, 8192, 50*gib) // the model changed underneath, or the first load was odd
	c.Record(key, 16384, 50*gib+8192*2048)

	got, calibrated := c.Predict(key, 8192, 999*gib, 1)
	if !calibrated {
		t.Fatal("expected a calibrated prediction")
	}
	if !withinMiB(got, 50*gib, 1) {
		t.Errorf("stale sample was not replaced: got=%d want≈%d", got, uint64(50*gib))
	}
}

func TestCalibrationIgnoresUnusableInput(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()

	c.Record(key, 0, 40*gib)  // no context length to attribute it to
	c.Record(key, 8192, 0)    // a load that reported nothing
	c.Record(key, -1, 40*gib) // nonsense

	if _, calibrated := c.Predict(key, 8192, 10*gib, 1024); calibrated {
		t.Error("unusable samples were recorded")
	}

	// A nil store is usable and simply never calibrates, so callers need no nil check.
	var nilStore *VRAMCalibration
	nilStore.Record(key, 8192, 40*gib)
	if got, calibrated := nilStore.Predict(key, 8192, 10*gib, 1024); calibrated || got != uint64(10*gib+8192*1024) {
		t.Errorf("nil store: got=%d calibrated=%v", got, calibrated)
	}
}

func TestCalibrationRejectsNegativeSlope(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()

	// Memory does not shrink as context grows. Samples that say otherwise describe
	// something this model cannot express, so the prior is safer than the line.
	c.Record(key, 8192, 40*gib)
	c.Record(key, 32768, 30*gib)

	got, calibrated := c.Predict(key, 262144, 50*gib, 1024)
	if calibrated {
		t.Error("extrapolated a negative slope instead of falling back")
	}
	if want := uint64(50*gib + 262144*1024); got != want {
		t.Errorf("got=%d want=%d", got, want)
	}
}

func TestCalibrationBoundsSampleCount(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()

	for i := 1; i <= maxCalibrationSamples+5; i++ {
		c.Record(key, i*1024, uint64(20*gib+i*1024*2048))
	}

	c.mu.Lock()
	n := len(c.samples[key])
	c.mu.Unlock()
	if n != maxCalibrationSamples {
		t.Errorf("sample count not bounded: got=%d want=%d", n, maxCalibrationSamples)
	}

	// Dropping the oldest must not disturb the fit, since the line is the same.
	got, calibrated := c.Predict(key, 100*1024, 999*gib, 1)
	if !calibrated {
		t.Fatal("expected a calibrated prediction")
	}
	if want := uint64(20*gib + 100*1024*2048); !withinMiB(got, want, 1) {
		t.Errorf("got=%d want=%d", got, want)
	}
}

// TestCalibrationAgainstMeasuredCurve uses VRAM actually measured on a two-card host, to
// check that calibrating from the two cheapest loads predicts the expensive ones. The
// metadata prediction is 7.05 GiB low at the longest context because it models one KV
// cache and this architecture allocates two; the point of calibrating is that the shortfall
// does not have to be understood to be corrected.
func TestCalibrationAgainstMeasuredCurve(t *testing.T) {
	measured := []struct {
		numCtx int
		vram   float64 // GiB, from llama-server's own buffer accounting
	}{
		{8192, 76.84},
		{32768, 77.71},
		{131072, 81.41},
		{262144, 86.35},
	}

	c := NewVRAMCalibration()
	key := testKey()
	for _, m := range measured[:2] {
		c.Record(key, m.numCtx, uint64(m.vram*gib))
	}

	for _, m := range measured[2:] {
		got, calibrated := c.Predict(key, m.numCtx, 76*gib, 10138)
		if !calibrated {
			t.Fatalf("ctx %d: expected a calibrated prediction", m.numCtx)
		}
		want := uint64(m.vram * gib)
		if !withinMiB(got, want, 600) {
			t.Errorf("ctx %d: got=%.2f GiB want=%.2f GiB (off by %.0f MiB)",
				m.numCtx, float64(got)/gib, m.vram, (float64(got)-float64(want))/mib)
		}
	}
}

func TestCalibrationSurvivesARoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vram-calibration.json")

	saved := NewVRAMCalibration()
	key := testKey()
	saved.Record(key, 8192, 20*gib+8192*2048)
	saved.Record(key, 32768, 20*gib+32768*2048)
	if err := saved.Save(path); err != nil {
		t.Fatal(err)
	}

	loaded := NewVRAMCalibration()
	loaded.Load(path)

	got, calibrated := loaded.Predict(key, 65536, 999*gib, 1)
	if !calibrated {
		t.Fatal("samples did not survive the round trip, so a restart would start cold")
	}
	if want := uint64(20*gib + 65536*2048); !withinMiB(got, want, 1) {
		t.Errorf("got=%d want=%d", got, want)
	}
}

// A file from another version, or one that is corrupt, must leave the store empty rather
// than half-populated: a cold start is always safe, a misread sample is not.
func TestCalibrationDiscardsUnusableFiles(t *testing.T) {
	dir := t.TempDir()

	for name, content := range map[string]string{
		"corrupt.json":     "{not json",
		"wrongver.json":    `{"version":999,"entries":[{"key":{"Model":"/models/test.gguf"},"samples":[{"num_ctx":8192,"vram":1}]}]}`,
		"emptysample.json": `{"version":1,"entries":[{"key":{"Model":"/models/test.gguf"},"samples":[{"num_ctx":0,"vram":0}]}]}`,
	} {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		c := NewVRAMCalibration()
		c.Load(path)
		if _, calibrated := c.Predict(testKey(), 8192, 10*gib, 1024); calibrated {
			t.Errorf("%s: was treated as usable", name)
		}
	}

	// A path that does not exist is the normal first-run case and must be silent.
	c := NewVRAMCalibration()
	c.Load(filepath.Join(dir, "absent.json"))
	if _, calibrated := c.Predict(testKey(), 8192, 10*gib, 1024); calibrated {
		t.Error("a missing file produced samples from nowhere")
	}
}

// TestMaxContextForInvertsPredict is the property that matters: whatever context this hands
// back must be one Predict then says fits. If the two disagree, a caller solves for a
// context with one line and is judged against another, and the answer corresponds to no
// decision anyone made.
func TestMaxContextForInvertsPredict(t *testing.T) {
	key := testKey()

	for name, setup := range map[string]func(*VRAMCalibration){
		"no samples": func(c *VRAMCalibration) {},
		"one sample": func(c *VRAMCalibration) { c.Record(key, 8192, 20*gib) },
		"fitted line": func(c *VRAMCalibration) {
			c.Record(key, 8192, 20*gib+8192*2048)
			c.Record(key, 32768, 20*gib+32768*2048)
		},
		"negative slope": func(c *VRAMCalibration) { c.Record(key, 8192, 40*gib); c.Record(key, 32768, 30*gib) },
	} {
		t.Run(name, func(t *testing.T) {
			c := NewVRAMCalibration()
			setup(c)

			const budget = 40 * gib
			ctx := c.MaxContextFor(key, budget, 20*gib, 2048)
			if ctx <= 0 {
				t.Fatalf("no context fits in %d GiB for a 20 GiB model", budget/gib)
			}

			// Rounding down means the answer is safe, so a small overshoot at ctx+1 is
			// expected and an overshoot at ctx is not.
			if got, _ := c.Predict(key, ctx, 20*gib, 2048); got > budget {
				t.Errorf("solved for %d, which Predict then puts at %d bytes -- over the %d budget",
					ctx, got, uint64(budget))
			}
			if got, _ := c.Predict(key, ctx+8192, 20*gib, 2048); got <= budget {
				t.Errorf("solved for %d but %d also fits, so this is not the largest", ctx, ctx+8192)
			}
		})
	}
}

// A model whose weights alone exceed the budget has no context to solve for, and saying
// "zero" is different from saying "a very small one": the caller must split or refuse, not
// shrink.
func TestMaxContextForRefusesWhenWeightsDoNotFit(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()
	c.Record(key, 8192, 80*gib+8192*2048)
	c.Record(key, 32768, 80*gib+32768*2048)

	if got := c.MaxContextFor(key, 40*gib, 80*gib, 2048); got != 0 {
		t.Errorf("offered %d context for a model whose weights are twice the budget", got)
	}
	if got := c.MaxContextFor(key, 0, 80*gib, 2048); got != 0 {
		t.Errorf("offered %d context against no budget at all", got)
	}
}

// The measured curve again, used the other way round: given the memory a card actually has,
// how much context does this model get? The answer has to agree with the loads that were
// really run at those sizes.
func TestMaxContextForAgainstMeasuredCurve(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()
	perGiB := float64(gib)
	c.Record(key, 8192, uint64(76.84*perGiB))
	c.Record(key, 32768, uint64(77.71*perGiB))

	// 86.35 GiB was measured at 262144, so a budget of that size should offer about that
	// context -- within a few percent, since the line is fitted from two shorter loads.
	got := c.MaxContextFor(key, uint64(86.35*perGiB), 0, 0)
	if got < 240000 || got > 285000 {
		t.Errorf("offered %d context for the budget a 262144 load actually used", got)
	}
}
