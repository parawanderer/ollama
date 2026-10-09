package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func fitsTestScheduler(t *testing.T, ctx context.Context, free uint64) (*Scheduler, ml.DeviceInfo) {
	t.Helper()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	g := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}}
	g.TotalMemory = 24 * format.GibiByte
	g.FreeMemory = free
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn
	s.freeMemoryFn = func([]string) map[string]uint64 { return nil }
	return s, g
}

func residentRunner(name string, dev ml.DeviceInfo, vram uint64, busy bool, keepAlive time.Duration) *runnerRef {
	r := &runnerRef{
		name:            name,
		modelKey:        name,
		llama:           &mockLlm{modelPath: name, vramByGPU: map[ml.DeviceID]uint64{dev.DeviceID: vram}},
		gpus:            []ml.DeviceID{dev.DeviceID},
		sessionDuration: keepAlive,
		numParallel:     1,
	}
	if busy {
		r.refCount = 1
	}
	return r
}

func TestFitsNothingResident(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), time.Second)
	defer done()
	s, _ := fitsTestScheduler(t, ctx, 1*format.GibiByte)
	scenario := newScenarioRequest(t, ctx, "fresh", 0, nil, nil)

	resp, err := s.fits(ctx, scenario.req.model, scenario.req.opts, true, true)
	require.NoError(t, err)
	require.True(t, resp.Fits)
	require.Empty(t, resp.Displaces)
	require.NotNil(t, resp.Displaces, "displaces is [] on the wire, never null")
	require.False(t, resp.Resident)
	require.Equal(t, []string{"0"}, resp.GPUIDs)
	require.Equal(t, "metadata", resp.Basis)
	require.Positive(t, resp.Predicted)
}

func TestFitsDisplacesInEvictionOrder(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), time.Second)
	defer done()
	// The card is full: two resident models hold 12 GiB each.
	s, dev := fitsTestScheduler(t, ctx, 0)
	busy := residentRunner("registry.ollama.ai/library/busy:latest", dev, 12*format.GibiByte, true, time.Minute)
	idle := residentRunner("registry.ollama.ai/library/idle:latest", dev, 12*format.GibiByte, false, time.Hour)
	s.loaded[busy.modelKey], s.loaded[idle.modelKey] = busy, idle

	scenario := newScenarioRequest(t, ctx, "newcomer", 0, nil, nil)
	resp, err := s.fits(ctx, scenario.req.model, scenario.req.opts, true, true)
	require.NoError(t, err)
	require.False(t, resp.Fits)
	// Idle goes first even with the longer keep-alive, and one eviction frees enough.
	require.Equal(t, []string{"idle:latest"}, resp.Displaces)
	require.Equal(t, busy, s.loaded[busy.modelKey], "a dry run unloads nothing")
	require.Equal(t, idle, s.loaded[idle.modelKey], "a dry run unloads nothing")
}

func TestFitsAtMaxLoadedModels(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), time.Second)
	defer done()
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "1")
	s, dev := fitsTestScheduler(t, ctx, 20*format.GibiByte)
	r := residentRunner("registry.ollama.ai/library/small:latest", dev, format.GibiByte, false, time.Minute)
	s.loaded[r.modelKey] = r

	scenario := newScenarioRequest(t, ctx, "other", 0, nil, nil)
	resp, err := s.fits(ctx, scenario.req.model, scenario.req.opts, true, true)
	require.NoError(t, err)
	require.False(t, resp.Fits, "memory is plentiful but the model count is at its limit")
	require.Equal(t, []string{"small:latest"}, resp.Displaces)
}

func TestFitsResidentModel(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), time.Second)
	defer done()
	s, dev := fitsTestScheduler(t, ctx, 20*format.GibiByte)
	scenario := newScenarioRequestWithContext(t, ctx, "registry.ollama.ai/library/resident:latest", format.GibiByte, nil,
		map[ml.DeviceID]uint64{dev.DeviceID: format.GibiByte}, 65536)
	scenario.req.opts.NumCtx = 8192
	opts := scenario.req.opts
	s.newServerFn = scenario.newServer
	require.False(t, s.load(scenario.req, getSystemInfoFn(), []ml.DeviceInfo{dev}, false))
	require.Len(t, s.loaded, 1)

	resp, err := s.fits(ctx, scenario.req.model, opts, false, false)
	require.NoError(t, err)
	require.True(t, resp.Fits)
	require.True(t, resp.Resident)
	require.Empty(t, resp.Displaces)
	require.Equal(t, "resident", resp.Basis)

	// A different context forces a reload, which unloads the resident copy first.
	opts.NumCtx = 16384
	resp, err = s.fits(ctx, scenario.req.model, opts, false, false)
	require.NoError(t, err)
	require.False(t, resp.Fits)
	require.False(t, resp.Resident)
	require.Equal(t, []string{"resident:latest"}, resp.Displaces)
	require.Equal(t, 16384, resp.NumCtx)
}

// The dry run is only worth asking if it gives the scheduler's own answer. Swept across
// free memory, the dry run says "fits" exactly when load() would proceed without evicting.
//
// Both batch modes, because they fail differently. An automatic batch only carries a
// surcharge when the prediction plus that surcharge is already within the fit rule, so a
// dry run that compared the bare prediction would still agree there; an explicit large
// batch (a Modelfile's num_batch) carries its surcharge regardless, and is where it would
// disagree.
func TestFitsAgreesWithLoad(t *testing.T) {
	for _, tc := range []struct {
		name     string
		numBatch int // 0: automatic
	}{
		{"automatic batch", 0},
		{"explicit large batch", llamaServerGenerationBatchLarge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, done := context.WithTimeout(t.Context(), 10*time.Second)
			defer done()

			resident := uint64(2 * format.GibiByte)
			ask := func(free uint64) (dry *api.FitsResponse, needEvict bool) {
				s, dev := fitsTestScheduler(t, ctx, free)
				r := residentRunner("registry.ollama.ai/library/resident:latest", dev, resident, false, time.Minute)
				s.loaded[r.modelKey] = r

				scenario := newScenarioRequestWithContext(t, ctx, "candidate", 0, nil, nil, 65536)
				scenario.req.opts.NumCtx = 65536
				auto := tc.numBatch == 0
				if !auto {
					scenario.req.opts.NumBatch = tc.numBatch
				}
				dry, err := s.fits(ctx, scenario.req.model, scenario.req.opts, false, auto)
				require.NoError(t, err)

				scenario.req.numBatchAuto = auto
				s.newServerFn = scenario.newServer
				gpus := s.getGpuFn(ctx, nil)
				s.updateFreeSpace(gpus)
				return dry, s.load(scenario.req, getSystemInfoFn(), gpus, true)
			}

			// A fixed grid, not one derived from the dry run's own prediction: a sweep
			// centred on the answer under test follows a wrong answer around and agrees
			// with it.
			var frees []uint64
			for free := uint64(0); free <= 6*format.GibiByte; free += 32 * format.MebiByte {
				frees = append(frees, free, free+1)
			}
			frees = append(frees, 20*format.GibiByte)
			var fit, evict int
			for _, free := range frees {
				dry, needEvict := ask(free)
				require.Equal(t, !needEvict, dry.Fits, "free=%d predicted=%d", free, dry.Predicted)
				if dry.Fits {
					fit++
				} else {
					evict++
					require.Equal(t, []string{"resident:latest"}, dry.Displaces)
				}
			}
			require.Positive(t, fit, "the sweep must cross the boundary")
			require.Positive(t, evict, "the sweep must cross the boundary")
		})
	}
}

// Through the handler, query string and all. The scheduler-level tests above skipped this
// layer, and the first live request with num_ctx failed: the option parser takes JSON numbers,
// and the handler handed it a Go int.
func TestFitsHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	ctx, done := context.WithTimeout(t.Context(), 5*time.Second)
	defer done()

	sched, _ := fitsTestScheduler(t, ctx, 20*format.GibiByte)
	s := Server{sched: sched, defaultNumCtx: 4096}
	_, digest := createBinFile(t, ggml.KV{
		"general.architecture":          "llama",
		"llama.block_count":             uint32(1),
		"llama.context_length":          uint32(65536),
		"llama.embedding_length":        uint32(4096),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(8),
		"tokenizer.ggml.tokens":         []string{""},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, []*ggml.Tensor{
		{Name: "token_embd.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "output.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
	})
	if w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: "fitme", Files: map[string]string{"file.gguf": digest}}); w.Code != http.StatusOK {
		t.Fatalf("create: %d %s", w.Code, w.Body.String())
	}

	get := func(query string) (int, map[string]any) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodGet, "/api/fits?"+query, nil)
		s.FitsHandler(c)
		var body map[string]any
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &body), w.Body.String())
		return w.Code, body
	}

	code, body := get("model=fitme&num_ctx=8192")
	require.Equal(t, http.StatusOK, code, body)
	require.Equal(t, true, body["fits"])
	require.Equal(t, []any{}, body["displaces"])
	require.EqualValues(t, 8192, body["num_ctx"])

	code, body = get("model=fitme")
	require.Equal(t, http.StatusOK, code, body)
	require.EqualValues(t, 4096, body["num_ctx"], "absent num_ctx is the server's default grant")

	for query, want := range map[string]int{
		"":                        http.StatusBadRequest,
		"model=fitme&num_ctx=abc": http.StatusBadRequest,
		"model=fitme&num_ctx=0":   http.StatusBadRequest,
		"model=missing":           http.StatusNotFound,
	} {
		code, body := get(query)
		require.Equal(t, want, code, "%q: %v", query, body)
	}
}

// A real load whose first placement is uncalibrated probes it and is placed again from the
// measurement. The dry run cannot probe, so it borrows a calibrated placement's measurement
// instead. Here the metadata estimate fits on neither card alone, so without the borrowing the
// dry run spans both and evicts; the real load measures, moves to the card with room, and
// evicts nothing. (gemma4:26b, live, 2026-10-09.)
func TestFitsBorrowsTheMeasurementAProbeWouldTake(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 5*time.Second)
	defer done()

	measured := uint64(100 * format.MebiByte)
	newSched := func() (*Scheduler, ml.DeviceInfo, ml.DeviceInfo) {
		s := InitScheduler(ctx)
		s.waitForRecovery = 10 * time.Millisecond
		g0 := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}}
		g0.TotalMemory, g0.FreeMemory = 24*format.GibiByte, 500*format.MebiByte
		g1 := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "1", Library: "Metal"}}
		g1.TotalMemory, g1.FreeMemory = 24*format.GibiByte, 600*format.MebiByte
		s.getGpuFn = func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo { return []ml.DeviceInfo{g0, g1} }
		s.getSystemInfoFn = getSystemInfoFn
		s.freeMemoryFn = func([]string) map[string]uint64 { return nil }
		s.fitProbe = func(_ context.Context, _ []ml.DeviceInfo, _ string, _ *ggml.GGML, _, _ []string, _ api.Options, _ int, _ string, _ llm.LlamaServerConfig, numCtx int) (uint64, int, error) {
			return measured, numCtx, nil
		}
		r := residentRunner("registry.ollama.ai/library/resident:latest", g0, 23*format.GibiByte, false, time.Minute)
		s.loaded[r.modelKey] = r
		return s, g0, g1
	}

	ask := func(withCalibration bool) (*api.FitsResponse, bool) {
		s, _, g1 := newSched()
		scenario := newScenarioRequestWithContext(t, ctx, "candidate", 0, nil, nil, 65536)
		scenario.req.opts.NumCtx = 65536
		scenario.req.opts.NumBatch = llamaServerGenerationBatchDefault
		if withCalibration {
			key := vramCalibrationKey(scenario.req, []ml.DeviceInfo{g1}, 1)
			s.vramCalibration.Record(key, 8192, measured-format.MebiByte)
			s.vramCalibration.Record(key, 131072, measured+format.MebiByte)
		}
		dry, err := s.fits(ctx, scenario.req.model, scenario.req.opts, false, false)
		require.NoError(t, err)

		s.newServerFn = scenario.newServer
		gpus := s.getGpuFn(ctx, nil)
		s.updateFreeSpace(gpus)
		return dry, s.load(scenario.req, getSystemInfoFn(), gpus, true)
	}

	dry, needEvict := ask(true)
	require.False(t, needEvict, "the real load measures, moves to card 1, and fits")
	require.True(t, dry.Fits, "displaces=%v predicted=%d basis=%s gpus=%v", dry.Displaces, dry.Predicted, dry.Basis, dry.GPUIDs)
	require.Equal(t, []string{"1"}, dry.GPUIDs)
	require.Equal(t, "calibration", dry.Basis)

	// With nothing measured to borrow, the dry run has only the metadata and stays
	// conservative: it is allowed to be wrong toward "doesn't fit", and says why.
	dry, _ = ask(false)
	require.False(t, dry.Fits)
	require.Equal(t, "metadata", dry.Basis)
}
