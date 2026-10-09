package server

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

// GET /api/fits?model=<name>[&num_ctx=<n>] asks whether loading a model now would unload
// anything resident, without loading it.
//
// It is the scheduler's own decision run dry: the same request, placement, prediction,
// fit rule and eviction order as a real load, walked one eviction at a time the way
// processPending walks it. What it leaves out is the probe. A real load of an invocation
// with fewer than two calibration samples measures itself first, which spawns a
// llama-server on the cards; a question must not do that, so an uncalibrated model is
// answered from the metadata estimate and says so in Basis.
func (s *Server) FitsHandler(c *gin.Context) {
	raw := c.Query("model")
	if raw == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	requestOpts := map[string]any{}
	if v := c.Query("num_ctx"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "num_ctx must be a positive integer"})
			return
		}
		// As a JSON number: the option parser takes what a request body decodes to, and an int
		// is refused as "must be of type integer".
		requestOpts["num_ctx"] = float64(n)
	}

	modelRef, err := parseAndValidateModelRef(raw)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", raw))
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.JSON(http.StatusBadRequest, gin.H{"error": "a cloud model is not loaded here"})
		return
	}
	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", raw)})
		return
	}
	m, err := GetModel(name.String())
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", raw)})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if m.IsMLX() || m.ModelPath == "" {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"error": "fit is only predicted for GGUF models"})
		return
	}

	// The same option resolution scheduleRunner applies, so the answer is about the load a
	// request with these options would cause.
	numCtxAuto := usesAutomaticNumCtx(m, requestOpts)
	embeddingBatchDefault := shouldApplyEmbeddingBatchDefault(m, requestOpts)
	numBatchAuto := usesAutomaticNumBatch(m, requestOpts) && !embeddingBatchDefault
	opts, err := s.modelOptionsWithEmbeddingBatchDefault(m, requestOpts, embeddingBatchDefault)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	resp, err := s.sched.fits(c.Request.Context(), m, opts, numCtxAuto, numBatchAuto)
	switch {
	case errors.Is(err, errAllGPUsLeased):
		c.JSON(http.StatusConflict, gin.H{"error": err.Error()})
	case err != nil:
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	default:
		c.JSON(http.StatusOK, resp)
	}
}

// fits runs the scheduler's load decision for m without loading it.
func (s *Scheduler) fits(ctx context.Context, m *Model, opts api.Options, numCtxAuto, numBatchAuto bool) (*api.FitsResponse, error) {
	req := newLlmRequest(ctx, m, opts, nil, numCtxAuto, numBatchAuto, nil)
	resp := &api.FitsResponse{Displaces: []string{}}

	s.loadedMu.Lock()
	self := s.loaded[schedulerModelKey(m)]
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()

	evicted := map[*runnerRef]bool{}
	if self != nil {
		if !self.needsReload(ctx, req) && !s.onLeasedDevice(self) {
			resp.Fits, resp.Resident, resp.Basis = true, true, "resident"
			self.refMu.Lock()
			if self.llama != nil {
				resp.NumCtx = self.llama.ContextLength()
				if granted, _ := self.llama.GrantedContext(); granted > 0 {
					resp.NumCtx = granted
				}
			}
			for _, d := range self.gpus {
				resp.GPUIDs = append(resp.GPUIDs, d.ID)
			}
			self.refMu.Unlock()
			return resp, nil
		}
		// A reload unloads the resident copy first, exactly as processPending does.
		evicted[self] = true
		resp.Displaces = append(resp.Displaces, runnerDisplayName(self))
	}

	var gpus []ml.DeviceInfo
	if opts.NumGPU != 0 {
		// The cached devices with live free memory: no discovery, so asking cannot start
		// a llama-server on an idle card.
		all := s.cachedDevices(ctx)
		gpus = s.withoutLeased(all)
		if len(all) > 0 && len(gpus) == 0 {
			return nil, errAllGPUsLeased
		}
	}
	systemInfo := s.getSystemInfoFn()
	maxRunners := int(envconfig.MaxRunners())
	if maxRunners <= 0 {
		n := len(gpus)
		if opts.NumGPU == 0 {
			n = len(s.cachedDevices(ctx))
		}
		maxRunners = defaultModelsPerGPU * max(n, 1)
	}

	f, err := llm.LoadModel(m.ModelPath, 1024)
	if err != nil {
		return nil, err
	}
	numParallel, completion, _ := loadParallelism(req)
	predictedCtx := effectiveLlamaServerContext(req.opts.NumCtx, f, numParallel)
	resp.NumCtx = effectiveModelContext(req.opts.NumCtx, f)

	for {
		var remaining []*runnerRef
		for _, r := range runners {
			if !evicted[r] {
				remaining = append(remaining, r)
			}
		}

		var next *runnerRef
		if len(remaining) > 0 && len(remaining) >= maxRunners {
			next = pickRunnerToUnload(remaining)
		} else {
			// A copy per pass: settleBatch rewrites the request's batch, and each pass
			// must start from what the request asked for.
			pass := *req
			devices := freeAfterEvicting(gpus, runners, evicted)
			placed, launchOpts, predictedForLoad, basis := s.dryPlacement(&pass, systemInfo, devices, f, completion, predictedCtx, numParallel)
			resp.GPUIDs = resp.GPUIDs[:0]
			for _, d := range placed {
				resp.GPUIDs = append(resp.GPUIDs, d.ID)
			}
			resp.Predicted, resp.Basis = int64(predictedForLoad), basis

			// The same three ways a load proceeds without evicting: nothing resident
			// (the first load takes the best fit it can get), no GPU, or an explicit
			// partial offload; otherwise the fit rule.
			if len(remaining) == 0 || len(placed) == 0 || explicitPartialGPUOffload(launchOpts, f) {
				break
			}
			if ok, _, _, _ := fitsInAvailableMemory(systemInfo, placed, launchOpts, predictedForLoad); ok {
				break
			}
			next = pickRunnerToUnload(remaining)
		}
		evicted[next] = true
		resp.Displaces = append(resp.Displaces, runnerDisplayName(next))
	}

	resp.Fits = len(resp.Displaces) == 0
	return resp, nil
}

// dryPlacement is load's placement and prediction with the probe left out: an invocation
// is predicted from its calibration where it has one and from metadata where it does not.
func (s *Scheduler) dryPlacement(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, f *ggml.GGML, completion bool, predictedCtx, numParallel int) (placed []ml.DeviceInfo, launchOpts api.Options, predictedForLoad uint64, basis string) {
	predicted := predictLlamaServerVRAM(s.vramCalibration, vramCalibrationKey(req, gpus, numParallel), req, f, predictedCtx)
	placed, launchOpts, key, predicted, _ := settlePlacement(predicted,
		func(estimate uint64) ([]ml.DeviceInfo, api.Options) {
			return selectLlamaServerPlacement(systemInfo, gpus, estimate, req.opts)
		},
		func(placed []ml.DeviceInfo, opts api.Options, estimate uint64, _ bool) (api.Options, llm.CalibrationKey, uint64, bool) {
			opts, key := req.settleBatch(systemInfo, placed, opts, estimate, completion, predictedCtx, numParallel)
			return opts, key, predictLlamaServerVRAM(s.vramCalibration, key, req, f, predictedCtx), false
		})
	_, calibrated := s.vramCalibration.Predict(key, predictedCtx, 0, 0)
	return placed, launchOpts, predicted + generationBatchSurchargeForCompletion(completion, launchOpts.NumBatch), predictionSource(calibrated, false)
}

// freeAfterEvicting is the device list as processPending would see it once the evicted
// runners were gone: updateFreeSpace's rule (the smaller of the driver's free figure and
// total minus what resident runners hold), applied after giving back what the evicted
// runners hold. A copy; the cached devices are not touched.
func freeAfterEvicting(gpus []ml.DeviceInfo, runners []*runnerRef, evicted map[*runnerRef]bool) []ml.DeviceInfo {
	out := make([]ml.DeviceInfo, len(gpus))
	copy(out, gpus)
	if len(runners) == 0 {
		return out
	}
	for i := range out {
		var held, freed uint64
		for _, r := range runners {
			r.refMu.Lock()
			if r.llama != nil {
				v := r.llama.VRAMByGPU(out[i].DeviceID)
				if evicted[r] {
					freed += v
				} else {
					held += v
				}
			}
			r.refMu.Unlock()
		}
		free := min(out[i].FreeMemory+freed, out[i].TotalMemory)
		if held > out[i].TotalMemory {
			free = 0
		} else if out[i].TotalMemory-held < free {
			free = out[i].TotalMemory - held
		}
		out[i].FreeMemory = free
	}
	return out
}

// runnerDisplayName names a runner the way /api/ps names it.
func runnerDisplayName(r *runnerRef) string {
	if name := model.ParseName(r.name).DisplayShortest(); name != "" {
		return name
	}
	return r.name
}
