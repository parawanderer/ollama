package llm

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"regexp"
	"strconv"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

// llama-server's fit pass reports what a load would occupy before it loads anything, one
// row per device:
//
//	| - CUDA0 (RTX PRO 6000) | 97249 = 12510 + (78688 = 78056 + 376 + 255) + 6050 |
//	                           total   free    self    model  context compute  unaccounted
//
// self is what this load would hold on that device. Summing it across devices gives the
// same quantity a completed load reports, which is what makes a probe interchangeable
// with a measurement.
// Only device rows are summed. A "Host" row appears even on a load that is entirely on
// GPU -- it is the mmap'd, file-backed view of the weights, whose span overlaps the device
// copy rather than adding to it, which is why ollama's own accounting excludes it too. It
// has no parentheses after the name, so this pattern skips it. Do not "fix" that: counting
// it would add the whole model file a second time.
var fitBreakdownRegex = regexp.MustCompile(`memory_breakdown_print:.*\|\s+-\s+\S+\s+\([^)]*\)\s+\|\s+\d+\s+=\s+\d+\s+\+\s+\(\s*(\d+)\s+=`)

// The fit pass always states its verdict, and a result is only usable when that verdict is
// the clean one. This is a positive requirement rather than a list of failure phrasings on
// purpose: an earlier version enumerated adjustment verbs and matched none of the ones
// llama-server actually prints ("cannot meet", "moved", "filling"), so a degraded probe
// would have been recorded as a measurement.
// "no changes needed" is the invariant across the verdict's three phrasings -- the wording
// before it names how the check was framed, and differs by device count:
//
//	multi-device:  targets for free memory can be met on all devices, no changes needed
//	single device: will leave 17886 >= 1024 MiB of free device memory, no changes needed
//	cpu only:      will leave 123126 >= 1024 MiB of system memory, no changes needed
//
// Matching the multi-device wording alone rejected every single-device probe, which is the
// common case here, so key on the part that does not vary.
var (
	fitCleanRegex = regexp.MustCompile(`common_params_fit_impl:.*no changes needed`)
	// Singular and plural both occur -- "cannot meet free memory target of 90000 MiB" on
	// one device, "...targets on all devices" on several. This is only an early exit:
	// correctness rests on the clean verdict above, which a degraded pass never prints.
	// But it is worth having, because giving up here takes 0.7s where letting the pass
	// exhaust its fallbacks takes 15.
	fitTooSmallRegex = regexp.MustCompile(`common_params_fit_impl: cannot meet free memory target`)
)

// fitProbeTimeout bounds one probe. Measured cost is 0.4-0.9s including process start,
// on models from 1 to 104 GiB -- the pass reads metadata and reserves graphs, never
// tensors, so size barely moves it. The timeout is generous against a cold page cache
// and exists only so a probe cannot become the thing that delays a load.
const fitProbeTimeout = 30 * time.Second

// ErrFitProbeWouldNotFit reports that the model could not be placed as asked in the memory
// free at the time, so llama-server began moving it to system memory instead.
//
// llama-server does not fail in this situation -- it degrades, printing a fresh breakdown
// for each fallback it tries (all experts to host, then dense layers back-to-front), each
// one smaller than the last. A reader that simply takes the final breakdown records about
// a gigabyte as the cost of a hundred-gigabyte model, and persists it.
var ErrFitProbeWouldNotFit = errors.New("model does not fit in the memory free right now, so it cannot be measured")

// ErrFitProbeDraftModel reports a load whose fit pass describes more than one model, which
// this cannot yet total correctly. See ProbeFitVRAM.
var ErrFitProbeDraftModel = errors.New("a draft model's memory cannot be separated from the main model's in a fit probe")

// ProbeFitVRAM reports what a model would occupy at numCtx, by running llama-server's fit
// pass and killing it as soon as it has answered. Nothing is loaded and no memory is
// reserved, so this is safe to run against a device that is already full.
//
// It exists for architectures whose per-token cost cannot be derived from their metadata
// -- sparse-attention indexers and latent attention both allocate caches the published
// head dimensions do not describe. For those, the metadata estimate is a lower bound that
// grows worse with context, and the only other source of truth is having loaded the model
// once already.
//
// Verified on Qwen3.8-Flash-Next (104 GiB, sparse attention): probed 76.85 GiB at 8k and
// 81.41 GiB at 128k, against 76.84 and 81.41 actually measured, in 0.5s per probe.
func ProbeFitVRAM(
	ctx context.Context,
	gpus []ml.DeviceInfo,
	modelPath string,
	f *ggml.GGML,
	adapters, projectors []string,
	opts api.Options,
	numParallel int,
	kvCacheType string,
	config LlamaServerConfig,
	numCtx int,
) (uint64, error) {
	exe, err := FindLlamaServer()
	if err != nil {
		return 0, err
	}

	// The probe must differ from the real invocation in the context length and nothing
	// else. Batch size and split mode both move the compute buffers, and only those --
	// measured on Qwen3.8-Flash-Next at 256k, the model and context terms are byte-identical
	// across all three of these and the whole spread is compute:
	//
	//	1 device, -b 512    model 78056 + ctx 8560 + compute  1809 =  86.35 GiB
	//	1 device, -b 2048   model 78056 + ctx 8560 + compute  7227 =  91.65 GiB
	//	2 devices, -b 2048  model 78055 + ctx 8559 + compute 24304 = 108.32 GiB
	//
	// Spreading a single-device load over two devices is not a double count: the second
	// device brings its own compute buffers and the pair costs 3.4x one, which is why
	// getting split mode wrong put a probe 39% over the truth rather than 2x.
	opts.NumCtx = numCtx
	launch, err := newLlamaServerLaunchConfig(gpus, modelPath, f, adapters, projectors, opts, numParallel, kvCacheType, config, newLlamaServerMediaMarker())
	if err != nil {
		return 0, err
	}

	// A draft model makes the fit pass describe two models, one breakdown each, and this
	// cannot yet tell what their totals mean together: they share the weights, so the
	// figures do not add, and the draft carries its own small context, so neither is the
	// answer on its own. Measured on qwen3.8:27b at 128k -- main 25279 MiB, draft 16635,
	// the load itself 26982. Refusing is the safe reading: the caller falls back to the
	// metadata estimate, which for these over-predicts, where a wrong probe under-predicts.
	if launch.draftType != "" {
		return 0, ErrFitProbeDraftModel
	}

	ctx, cancel := context.WithTimeout(ctx, fitProbeTimeout)
	defer cancel()

	// Port 0 is never bound: the process is killed before it listens. Passing a real port
	// would risk colliding with a runner that is about to start on it.
	params := append(llamaServerParams(launch, 0), "--fit", "on")

	cmd := exec.Command(exe, params...)
	cmd.SysProcAttr = LlamaServerSysProcAttr
	SetupLlamaServerCommandEnv(cmd, exe, launch.gpuLibs, launch.extraEnvsForStart())

	out, err := cmd.StdoutPipe()
	if err != nil {
		return 0, err
	}
	cmd.Stderr = cmd.Stdout

	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("starting fit probe: %w", err)
	}

	// The process is killed the moment the answer is in hand rather than left to exit on
	// its own: it would otherwise go on to load the model, which is the entire cost this
	// is avoiding. Kill and Wait both run on every path, including the timeout, so a
	// probe cannot outlive the call that made it.
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			_ = cmd.Process.Kill()
		case <-done:
		}
	}()

	total, clean := scanFitBreakdown(out)
	close(done)
	_ = cmd.Process.Kill()
	_, _ = io.Copy(io.Discard, out)
	_ = cmd.Wait()

	switch {
	case total == 0:
		if err := ctx.Err(); err != nil {
			return 0, fmt.Errorf("fit probe did not report in %s: %w", fitProbeTimeout, err)
		}
		return 0, errors.New("fit probe produced no memory breakdown")
	case !clean:
		return 0, ErrFitProbeWouldNotFit
	}

	slog.Debug("probed model memory without loading it", "model", modelPath, "num_ctx", numCtx, "vram", total, "cmd", cmd)
	return total, nil
}

// scanFitBreakdown sums the per-device figures of the breakdown the fit pass reaches its
// verdict on, and reports whether that verdict was the clean one.
//
// The pass prints one breakdown per configuration it considers, so a running total has to
// be replaced by each new breakdown rather than added to. Scanning stops at the verdict,
// which is what makes the figure correspond to the configuration that was requested rather
// than to whichever fallback was printed last.
func scanFitBreakdown(r io.Reader) (total uint64, clean bool) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var current uint64
	var sawRow bool
	for scanner.Scan() {
		line := scanner.Text()

		if match := fitBreakdownRegex.FindStringSubmatch(line); match != nil {
			if !sawRow {
				current = 0
			}
			if mib, err := strconv.ParseUint(match[1], 10, 64); err == nil {
				current += mib * 1024 * 1024
				sawRow = true
			}
			continue
		}
		if sawRow {
			// The rows of one breakdown are consecutive, so the first line that is not a
			// row ends it. The largest is kept rather than the latest: a later breakdown
			// is not necessarily a better one, and taking the last silently reported a
			// draft model's 16635 MiB as the cost of a load that used 26982.
			total, sawRow = max(total, current), false
		}
		switch {
		case fitTooSmallRegex.MatchString(line):
			return total, false
		case fitCleanRegex.MatchString(line):
			return total, total > 0
		}
	}
	if sawRow {
		total = current
	}
	return total, false
}
