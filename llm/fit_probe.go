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
var (
	fitBreakdownRegex = regexp.MustCompile(`memory_breakdown_print:.*\|\s+-\s+\S+\s+\([^)]*\)\s+\|\s+\d+\s+=\s+\d+\s+\+\s+\(\s*(\d+)\s+=`)
	fitConcludedRegex = regexp.MustCompile(`common_params_fit_impl: (projected to use|targets for free memory)`)
	fitAdjustedRegex  = regexp.MustCompile(`common_params_fit_impl: (reducing|decreasing|increasing|adjusting|moving)`)
)

// fitProbeTimeout bounds one probe. Measured cost is 0.4-0.9s including process start,
// on models from 1 to 104 GiB -- the pass reads metadata and reserves graphs, never
// tensors, so size barely moves it. The timeout is generous against a cold page cache
// and exists only so a probe cannot become the thing that delays a load.
const fitProbeTimeout = 30 * time.Second

// ErrFitProbeAdjusted reports that llama-server changed the parameters it was asked to
// measure, so the figure it printed describes a different load than the one requested.
var ErrFitProbeAdjusted = errors.New("fit probe adjusted the parameters it was asked to measure")

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
	// else. Batch size and split mode both change the compute buffers, and a probe that
	// spreads a single-device load across two devices counts them twice -- that alone
	// put a measured probe 39% over the truth.
	opts.NumCtx = numCtx
	launch, err := newLlamaServerLaunchConfig(gpus, modelPath, f, adapters, projectors, opts, numParallel, kvCacheType, config, newLlamaServerMediaMarker())
	if err != nil {
		return 0, err
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

	total, adjusted := scanFitBreakdown(out)
	close(done)
	_ = cmd.Process.Kill()
	_, _ = io.Copy(io.Discard, out)
	_ = cmd.Wait()

	switch {
	case adjusted:
		return 0, ErrFitProbeAdjusted
	case total == 0:
		if err := ctx.Err(); err != nil {
			return 0, fmt.Errorf("fit probe did not report in %s: %w", fitProbeTimeout, err)
		}
		return 0, errors.New("fit probe produced no memory breakdown")
	}

	slog.Debug("probed model memory without loading it", "model", modelPath, "num_ctx", numCtx, "vram", total)
	return total, nil
}

// scanFitBreakdown sums the per-device figures of the last breakdown printed, and reports
// whether llama-server changed the parameters it was measuring. The pass prints a
// breakdown per candidate configuration, so only the final one describes what it settled
// on; each new breakdown therefore replaces the running total rather than adding to it.
func scanFitBreakdown(r io.Reader) (total uint64, adjusted bool) {
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
			// row ends it.
			total, sawRow = current, false
		}
		if fitAdjustedRegex.MatchString(line) {
			adjusted = true
		}
		if fitConcludedRegex.MatchString(line) && total > 0 {
			return total, adjusted
		}
	}
	if sawRow {
		total = current
	}
	return total, adjusted
}
