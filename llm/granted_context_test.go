package llm

import (
	"fmt"
	"io"
	"strings"
	"testing"
)

// The log lines below are transcribed verbatim from granite4.1:3b loaded with num_ctx 12345 on
// 2026-09-11 -- including the column padding, which the regexes have to survive. Only the
// numbers are varied, and only where a test needs the dry-run and the real pass to disagree:
// with ollama passing -c they agree, so the real capture cannot show the case that matters
// once it stops (the fit pass measures at the initial parameters, then reduces them).

func fitDryRun(ctx int) string {
	return fmt.Sprintf(`load_tensors:        CUDA0 model buffer size =     0.00 MiB
load_tensors:    CUDA_Host model buffer size =     0.00 MiB
llama_context: n_seq_max             = 1
llama_context: n_ctx                 = %[1]d
llama_context: n_ctx_seq             = %[1]d
llama_context: n_ctx_seq (%[1]d) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
common_params_fit_impl: projected to use 3129 MiB of device memory vs. 96690 MiB of free device memory
common_params_fit_impl: will leave 93560 >= 1024 MiB of free device memory, no changes needed
common_fit_params: successfully fit params to free device memory
`, ctx)
}

func realLoad(slots, total, perSlot int) string {
	return fmt.Sprintf(`print_info: n_ctx_train           = 131072
load_tensors:   CPU_Mapped model buffer size =   200.98 MiB
load_tensors:        CUDA0 model buffer size =  1998.84 MiB
llama_context: n_seq_max             = %d
llama_context: n_ctx                 = %d
llama_context: n_ctx_seq             = %d
`, slots, total, perSlot)
}

// feed writes line by line, as a pipe usually delivers them.
func feed(t *testing.T, r *llamaServerRunner, log string) {
	t.Helper()
	w := &memoryParsingWriter{inner: io.Discard, runner: r}
	for _, line := range strings.SplitAfter(log, "\n") {
		if _, err := w.Write([]byte(line)); err != nil {
			t.Fatal(err)
		}
	}
}

// The case step 2 of the estimator depends on: the fit pass measures at the initial context
// (here 262144) and the real load runs at the reduced one (82944, a figure llama.cpp
// actually chose in an earlier experiment). Reading the first line seen would report the
// context that was rejected.
func TestGrantedContextIsTheServingContextNotTheFitDryRun(t *testing.T) {
	r := &llamaServerRunner{}
	feed(t, r, fitDryRun(262144)+realLoad(1, 82944, 82944))

	if seq, total := r.GrantedContext(); seq != 82944 || total != 82944 {
		t.Errorf("GrantedContext = (%d, %d), want (82944, 82944); 262144 is the fit pass's "+
			"dry run at the initial parameters, which it then rejected", seq, total)
	}
}

// One Write can carry many lines. If the dry-run's context line and the first real buffer
// arrive together, "the real load began in this chunk" must not license the dry-run line
// that came before it in the same chunk.
func TestGrantedContextJudgesLinesByPositionWithinAChunk(t *testing.T) {
	r := &llamaServerRunner{}
	w := &memoryParsingWriter{inner: io.Discard, runner: r}
	if _, err := w.Write([]byte(fitDryRun(262144) + realLoad(1, 82944, 82944))); err != nil {
		t.Fatal(err)
	}

	if seq, _ := r.GrantedContext(); seq != 82944 {
		t.Errorf("per-slot = %d, want 82944; the dry-run line was accepted because the real "+
			"load started later in the same write", seq)
	}
}

// Real capture, two slots, num_ctx 12345: llama.cpp rounds EACH slot to a multiple of 256 and
// multiplies back, 2 x 12544 = 25088. Rounding the total instead gives 24832, 12416 per slot --
// which is why this is read rather than computed, and why per-slot and total are kept apart:
// /api/ps reports per slot, calibration's axis is the total.
func TestGrantedContextKeepsPerSlotAndTotalApart(t *testing.T) {
	r := &llamaServerRunner{}
	feed(t, r, fitDryRun(12544)+realLoad(2, 25088, 12544))

	seq, total := r.GrantedContext()
	if seq != 12544 {
		t.Errorf("per slot = %d, want 12544", seq)
	}
	if total != 25088 {
		t.Errorf("total = %d, want 25088; filing calibration at the per-slot figure would put "+
			"every multi-slot sample at half its real context", total)
	}
}

// A draft model builds its own context after the main one. The main model's is the one that
// serves requests; taking the most recent line would report the draft's.
func TestGrantedContextIgnoresADraftModelsContext(t *testing.T) {
	r := &llamaServerRunner{}
	feed(t, r, fitDryRun(32768)+realLoad(1, 32768, 32768)+realLoad(1, 4096, 4096))

	if seq, total := r.GrantedContext(); seq != 32768 || total != 32768 {
		t.Errorf("GrantedContext = (%d, %d), want the main model's (32768, 32768), not the "+
			"draft model's that followed it", seq, total)
	}
}

// Neither the training context nor the advisory line describes what was allocated.
func TestGrantedContextIgnoresTrainingContextAndTheAdvisory(t *testing.T) {
	r := &llamaServerRunner{}
	feed(t, r, `print_info: n_ctx_train           = 131072
load_tensors:        CUDA0 model buffer size =  1998.84 MiB
llama_context: n_ctx_seq (12544) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
`)
	if seq, total := r.GrantedContext(); seq != 0 || total != 0 {
		t.Errorf("GrantedContext = (%d, %d), want (0, 0): nothing here states the allocated context", seq, total)
	}
}

// TestContextLengthStaysWhatWasAsked guards the bug that the obvious version of this change
// would have introduced. The scheduler overwrites the runner's options with ContextLength()
// after a load, and decides reuse by comparing those options to the next request's with
// reflect.DeepEqual. Were ContextLength to return the granted 12544, the runner would record
// 12544, every later request for 12345 would compare unequal, and the model would reload on
// every single request -- without an error anywhere, only slowness.
func TestContextLengthStaysWhatWasAsked(t *testing.T) {
	r := &llamaServerRunner{}
	r.options.NumCtx = 12345
	feed(t, r, fitDryRun(12544)+realLoad(1, 12544, 12544))

	if got := r.ContextLength(); got != 12345 {
		t.Errorf("ContextLength = %d, want the requested 12345; runner reuse compares this "+
			"against the next request, and the granted figure never equals it", got)
	}
	if seq, _ := r.GrantedContext(); seq != 12544 {
		t.Errorf("GrantedContext per slot = %d, want 12544", seq)
	}
}
