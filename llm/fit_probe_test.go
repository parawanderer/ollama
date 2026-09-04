package llm

import (
	"strings"
	"testing"
)

// Lines as llama-server actually prints them, from a two-device host.
const fitProbeLog = `0.00.230.158 I common_params_fit_impl: getting device memory data for initial parameters:
0.00.477.713 I common_memory_breakdown_print: | memory breakdown [MiB]                                 | total    free    self   model   context   compute    unaccounted |
0.00.477.726 I common_memory_breakdown_print: |   - CUDA0 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 52266 + ( 359 =   212 +      66 +      80) +       44623 |
0.00.477.726 I common_memory_breakdown_print: |   - CUDA1 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 96644 + ( 746 =   551 +     106 +      88) +        -141 |
0.00.514.752 I common_params_fit_impl: projected memory use with initial parameters [MiB]:
0.00.514.757 I common_params_fit_impl: targets for free memory can be met on all devices, no changes needed
`

func TestScanFitBreakdownSumsDevices(t *testing.T) {
	total, adjusted := scanFitBreakdown(strings.NewReader(fitProbeLog))
	if adjusted {
		t.Error("reported an adjustment where the log says no changes were needed")
	}
	// 359 + 746 MiB: what this load would hold, summed across both devices.
	if want := uint64(1105) * mib; total != want {
		t.Errorf("total = %d MiB, want %d MiB", total/mib, want/mib)
	}
}

// A single-device breakdown, from a load pinned with --split-mode none. Getting this
// wrong is not a rounding error: counting a second device's compute buffers into a
// single-device load put a measured probe 39% over the truth.
func TestScanFitBreakdownSingleDevice(t *testing.T) {
	log := `0.00.408.024 I common_memory_breakdown_print: |   - CUDA0 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 12510 + (83368 = 78056 +    4336 +     975) +        1370 |
0.00.450.419 I common_params_fit_impl: projected to use 83368 MiB of device memory
`
	total, _ := scanFitBreakdown(strings.NewReader(log))
	if want := uint64(83368) * mib; total != want {
		t.Errorf("total = %d MiB, want %d MiB", total/mib, want/mib)
	}
}

// The pass prints a breakdown per candidate configuration. Only the last describes what
// it settled on, so the totals must replace rather than accumulate -- summing them would
// report several times the real figure and refuse loads that fit.
func TestScanFitBreakdownKeepsOnlyTheLastPass(t *testing.T) {
	log := `common_memory_breakdown_print: |   - CUDA0 (GPU) | 97249 = 12510 + (90000 = 88000 +    1000 +    1000) +        1370 |
common_params_fit_impl: reducing the context size to fit
common_memory_breakdown_print: |   - CUDA0 (GPU) | 97249 = 12510 + (40000 = 38000 +    1000 +    1000) +        1370 |
common_params_fit_impl: projected to use 40000 MiB of device memory
`
	total, adjusted := scanFitBreakdown(strings.NewReader(log))
	if want := uint64(40000) * mib; total != want {
		t.Errorf("total = %d MiB, want %d MiB", total/mib, want/mib)
	}
	if !adjusted {
		t.Error("did not notice that the parameters were changed, so the figure would be filed against a load that was never requested")
	}
}

func TestScanFitBreakdownWithoutABreakdown(t *testing.T) {
	if total, _ := scanFitBreakdown(strings.NewReader("nothing to see here\n")); total != 0 {
		t.Errorf("invented a total of %d from a log with no breakdown", total)
	}
}
