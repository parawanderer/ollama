package llm

import (
	"strings"
	"testing"
)

// Every fixture below is transcribed from llama-server's real output on a two-card host.
// The previous version of this test used lines written from memory, and the adjustment
// detection it validated matched none of the phrasings llama-server actually prints.

// A load that fits: the pass considers one configuration and confirms it.
const fitProbeCleanLog = `0.00.410.056 I common_params_fit_impl: getting device memory data for initial parameters:
0.00.584.169 I common_memory_breakdown_print: | memory breakdown [MiB]                                 | total    free     self   model   context   compute    unaccounted |
0.00.584.182 I common_memory_breakdown_print: |   - CUDA0 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 96628 + (40771 = 40277 +     191 +     302) +      -40150 |
0.00.584.182 I common_memory_breakdown_print: |   - CUDA1 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 96634 + (38398 = 37778 +     184 +     434) +      -37782 |
0.00.584.183 I common_memory_breakdown_print: |   - Host                                               |                  28173 = 28110 +       0 +      63                |
0.00.618.892 I common_params_fit_impl: projected to use 79170 MiB of device memory vs. 193262 MiB of free device memory
0.00.618.892 I common_params_fit_impl: targets for free memory can be met on all devices, no changes needed
`

// The same model against memory it does not fit in. llama-server does not fail here: it
// prints a fresh, smaller breakdown for each fallback it tries.
const fitProbeTooSmallLog = `0.00.645.129 I common_memory_breakdown_print: |   - CUDA0 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 96628 + (40771 = 40277 +     191 +     302) +      -40150 |
0.00.645.129 I common_memory_breakdown_print: |   - CUDA1 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 96634 + (38398 = 37778 +     184 +     434) +      -37782 |
0.00.679.672 I common_params_fit_impl: projected to use 79170 MiB of device memory vs. 193262 MiB of free device memory
0.00.679.673 I common_params_fit_impl: cannot meet free memory target of 90000 MiB, need to reduce device memory by 161235 MiB
0.00.679.678 I common_params_fit_impl: getting device memory data with all MoE tensors moved to system memory:
0.00.891.744 I common_memory_breakdown_print: |   - CUDA0 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 96628 + (  3575 =   2077 +     191 +    1305) +       -2954 |
0.00.891.744 I common_memory_breakdown_print: |   - CUDA1 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 96634 + (  2881 =   2528 +     184 +     168) +       -2266 |
0.00.925.798 I common_params_fit_impl: with only dense weights in device memory there is a total surplus of 6805 MiB
0.01.144.645 I common_memory_breakdown_print: |   - CUDA0 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 96688 + (  1053 =      0 +       0 +    1053) +        -492 |
0.01.144.645 I common_memory_breakdown_print: |   - CUDA1 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 96688 + (     0 =      0 +       0 +       0) +         561 |
0.01.179.394 I common_params_fit_impl: filling dense-only layers back-to-front:
`

func TestScanFitBreakdownSumsDevices(t *testing.T) {
	total, clean := scanFitBreakdown(strings.NewReader(fitProbeCleanLog), false)
	if !clean {
		t.Error("rejected a fit the log says needed no changes")
	}
	// 40771 + 38398 MiB. The Host row is excluded: it is the mmap'd view of the weights,
	// whose span overlaps the device copy, so adding it would count the model file twice.
	if want := uint64(79169) * mib; total != want {
		t.Errorf("total = %d MiB, want %d MiB", total/mib, want/mib)
	}
}

// The case the whole verdict check exists for. Every breakdown after the first describes a
// fallback nobody asked for, and the smallest of them reports about a gigabyte for a
// hundred-gigabyte model -- a figure that would otherwise be recorded and persisted.
func TestScanFitBreakdownRejectsAModelThatDoesNotFit(t *testing.T) {
	_, clean := scanFitBreakdown(strings.NewReader(fitProbeTooSmallLog), false)
	if clean {
		t.Fatal("accepted a probe of a model llama-server was busy moving to system memory")
	}
}

// A single-device breakdown, from a load pinned with --split-mode none. Getting this wrong
// is not a rounding error: counting a second device's compute buffers into a single-device
// load put a measured probe 39% over the truth.
// The degraded pass never prints a clean verdict, so a truncated or unrecognised
// giving-up message still rejects. Measured: "no changes needed" appears zero times in a
// log where the fit pass spent 15s falling back to 0 layers on device.
func TestScanFitBreakdownRejectsDegradationItCannotName(t *testing.T) {
	log := `common_memory_breakdown_print: |   - CUDA0 (GPU) | 97249 = 96628 + (78688 = 78056 + 376 + 255) + -78012 |
common_params_fit_impl: some future wording for giving up
common_memory_breakdown_print: |   - CUDA0 (GPU) | 97249 = 96688 + ( 1053 = 0 + 0 + 1053) + -492 |
common_params_fit_impl: id=0, n_layer= 0, n_part= 0, overflow_type=4, mem=  1053 MiB
`
	if _, clean := scanFitBreakdown(strings.NewReader(log), false); clean {
		t.Error("accepted a degraded pass because it did not recognise the giving-up message")
	}
}

func TestScanFitBreakdownSingleDevice(t *testing.T) {
	log := `0.00.585.056 I common_memory_breakdown_print: |   - CUDA0 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 96574 + (78688 = 78056 +     376 +     255) +      -78012 |
0.00.585.057 I common_memory_breakdown_print: |   - Host                                               |                  28137 = 28110 +       0 +      27                |
0.00.621.366 I common_params_fit_impl: will leave 17886 >= 1024 MiB of free device memory, no changes needed
`
	total, clean := scanFitBreakdown(strings.NewReader(log), false)
	if !clean {
		t.Error("rejected a clean single-device fit")
	}
	if want := uint64(78688) * mib; total != want {
		t.Errorf("total = %d MiB, want %d MiB", total/mib, want/mib)
	}
}

// A breakdown with no verdict after it is not a measurement. Truncated output means the
// process was killed mid-pass, and what it had printed so far may be a candidate it was
// about to reject.
func TestScanFitBreakdownRequiresAVerdict(t *testing.T) {
	log := `0.00.585.056 I common_memory_breakdown_print: |   - CUDA0 (GPU) | 97249 = 96574 + (78688 = 78056 + 376 + 255) + -78012 |
`
	if _, clean := scanFitBreakdown(strings.NewReader(log), false); clean {
		t.Error("accepted a breakdown the fit pass never reached a verdict on")
	}
}

// The verdict is worded three ways depending on how the fit was framed, and only the tail
// is common to all of them. An earlier version matched the multi-device wording alone and
// rejected every single-device probe -- the common case -- while its own fixture wrongly
// carried the multi-device line, so the test passed and the deployment did not.
func TestScanFitBreakdownAcceptsEveryVerdictPhrasing(t *testing.T) {
	row := `common_memory_breakdown_print: |   - CUDA0 (GPU) | 97249 = 96574 + (78688 = 78056 + 376 + 255) + -78012 |` + "\n"
	for name, verdict := range map[string]string{
		"multi device": "common_params_fit_impl: targets for free memory can be met on all devices, no changes needed",
		"one device":   "common_params_fit_impl: will leave 17886 >= 1024 MiB of free device memory, no changes needed",
		"cpu only":     "common_params_fit_impl: will leave 123126 >= 1024 MiB of system memory, no changes needed",
	} {
		t.Run(name, func(t *testing.T) {
			if _, clean := scanFitBreakdown(strings.NewReader(row+verdict+"\n"), false); !clean {
				t.Errorf("rejected a fit the log says needed no changes: %q", verdict)
			}
		})
	}
}

func TestScanFitBreakdownWithoutABreakdown(t *testing.T) {
	if total, _ := scanFitBreakdown(strings.NewReader("nothing to see here\n"), false); total != 0 {
		t.Errorf("invented a total of %d from a log with no breakdown", total)
	}
}

// A vision model, transcribed from qwen2.5vl:3b at 8k. Two of the three things it holds are
// outside the breakdown: the projector's worst-case reservation is printed before the fit
// verdict, and the vision encoder's graph is reserved afterwards. Reading only the breakdown
// gives 2291 MiB for a load that used 5207.
const fitProbeVisionLog = `0.00.353.024 I srv    load_model: [mtmd] estimated worst-case memory usage of mmproj is 2211.44 MiB (took 51.65 ms)
0.00.486.599 I common_memory_breakdown_print: |   - CUDA0 (RTX PRO 6000 Blackwell Workstation Edition) | 97249 = 91836 + (2291 =  1834 +     288 +     169) +        3120 |
0.00.486.599 I common_memory_breakdown_print: |   - Host                                               |                   275 =   243 +       0 +      32                |
0.00.570.000 I common_params_fit_impl: will leave 89545 >= 1024 MiB of free device memory, no changes needed
0.00.856.041 I clip_ctx: CLIP using CUDA0 backend
0.01.030.317 I reserve_compute_meta:      CUDA0 compute buffer size =   704.79 MiB
0.01.030.323 I reserve_compute_meta:        CPU compute buffer size =   292.41 MiB
`

func TestScanFitBreakdownCountsTheProjectorAndItsGraph(t *testing.T) {
	total, clean := scanFitBreakdown(strings.NewReader(fitProbeVisionLog), true)
	if !clean {
		t.Fatal("rejected a fit the log says needed no changes")
	}

	// 2291 breakdown + 2211.44 projector + 704.79 encoder graph. The CPU-side
	// reserve_compute_meta is excluded for the same reason every host row is: it is not
	// device memory.
	perMiB := float64(mib)
	want := uint64(2291)*mib + uint64(2211.44*perMiB) + uint64(704.79*perMiB)
	if !withinMiB(total, want, 1) {
		t.Errorf("total = %d MiB, want %d MiB (%d MiB was reported by /api/ps for this load)",
			total/mib, want/mib, 5207)
	}
}

// Without a projector the scan must still stop at the verdict rather than waiting for lines
// that will never come -- a text model prints no clip reservation, and waiting for one would
// hang every probe until the timeout.
func TestScanFitBreakdownDoesNotWaitForAProjectorThatIsAbsent(t *testing.T) {
	total, clean := scanFitBreakdown(strings.NewReader(fitProbeCleanLog), false)
	if !clean || total != uint64(79169)*mib {
		t.Errorf("total = %d MiB clean = %v; want 79169 MiB and clean", total/mib, clean)
	}
}
