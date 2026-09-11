package server

import (
	"math"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

var (
	rtxPro  = ml.DeviceID{ID: "0", Library: "CUDA"}
	rtx3090 = ml.DeviceID{ID: "1", Library: "CUDA"}
)

const (
	gbps1792 = 1_792_000_000_000 // RTX PRO 6000, GDDR7
	gbps936  = 936_000_000_000   // RTX 3090, GDDR6X
)

func denseModel(gpus []ml.DeviceID, mem map[ml.DeviceID]api.MemoryBreakdown, bw map[ml.DeviceID]uint64) loadedModel {
	var vram int64
	for _, m := range mem {
		vram += m.Total()
	}
	return loadedModel{
		gpus: gpus, memByGPU: mem, bandwidthByGPU: bw,
		pciByGPU: map[ml.DeviceID]string{rtxPro: "0000:01:00.0", rtx3090: "0000:03:00.0"},
		size:     vram, sizeVRAM: vram, grantedCtxTotal: 131072,
	}
}

func TestRooflineOnOneDeviceIsBytesOverBandwidth(t *testing.T) {
	r := modelRoofline(denseModel(
		[]ml.DeviceID{rtxPro},
		map[ml.DeviceID]api.MemoryBreakdown{rtxPro: {Weights: 2_000_000_000}},
		map[ml.DeviceID]uint64{rtxPro: gbps1792}))

	if r == nil || r.Unavailable != "" {
		t.Fatalf("roofline = %+v, want a ceiling", r)
	}
	if want := 896.0; math.Abs(r.CeilingTokensPerSec-want) > 0.01 {
		t.Errorf("ceiling = %.2f tok/s, want %.2f (1792 GB/s over 2 GB per token)", r.CeilingTokensPerSec, want)
	}
	if r.Basis != "dense_weights" {
		t.Errorf("basis = %q", r.Basis)
	}
}

// A layer split runs its devices in sequence, one token through each card's layers in turn,
// so step time is the SUM of per-device times. On identical cards that equals total bytes over
// the shared bandwidth, which is why a test on identical cards could not tell the two formulas
// apart. On mismatched cards they differ, and the average hides the slow card.
func TestRooflineSumsSplitDevicesRatherThanAveragingThem(t *testing.T) {
	r := modelRoofline(denseModel(
		[]ml.DeviceID{rtxPro, rtx3090},
		map[ml.DeviceID]api.MemoryBreakdown{
			rtxPro:  {Weights: 10_000_000_000},
			rtx3090: {Weights: 10_000_000_000},
		},
		map[ml.DeviceID]uint64{rtxPro: gbps1792, rtx3090: gbps936}))

	sum := 1 / (10e9/gbps1792 + 10e9/gbps936)
	average := 20e9 / ((gbps1792 + gbps936) / 2.0)
	average = 1 / average
	if math.Abs(r.CeilingTokensPerSec-sum) > 0.01 {
		t.Errorf("ceiling = %.3f tok/s, want %.3f (sum of per-device times)", r.CeilingTokensPerSec, sum)
	}
	if math.Abs(sum-average) < 1 {
		t.Fatal("fixture: the two formulas must differ on these cards for the test to mean anything")
	}
	if len(r.Devices) != 2 || r.Devices[1].MemoryBandwidth != gbps936 {
		t.Errorf("per-device inputs missing or wrong: %+v", r.Devices)
	}
}

// Recurrent state is read and rewritten every token and does not grow with context, so it
// belongs with the weights; the KV cache is the per-context-token term.
func TestRooflineCountsRecurrentStateAsAPerTokenRead(t *testing.T) {
	r := modelRoofline(denseModel(
		[]ml.DeviceID{rtxPro},
		map[ml.DeviceID]api.MemoryBreakdown{rtxPro: {Weights: 2_000_000_000, RecurrentState: 700_000_000, KVCache: 1_048_576_000}},
		map[ml.DeviceID]uint64{rtxPro: gbps1792}))

	if r.BytesPerToken != 2_700_000_000 {
		t.Errorf("bytes per token = %d, want weights + recurrent state", r.BytesPerToken)
	}
	if want := uint64(1_048_576_000 / 131072); r.KVBytesPerContextToken != want {
		t.Errorf("kv bytes per context token = %d, want %d (cache over the allocated context)", r.KVBytesPerContextToken, want)
	}
}

// A sliding-window model's windowed layers stop growing at the window, so its KV does not follow
// a per-token rate and the term is omitted rather than reported wrong.
func TestRooflineOmitsTheKVRateForSlidingWindow(t *testing.T) {
	v := denseModel(
		[]ml.DeviceID{rtxPro},
		map[ml.DeviceID]api.MemoryBreakdown{rtxPro: {Weights: 2_000_000_000, KVCache: 1_000_000_000}},
		map[ml.DeviceID]uint64{rtxPro: gbps1792})
	v.slidingWindow = true

	r := modelRoofline(v)
	if r.CeilingTokensPerSec == 0 {
		t.Fatal("the empty-context ceiling still applies to a sliding-window model")
	}
	if r.KVBytesPerContextToken != 0 || r.Devices[0].KVBytesPerContextToken != 0 {
		t.Errorf("kv rate reported for a sliding-window model: %+v", r)
	}
}

func TestRooflineSaysWhyItCannotBound(t *testing.T) {
	base := func() loadedModel {
		return denseModel(
			[]ml.DeviceID{rtxPro},
			map[ml.DeviceID]api.MemoryBreakdown{rtxPro: {Weights: 2_000_000_000}},
			map[ml.DeviceID]uint64{rtxPro: gbps1792})
	}
	moe := base()
	moe.expertCount = 128
	spilled := base()
	spilled.size = spilled.sizeVRAM + 1_000_000_000
	unknown := base()
	unknown.bandwidthByGPU = map[ml.DeviceID]uint64{}

	for want, v := range map[string]loadedModel{
		// A token reads only its active experts, so a dense bound would sit BELOW measured speed.
		"mixture_of_experts": moe,
		"partly_on_cpu":      spilled,
		"bandwidth_unknown":  unknown,
	} {
		r := modelRoofline(v)
		if r == nil || r.Unavailable != want {
			t.Errorf("want unavailable=%q, got %+v", want, r)
			continue
		}
		if r.CeilingTokensPerSec != 0 || r.BytesPerToken != 0 {
			t.Errorf("%s: a ceiling was reported alongside the reason there is none: %+v", want, r)
		}
	}

	if r := modelRoofline(loadedModel{}); r != nil {
		t.Errorf("a model on no GPU has nothing to bound; got %+v", r)
	}
}
