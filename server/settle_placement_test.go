package server

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

const gib = uint64(1 << 30)

var (
	card0 = ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}}
	card1 = ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}}
)

// One card up to 90 GiB, both above, as selectLlamaServerPlacement decides on this box.
func placeByEstimate(estimate uint64) ([]ml.DeviceInfo, api.Options) {
	if estimate <= 90*gib {
		return []ml.DeviceInfo{card0}, api.Options{}
	}
	return []ml.DeviceInfo{card0, card1}, api.Options{}
}

// measured records one call to the measure callback.
type measured struct {
	cards      int
	decideOnly bool
}

// measureAs answers with a fixed cost per device count, and records every call.
func measureAs(t *testing.T, oneCard, twoCards uint64, calls *[]measured) func([]ml.DeviceInfo, api.Options, uint64, bool) (api.Options, llm.CalibrationKey, uint64, bool) {
	return func(placed []ml.DeviceInfo, opts api.Options, _ uint64, decideOnly bool) (api.Options, llm.CalibrationKey, uint64, bool) {
		*calls = append(*calls, measured{len(placed), decideOnly})
		if len(*calls) > 2 {
			t.Fatal("settlePlacement kept re-measuring; it must move at most once")
		}
		if len(placed) == 1 {
			return opts, llm.CalibrationKey{NumGPU: 1}, oneCard, true
		}
		return opts, llm.CalibrationKey{NumGPU: 2}, twoCards, true
	}
}

func TestSettlePlacement(t *testing.T) {
	for _, tc := range []struct {
		name              string
		estimate          uint64
		oneCard, twoCards uint64
		want              []measured
		wantPredicted     uint64
	}{
		// Metadata too low: it would pin the model to one card, where it can only spill.
		{"an under-estimate moves to two cards", 40 * gib, 100 * gib, 120 * gib,
			[]measured{{1, true}, {2, false}}, 120 * gib},
		// Metadata far too high (gemma4:26b at 262144 on 2026-09-12: 121.7 GiB against
		// 23.5): the measurement brings it back to one card.
		{"an over-estimate moves back to one card", 400 * gib, 43 * gib, 44 * gib,
			[]measured{{2, true}, {1, false}}, 43 * gib},
		{"agreement measures the placement it started on", 40 * gib, 43 * gib, 44 * gib,
			[]measured{{1, true}, {1, false}}, 43 * gib},
		// Two measurements that disagree with each other do not chase: the second placement stands.
		{"moves at most once", 40 * gib, 100 * gib, 30 * gib,
			[]measured{{1, true}, {2, false}}, 30 * gib},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls []measured
			gpus, _, key, predicted, probed := settlePlacement(tc.estimate, placeByEstimate, measureAs(t, tc.oneCard, tc.twoCards, &calls))
			if !slices.Equal(calls, tc.want) || predicted != tc.wantPredicted {
				t.Fatalf("measured %v (want %v), predicted %d GiB (want %d)", calls, tc.want, predicted/gib, tc.wantPredicted/gib)
			}
			// The full measurement is the one the load is filed under, so it must describe
			// where the load runs -- never the placement that was abandoned.
			if last := calls[len(calls)-1]; last.decideOnly || last.cards != len(gpus) {
				t.Errorf("the full measurement was taken on %d cards but the load runs on %d", last.cards, len(gpus))
			}
			if key.NumGPU != len(gpus) {
				t.Errorf("the key describes %d cards but the load runs on %d", key.NumGPU, len(gpus))
			}
			if !probed {
				t.Error("a probe that ran was not reported")
			}
		})
	}
}

// probed labels where the prediction came from, so it describes the final placement only.
// A probe of the placement that was abandoned contributed nothing to the number: after a
// move onto an already-calibrated placement the source is calibration. (This test used to
// assert the opposite, which let the label say "probe" about a calibrated figure.)
func TestSettlePlacementLabelsTheFinalMeasurement(t *testing.T) {
	measure := func(placed []ml.DeviceInfo, opts api.Options, _ uint64, decideOnly bool) (api.Options, llm.CalibrationKey, uint64, bool) {
		if decideOnly {
			return opts, llm.CalibrationKey{}, 100 * gib, true // probed here, then moved away
		}
		return opts, llm.CalibrationKey{}, 120 * gib, false // already calibrated
	}
	if _, _, _, _, probed := settlePlacement(40*gib, placeByEstimate, measure); probed {
		t.Fatal("the abandoned placement's probe was reported as the source of the prediction")
	}
}

// The measurement that decided the placement is one of the line's points, so a placement
// that stands costs no more probes than it did before the decision was added.
func TestProbeCalibrationReusesTheDecidingSample(t *testing.T) {
	f := trainedAt(t, 262144)
	for _, tc := range []struct {
		name        string
		numCtx      int
		premeasured map[int]uint64
		wantProbed  []int
	}{
		{"inside the range", 32768, map[int]uint64{32768: 20 * gib}, []int{131072}},
		{"beyond the range", 262144, map[int]uint64{262144: 30 * gib}, []int{8192, 131072}},
		// A deciding probe that could not measure is not asked again moments later.
		{"a failed decision is not retried", 32768, map[int]uint64{32768: 0}, []int{131072}},
		{"nothing to reuse", 32768, nil, []int{32768, 131072}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var probed []int
			s := &Scheduler{vramCalibration: llm.NewVRAMCalibration()}
			s.fitProbe = func(_ context.Context, _ []ml.DeviceInfo, _ string, _ *ggml.GGML, _, _ []string, _ api.Options, _ int, _ string, _ llm.LlamaServerConfig, numCtx int) (uint64, int, error) {
				probed = append(probed, numCtx)
				return 10*gib + uint64(numCtx)*1024, numCtx, nil
			}
			req := &LlmRequest{ctx: t.Context(), model: &Model{ModelPath: "m"}}
			key := llm.CalibrationKey{Model: "m", NumGPU: 1}

			s.probeCalibration(t.Context(), key, req, f, api.Options{}, []ml.DeviceInfo{card0}, 1, tc.numCtx, tc.premeasured)

			if !slices.Equal(probed, tc.wantProbed) {
				t.Errorf("probed %v, want %v", probed, tc.wantProbed)
			}
			want := len(tc.wantProbed)
			for _, v := range tc.premeasured {
				if v > 0 {
					want++
				}
			}
			if want < 2 {
				want = 0 // one sample is discarded rather than trusted
			}
			if got := s.vramCalibration.SampleCount(key); got != want {
				t.Errorf("%d samples filed, want %d", got, want)
			}
		})
	}
}

func trainedAt(t *testing.T, ctx int) *ggml.GGML {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.gguf")
	w, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := ggml.WriteGGUF(w, ggml.KV{"general.architecture": "llama", "llama.context_length": uint32(ctx)}, nil); err != nil {
		t.Fatal(err)
	}
	w.Close()
	r, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	f, err := ggml.Decode(r, -1)
	if err != nil {
		t.Fatal(err)
	}
	return f
}
