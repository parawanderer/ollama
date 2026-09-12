package server

import (
	"testing"

	"github.com/ollama/ollama/api"
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

// measureAs answers with a fixed cost per device count, and counts how often it was asked.
func measureAs(t *testing.T, oneCard, twoCards uint64, calls *int) func([]ml.DeviceInfo, api.Options, uint64) (api.Options, llm.CalibrationKey, uint64, bool) {
	return func(placed []ml.DeviceInfo, opts api.Options, _ uint64) (api.Options, llm.CalibrationKey, uint64, bool) {
		*calls++
		if *calls > 3 {
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
		name               string
		estimate           uint64
		oneCard, twoCards  uint64
		wantCards, wantMsr int
		wantPredicted      uint64
	}{
		// Metadata too low: it would pin the model to one card, where it can only spill.
		{"an under-estimate moves to two cards", 40 * gib, 100 * gib, 120 * gib, 2, 2, 120 * gib},
		// Metadata far too high (gemma4:31b's split): the measurement brings it back to one.
		{"an over-estimate moves back to one card", 400 * gib, 43 * gib, 44 * gib, 1, 2, 43 * gib},
		{"agreement measures once", 40 * gib, 43 * gib, 44 * gib, 1, 1, 43 * gib},
		// Two measurements that disagree with each other do not chase: the second placement stands.
		{"moves at most once", 40 * gib, 100 * gib, 30 * gib, 2, 2, 30 * gib},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls int
			gpus, _, key, predicted, probed := settlePlacement(tc.estimate, placeByEstimate, measureAs(t, tc.oneCard, tc.twoCards, &calls))
			if len(gpus) != tc.wantCards || calls != tc.wantMsr || predicted != tc.wantPredicted {
				t.Fatalf("cards %d (want %d), measured %d times (want %d), predicted %d GiB (want %d)",
					len(gpus), tc.wantCards, calls, tc.wantMsr, predicted/gib, tc.wantPredicted/gib)
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

func TestSettlePlacementKeepsAnEarlierProbe(t *testing.T) {
	calls := 0
	measure := func(placed []ml.DeviceInfo, opts api.Options, _ uint64) (api.Options, llm.CalibrationKey, uint64, bool) {
		calls++
		if len(placed) == 1 {
			return opts, llm.CalibrationKey{}, 100 * gib, true // probed here
		}
		return opts, llm.CalibrationKey{}, 120 * gib, false // already calibrated
	}
	if _, _, _, _, probed := settlePlacement(40*gib, placeByEstimate, measure); !probed {
		t.Fatal("the first placement's probe was forgotten after the move")
	}
}
