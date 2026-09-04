package server

import (
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestFrameRingDropsFramesPastTheWindow(t *testing.T) {
	r := newFrameRing(time.Minute)
	now := time.Now()

	r.add(retainedFrame{at: now.Add(-90 * time.Second), kind: "sample"})
	r.add(retainedFrame{at: now.Add(-30 * time.Second), kind: "sample"})
	r.add(retainedFrame{at: now, kind: EventLoadStart})

	frames, _ := r.since(time.Hour)
	if len(frames) != 2 {
		t.Errorf("got %d frames, want 2 (the 90s-old one is past a 60s window)", len(frames))
	}
}

// A client asking for more history than exists must be given what there is. Padding it, or
// silently returning less without saying so, both end up drawn as a continuous line across
// a period nobody measured.
func TestFrameRingReportsHowFarBackItReaches(t *testing.T) {
	r := newFrameRing(time.Hour)
	r.add(retainedFrame{at: time.Now().Add(-20 * time.Second), kind: "sample"})
	r.add(retainedFrame{at: time.Now(), kind: "sample"})

	frames, reach := r.since(time.Hour)
	if len(frames) != 2 {
		t.Fatalf("got %d frames, want 2", len(frames))
	}
	if reach < 19*time.Second || reach > 25*time.Second {
		t.Errorf("reach = %v, want about 20s", reach)
	}

	recent, _ := r.since(5 * time.Second)
	if len(recent) != 1 {
		t.Errorf("since(5s) returned %d frames, want 1", len(recent))
	}
}

func TestFrameRingEmpty(t *testing.T) {
	frames, reach := newFrameRing(time.Minute).since(time.Hour)
	if frames != nil || reach != 0 {
		t.Errorf("empty ring returned frames=%v reach=%v", frames, reach)
	}
}

// The body is compared to decide whether anything changed, so equal bodies must compare
// equal and a moved number must not.
func TestEncodeForCompareDetectsChange(t *testing.T) {
	a := &api.ProcessResponse{Models: []api.ProcessModelResponse{{Name: "m", SizeVRAM: 100}}}
	b := &api.ProcessResponse{Models: []api.ProcessModelResponse{{Name: "m", SizeVRAM: 100}}}
	c := &api.ProcessResponse{Models: []api.ProcessModelResponse{{Name: "m", SizeVRAM: 200}}}

	if encodeForCompare(a) != encodeForCompare(b) {
		t.Error("identical bodies compared unequal, so the sampler would never go idle")
	}
	if encodeForCompare(a) == encodeForCompare(c) {
		t.Error("a changed size_vram compared equal, so a filling KV cache would be missed")
	}
	if encodeForCompare(nil) != "" {
		t.Error("nil body should encode to empty")
	}
}
