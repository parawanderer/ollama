package server

import (
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	// sampleFastInterval is used while something is changing: a model loading, or a body
	// that differs from the last one. A KV cache filling during a long generation is a
	// level with no discrete event behind it, and it is most of what a resource panel
	// exists to show, so it has to be sampled rather than waited for.
	sampleFastInterval = 1 * time.Second

	// sampleIdleInterval is used when nothing has changed. An idle box emitting a frame a
	// second forever is a lot of traffic describing nothing.
	sampleIdleInterval = 15 * time.Second

	// sampleLoadingInterval is used while a load is in flight. A load is the only time
	// device memory moves by tens of gigabytes, and it moves in two steps -- the weights,
	// then the context -- about a second apart on a warm load. Sampling at one second can
	// land between them and render that as a single jump, so the interval that matters for
	// drawing the climb is shorter than the one that matters for everything else.
	sampleLoadingInterval = 250 * time.Millisecond

	// retainedWindow is how much history is kept for a client that reconnects. A dropped
	// stream is total blindness until reconnect, unlike a failed poll which loses one
	// sample, so some backfill is the difference between a gap and a hole.
	retainedWindow = 10 * time.Minute
)

// retainedFrame is one frame as it happened, held for backfill. Times are absolute here
// and converted to a per-connection offset when served, since each connection anchors on
// its own hello.
type retainedFrame struct {
	at     time.Time
	kind   string
	model  string
	reason string
	ps     *api.ProcessResponse
	info   *api.InfoResponse
}

type frameRing struct {
	mu     sync.Mutex
	frames []retainedFrame
	window time.Duration
}

func newFrameRing(window time.Duration) *frameRing {
	return &frameRing{window: window}
}

func (r *frameRing) add(f retainedFrame) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.frames = append(r.frames, f)

	// Drop anything past the window. The slice is in time order, so this is a prefix.
	cutoff := f.at.Add(-r.window)
	keep := 0
	for keep < len(r.frames) && r.frames[keep].at.Before(cutoff) {
		keep++
	}
	if keep > 0 {
		r.frames = append(r.frames[:0], r.frames[keep:]...)
	}
}

// since returns the frames newer than the given duration ago, and how far back the ring
// actually reaches. A caller asking for more history than exists gets what there is: the
// missing part must stay missing rather than being implied by a line drawn across it.
func (r *frameRing) since(d time.Duration) ([]retainedFrame, time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.frames) == 0 {
		return nil, 0
	}

	cutoff := time.Now().Add(-d)
	out := make([]retainedFrame, 0, len(r.frames))
	for _, f := range r.frames {
		if !f.at.Before(cutoff) {
			out = append(out, f)
		}
	}
	return out, time.Since(r.frames[0].at)
}

// startSampler publishes periodic snapshots until ctx is done.
//
// Cadence adapts: fast while a load is in flight or the body is changing, idle otherwise.
// The idle tick still emits, so a client can tell "nothing is happening" from "the stream
// died" without waiting for a heartbeat to disambiguate it.
func (s *Scheduler) startSampler(done <-chan struct{}) {
	go func() {
		var lastPS, lastInfo string
		timer := time.NewTimer(sampleFastInterval)
		defer timer.Stop()

		for {
			select {
			case <-done:
				return
			case <-timer.C:
			case <-s.samplerWake:
				// Something moved memory. Read it now; the cadence below still decides
				// when the reading after this one happens.
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
			}

			interval := sampleIdleInterval
			if s.psFn != nil {
				ps := s.psFn()
				if encoded := encodeForCompare(ps); encoded != lastPS {
					lastPS = encoded
					interval = sampleFastInterval
				}
				loading := s.anyRunnerLoading()
				if loading {
					interval = sampleLoadingInterval
				}

				// Capacity is sent when it has moved. During a load the model list does
				// not change at all -- no runner exists yet -- so device free memory is
				// the only thing that shows the memory arriving.
				var info *api.InfoResponse
				if s.infoFn != nil {
					// No refresh here: every reported figure goes through
					// devicesWithLiveFreeMemory, so a fast tick already reads the driver
					// rather than restating a cached number.
					if candidate := s.infoFn(); encodeInfoForCompare(candidate) != lastInfo {
						lastInfo = encodeInfoForCompare(candidate)
						info = candidate
					}
				}
				s.publishSample(ps, info)
			}
			timer.Reset(interval)
		}
	}()
}

// anyRunnerLoading reports whether a load is in flight, counted rather than looked up: the
// runner object does not exist for most of a load, so the runner list would answer no for
// the period that matters most.
func (s *Scheduler) anyRunnerLoading() bool {
	return s.loadsInFlight.Load() > 0
}
