package server

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
)

// Model lifecycle event types. These name transitions a client cannot reconstruct by
// polling: a model evicted to make room for another is gone and replaced between two
// samples, and a load that takes 45 seconds is invisible for all of them.
const (
	EventLoadStart    = "load.start"
	EventLoadComplete = "load.complete"

	// EventLoadWeights divides a load in two. Before it the weights are being read and
	// transferred, which dominates a cold load; after it the context is constructed -- the
	// KV cache and compute buffers, allocated in one step near the end rather than spread
	// across the load, and on a long-context model most of what the model ends up holding.
	EventLoadWeights = "load.weights"
	EventLoadFailed  = "load.failed"
	// EventEstimate reports the placement decision, ahead of the load it decided.
	EventEstimate = "estimate"
	EventEvict    = "evict"
	EventUnload   = "unload"

	// EventExpires reports a keep-alive deadline moving. It is emitted when the deadline
	// is written, which happens only as a request finishes -- so a client that draws a
	// countdown has no way to know from /api/ps alone that the number it is counting from
	// is being held rather than approaching.
	EventExpires = "expires"

	// EventBusyStart and EventBusyEnd bracket a model actually working. They fire on the
	// transitions in and out of idle, not per request, so overlapping requests produce one
	// span rather than nested ones. The end matters twice over: it is also the only moment
	// the keep-alive deadline moves, so a countdown is meaningless before it.
	// EventGenStart and EventGenEnd bracket one generation, and gen.end carries the
	// engine's own measurement of how it divided between prefill and decode.
	//
	// There is deliberately no gen.phase. A live transition could only be timestamped when
	// a poll of the engine observed it, which is a moment nobody measured -- and drawn
	// beside a memory trace that *is* measured, the two disagree in a way that reads as a
	// fault in the box. The engine reports the split once, at the end, so that is when it
	// is emitted; a consumer places the phases by working backwards from gen.end.
	EventGenStart = "gen.start"
	EventGenEnd   = "gen.end"

	// A job outside ollama asked for GPUs (lease.start), got them once ollama had vacated
	// them (lease.granted), and finished (lease.end). See server/lease.go.
	EventLeaseStart   = "lease.start"
	EventLeaseGranted = "lease.granted"
	EventLeaseEnd     = "lease.end"

	EventBusyStart = "busy.start"
	EventBusyEnd   = "busy.end"
)

// eventSubscriberBuffer is how many events a subscriber may fall behind before its events
// are dropped. A UI that stops reading -- a suspended tab, a closed laptop -- must never
// be able to block the scheduler, so a full buffer drops rather than waits.
const eventSubscriberBuffer = 64

type eventSubscriber struct {
	ch      chan api.ModelEvent
	dropped atomic.Uint64
}

type eventBus struct {
	mu     sync.Mutex
	subs   map[*eventSubscriber]struct{}
	nextID atomic.Uint64
}

func newEventBus() *eventBus {
	return &eventBus{subs: make(map[*eventSubscriber]struct{})}
}

// Subscribe returns a channel of events and a function that stops the subscription. The
// caller must call the returned function or the subscriber leaks.
func (b *eventBus) Subscribe() (<-chan api.ModelEvent, func()) {
	sub := &eventSubscriber{ch: make(chan api.ModelEvent, eventSubscriberBuffer)}

	b.mu.Lock()
	b.subs[sub] = struct{}{}
	b.mu.Unlock()

	return sub.ch, func() {
		b.mu.Lock()
		if _, ok := b.subs[sub]; ok {
			delete(b.subs, sub)
			close(sub.ch)
		}
		b.mu.Unlock()
	}
}

// Publish delivers an event to every subscriber without blocking on any of them.
//
// A subscriber that cannot keep up loses events rather than stalling the scheduler, and
// the count of what it lost travels with the next event it does receive, so a client can
// tell a quiet period from a gap in its own record.
func (b *eventBus) Publish(ev api.ModelEvent) {
	if ev.At.IsZero() {
		ev.At = time.Now().UTC()
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	for sub := range b.subs {
		ev.Dropped = sub.dropped.Load()
		select {
		case sub.ch <- ev:
		default:
			sub.dropped.Add(1)
		}
	}
}

// subscriberCount is used by tests to confirm unsubscribing releases the subscriber.
func (b *eventBus) subscriberCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.subs)
}

// acceptsGzip reports whether an Accept-Encoding header asks for gzip.
//
// Parsed rather than substring-matched because "gzip;q=0" means the client explicitly does
// NOT want it, and a naive strings.Contains reads that as consent. "*" counts unless it too
// is disqualified by q=0.
func acceptsGzip(header string) bool {
	wildcard := false
	for _, part := range strings.Split(header, ",") {
		token, params, _ := strings.Cut(strings.TrimSpace(part), ";")
		token = strings.ToLower(strings.TrimSpace(token))
		if token != "gzip" && token != "*" {
			continue
		}
		// q=0 is a refusal. Any other q, or none, is acceptance -- the relative ordering
		// of acceptable encodings does not matter when gzip is the only one offered.
		refused := false
		for _, p := range strings.Split(params, ";") {
			k, v, ok := strings.Cut(strings.TrimSpace(p), "=")
			if ok && strings.EqualFold(strings.TrimSpace(k), "q") {
				if q, err := strconv.ParseFloat(strings.TrimSpace(v), 64); err == nil && q == 0 {
					refused = true
				}
			}
		}
		if refused {
			if token == "gzip" {
				return false // an explicit refusal of gzip beats a permissive wildcard
			}
			continue
		}
		if token == "gzip" {
			return true
		}
		wildcard = true
	}
	return wildcard
}

// eventEncoder writes NDJSON frames to a streaming response, optionally gzipped.
//
// The reason this is a type rather than three lines in the handler is the flushing, which is
// the whole difficulty of compressing a stream. A gzip writer holds bytes back until it has a
// block worth emitting, so without an explicit flush after every frame a client sees nothing
// for seconds and then a burst -- a live stream silently becomes a batch feed, and it still
// passes any test that only checks the bytes eventually arrive and decode. gzip.Writer.Flush
// is a Z_SYNC_FLUSH: it ends the current block and pads to a byte boundary so everything
// written so far is decodable, without resetting the compression window.
//
// Keeping the window across frames is what makes this worth doing at all. The stream repeats
// the same model names, digests and device ids in frame after frame, and 13.3x of the 13.3x
// measured comes from matching those against earlier frames. Compressing each frame
// independently gets 1.9x.
type eventEncoder struct {
	enc  *json.Encoder
	gz   *gzip.Writer
	http http.Flusher
}

func newEventEncoder(w io.Writer, flusher http.Flusher, compress bool) *eventEncoder {
	e := &eventEncoder{http: flusher}
	if compress {
		e.gz = gzip.NewWriter(w)
		w = e.gz
	}
	e.enc = json.NewEncoder(w)
	return e
}

// Encode writes one frame and pushes it all the way out.
func (e *eventEncoder) Encode(v any) error {
	if err := e.enc.Encode(v); err != nil {
		return err
	}
	if e.gz != nil {
		if err := e.gz.Flush(); err != nil {
			return err
		}
	}
	if e.http != nil {
		e.http.Flush()
	}
	return nil
}

// Close finishes the gzip stream. Nothing to do for the uncompressed case.
func (e *eventEncoder) Close() error {
	if e.gz != nil {
		return e.gz.Close()
	}
	return nil
}
