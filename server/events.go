package server

import (
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
	EventEvict       = "evict"
	EventUnload      = "unload"

	// EventExpires reports a keep-alive deadline moving. It is emitted when the deadline
	// is written, which happens only as a request finishes -- so a client that draws a
	// countdown has no way to know from /api/ps alone that the number it is counting from
	// is being held rather than approaching.
	EventExpires = "expires"

	// EventBusyStart and EventBusyEnd bracket a model actually working. They fire on the
	// transitions in and out of idle, not per request, so overlapping requests produce one
	// span rather than nested ones. The end matters twice over: it is also the only moment
	// the keep-alive deadline moves, so a countdown is meaningless before it.
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
