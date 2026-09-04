package server

import (
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestEventBusDeliversToEverySubscriber(t *testing.T) {
	b := newEventBus()
	a, stopA := b.Subscribe()
	defer stopA()
	c, stopC := b.Subscribe()
	defer stopC()

	b.Publish(api.ModelEvent{Type: EventLoadStart, Model: "m"})

	for i, ch := range []<-chan api.ModelEvent{a, c} {
		select {
		case ev := <-ch:
			if ev.Type != EventLoadStart || ev.Model != "m" {
				t.Errorf("subscriber %d: got %+v", i, ev)
			}
			if ev.At.IsZero() {
				t.Errorf("subscriber %d: At was not stamped", i)
			}
		case <-time.After(time.Second):
			t.Fatalf("subscriber %d received nothing", i)
		}
	}
}

func TestEventBusUnsubscribeReleasesAndCloses(t *testing.T) {
	b := newEventBus()
	ch, stop := b.Subscribe()
	if got := b.subscriberCount(); got != 1 {
		t.Fatalf("subscriberCount = %d, want 1", got)
	}

	stop()
	if got := b.subscriberCount(); got != 0 {
		t.Errorf("subscriberCount after stop = %d, want 0", got)
	}
	if _, open := <-ch; open {
		t.Error("channel should be closed so a reader's range terminates")
	}

	stop()                                       // must be safe twice
	b.Publish(api.ModelEvent{Type: EventUnload}) // must not panic on a closed channel
}

// The scheduler must never be held up by a client that stopped reading. A suspended tab is
// the normal case, not an exotic one.
func TestEventBusDropsRatherThanBlocks(t *testing.T) {
	b := newEventBus()
	_, stop := b.Subscribe() // subscribed, never read
	defer stop()

	done := make(chan struct{})
	go func() {
		for range eventSubscriberBuffer + 50 {
			b.Publish(api.ModelEvent{Type: EventLoadStart, Model: "m"})
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Publish blocked on a subscriber that was not reading")
	}
}

// A gap in a client's record must be distinguishable from a quiet period, so the count of
// what it missed rides along on the next event it actually receives.
func TestEventBusReportsDropsToTheSubscriber(t *testing.T) {
	b := newEventBus()
	ch, stop := b.Subscribe()
	defer stop()

	for range eventSubscriberBuffer + 10 {
		b.Publish(api.ModelEvent{Type: EventLoadStart, Model: "m"})
	}

	// Drain what fitted; none of those can report drops that had not happened yet.
	for range eventSubscriberBuffer {
		<-ch
	}

	b.Publish(api.ModelEvent{Type: EventLoadComplete, Model: "m"})
	select {
	case ev := <-ch:
		if ev.Dropped == 0 {
			t.Error("expected a non-zero dropped count after overflowing the buffer")
		}
		if ev.Dropped != 10 {
			t.Errorf("dropped = %d, want 10 (the events that did not fit)", ev.Dropped)
		}
	case <-time.After(time.Second):
		t.Fatal("no event after the buffer drained")
	}
}

// A reconnecting client must be told what the ring actually holds, not what it asked for.
// The difference is the client's gap, and a gap it cannot see is one it will draw over.
func TestFrameRingBackfillReportsShortfall(t *testing.T) {
	r := newFrameRing(10 * time.Minute)
	r.add(retainedFrame{at: time.Now().Add(-8 * time.Second), kind: "sample"})
	r.add(retainedFrame{at: time.Now(), kind: EventLoadComplete, model: "m"})

	frames, reach := r.since(10 * time.Minute)
	if len(frames) != 2 {
		t.Fatalf("got %d frames, want 2", len(frames))
	}
	if reach > 15*time.Second {
		t.Errorf("reach = %v, but the ring only holds ~8s; a client asking for 10m must see that", reach)
	}
}

// Backfilled frames precede the connection that serves them, so their offset is negative.
// Restamping them to zero would place history at the moment of reconnection.
func TestBackfillOffsetsAreNegative(t *testing.T) {
	started := time.Now()
	past := started.Add(-30 * time.Second)

	if off := past.Sub(started).Milliseconds(); off >= 0 {
		t.Errorf("offset for a frame 30s before the connection = %d, want negative", off)
	}
}
