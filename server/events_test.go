package server

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
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
	r.add(retainedFrame{at: time.Now().Add(-8 * time.Second), event: api.ModelEvent{Type: "sample"}})
	r.add(retainedFrame{at: time.Now(), event: api.ModelEvent{Type: EventLoadComplete, Model: "m"}})

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

// TestEventFrameCopiesEveryCommonField walks the two structs by JSON name and fails on any
// field the conversion forgot. The conversion is written by hand and has dropped a field
// three times -- info, expires_at, then duration_ms -- each time shipping a server that
// reported something no client could see. A name present on both types is a field the
// wire is meant to carry, so it must survive.
func TestEventFrameCopiesEveryCommonField(t *testing.T) {
	// Values are deliberately non-zero: a field left at its zero value is exactly what a
	// forgotten copy looks like, so it must be distinguishable from one that was copied.
	at := time.Now().UTC()
	expires := at.Add(time.Hour)
	ev := api.ModelEvent{
		Type:       EventLoadComplete,
		Model:      "some-model:latest",
		Reason:     "because",
		At:         at,
		DurationMs: 8000,
		WeightsMs:  2000,
		ContextMs:  6000,
		SizeVRAM:   86 << 30,
		SizeTotal:  88 << 30,
		Memory: &api.MemoryBreakdown{
			Weights: 70 << 30, KVCache: 14 << 30, Compute: 2 << 30, Output: 1 << 20,
		},
		MemoryHost:    &api.MemoryBreakdown{Weights: 2 << 30},
		WeightsOnDisk: 69 << 30,
		Timings: &api.GenerationTimings{
			PromptTokens: 4098, PromptTokensCached: cachedTokens(4097),
			PromptMs: 3.4, EvalMs: 111.5, Decoded: 40,
		},
		Hint: &api.RequestHint{Use: "agent", Session: "chat-123"},
		Placement: &api.ModelPlacement{
			NumLayers: 4,
			Devices: []api.PlacementRange{
				{Device: "CUDA0", FirstLayer: 0, LastLayer: 1, Layers: 2},
				{Device: "CUDA1", FirstLayer: 2, LastLayer: 3, Layers: 2},
			},
			SWALayers: []int{1, 3},
		},
		Estimate:  &api.LoadEstimate{Predicted: 86 << 30, Source: "probe"},
		Dropped:   3,
		ExpiresAt: &expires,
		PS:        &api.ProcessResponse{},
		Info:      &api.InfoResponse{},
	}

	frame := frameFromEvent(ev, at)

	jsonNames := func(v any) map[string]reflect.Value {
		out := map[string]reflect.Value{}
		rv := reflect.ValueOf(v)
		rt := rv.Type()
		for i := range rt.NumField() {
			name, _, _ := strings.Cut(rt.Field(i).Tag.Get("json"), ",")
			if name != "" && name != "-" {
				out[name] = rv.Field(i)
			}
		}
		return out
	}

	// "type" on the event is "kind" on the frame; both are set from the same value and
	// the assertion below covers it via Kind. Everything else pairs by name.
	renamed := map[string]string{"type": "kind"}
	frameFields := jsonNames(frame)

	// A field the frame does not have is the failure this test missed once already:
	// SizeVRAM was set on every load.complete and reached no client, because the frame had
	// no such field and this loop read that as "not meant for the wire". Anything genuinely
	// internal has to be named here, so adding one is a decision rather than an oversight.
	eventOnly := map[string]bool{
		"at":   true, // the wire carries a per-connection offset, t, instead
		"gpus": true, // placement is served through the ps body
	}

	for name, evField := range jsonNames(ev) {
		if r, ok := renamed[name]; ok {
			name = r
		}
		frameField, ok := frameFields[name]
		if !ok {
			if !eventOnly[name] {
				t.Errorf("%s exists on the event but not on the frame, so it can never reach a client; add it to the frame, or to eventOnly if that is deliberate", name)
			}
			continue
		}
		if evField.IsZero() {
			t.Fatalf("%s: the fixture left this zero, so the test cannot detect a dropped copy", name)
		}
		if frameField.IsZero() {
			t.Errorf("%s is set on the event but zero on the frame: the conversion dropped it", name)
		}
	}

	if frame.Kind != ev.Type {
		t.Errorf("kind = %q, want %q", frame.Kind, ev.Type)
	}
	if frame.T != 0 {
		t.Errorf("t = %d, want 0 for an event at the stream's start time", frame.T)
	}
}

func TestAcceptsGzip(t *testing.T) {
	tests := []struct {
		header string
		want   bool
	}{
		{"", false},
		{"gzip", true},
		{"gzip, deflate, br", true},
		{"deflate, br", false},
		{"identity", false},
		{"*", true},
		{"gzip;q=1.0", true},
		{"gzip;q=0.5, *;q=0.1", true},
		// An explicit refusal. A substring match reads this as consent and sends a body the
		// client has said it cannot read.
		{"gzip;q=0", false},
		{"gzip;q=0, deflate", false},
		// A refusal of gzip beats a permissive wildcard, whichever order they appear in.
		{"*, gzip;q=0", false},
		{"gzip;q=0, *", false},
		{"*;q=0", false},
		{"GZIP", true},
		{"  gzip  ;  q=0.9  ", true},
	}
	for _, tt := range tests {
		t.Run(tt.header, func(t *testing.T) {
			if got := acceptsGzip(tt.header); got != tt.want {
				t.Errorf("acceptsGzip(%q) = %v, want %v", tt.header, got, tt.want)
			}
		})
	}
}

// TestEventEncoderFlushesEachFrame is the test this whole type exists for.
//
// A gzip writer holds bytes until it has a block worth emitting. Without a flush after every
// frame the client receives nothing for seconds and then a burst, which turns a live stream
// into a batch feed -- and every naive test still passes, because the bytes do all arrive and
// do all decode, just not when they were written. So the assertion has to be that frame N is
// readable BEFORE frame N+1 is written.
func TestEventEncoderFlushesEachFrame(t *testing.T) {
	for _, compress := range []bool{false, true} {
		name := "plain"
		if compress {
			name = "gzip"
		}
		t.Run(name, func(t *testing.T) {
			var buf bytes.Buffer
			enc := newEventEncoder(&buf, nil, compress)

			for i := 1; i <= 3; i++ {
				if err := enc.Encode(api.EventFrame{V: 1, Kind: "sample", Model: fmt.Sprint("m", i)}); err != nil {
					t.Fatalf("encode %d: %v", i, err)
				}
				// Read what a client would have received by now, without closing the
				// stream -- closing is what a batch encoder would need to be readable.
				got := decodeFrames(t, buf.Bytes(), compress)
				if len(got) != i {
					t.Fatalf("after writing %d frames the client can read %d; the encoder is "+
						"buffering, so frames arrive late and in bursts", i, len(got))
				}
				if got[i-1].Model != fmt.Sprint("m", i) {
					t.Errorf("frame %d = %q, want %q", i, got[i-1].Model, fmt.Sprint("m", i))
				}
			}
			if err := enc.Close(); err != nil {
				t.Fatalf("close: %v", err)
			}
		})
	}
}

// TestEventEncoderRoundTrips checks the compressed stream carries exactly what the plain one
// does -- same frames, same order, same fields.
func TestEventEncoderRoundTrips(t *testing.T) {
	frames := []api.EventFrame{
		{V: 1, Kind: "hello", Box: "abc"},
		{V: 1, Kind: "load.complete", Model: "granite4.1:3b", SizeVRAM: 2915943054,
			Memory: &api.MemoryBreakdown{Weights: 2095935651, KVCache: 671088640}},
		{V: 1, Kind: "sample", Model: "granite4.1:3b"},
	}

	encode := func(compress bool) []byte {
		var buf bytes.Buffer
		enc := newEventEncoder(&buf, nil, compress)
		for _, f := range frames {
			if err := enc.Encode(f); err != nil {
				t.Fatal(err)
			}
		}
		if err := enc.Close(); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}

	plain, gzipped := encode(false), encode(true)
	if !reflect.DeepEqual(decodeFrames(t, plain, false), decodeFrames(t, gzipped, true)) {
		t.Error("the gzipped stream does not decode to the same frames as the plain one")
	}

	// Not an assertion about a particular ratio -- only that the window is shared across
	// frames. Compressing each frame alone would not beat the plain encoding on a payload
	// this small, because every frame would carry its own 18-byte gzip header.
	if len(gzipped) >= len(plain) {
		t.Errorf("gzipped %d bytes is not smaller than plain %d; the window is not being reused",
			len(gzipped), len(plain))
	}
}

// decodeFrames reads however many complete frames are present, without requiring the stream
// to be finished. io.ErrUnexpectedEOF is the normal case mid-stream and not a failure.
func decodeFrames(t *testing.T, b []byte, compressed bool) []api.EventFrame {
	t.Helper()
	var r io.Reader = bytes.NewReader(b)
	if compressed {
		zr, err := gzip.NewReader(r)
		if err != nil {
			if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
				return nil
			}
			t.Fatalf("gzip reader: %v", err)
		}
		zr.Multistream(false)
		r = zr
	}
	var out []api.EventFrame
	dec := json.NewDecoder(r)
	for {
		var f api.EventFrame
		err := dec.Decode(&f)
		if err != nil {
			return out
		}
		out = append(out, f)
	}
}

// TestBackfilledEdgeCarriesTheSamePayloadAsLive is the regression for a real bug: an edge
// replayed from the ring arrived carrying only its model name.
//
// The ring stored a hand-picked six fields of api.ModelEvent and the replay loop rebuilt a
// frame from those, so every field that makes an edge worth having was dropped on the way
// through — weights_ms, context_ms, size_vram, memory, placement, timings, estimate,
// expires_at. Nothing broke: a client reads each conditionally and absence already means
// "not reported", so it drew an empty span instead of a wrong one. That is the failure this
// endpoint exists to prevent, arriving quietly.
//
// It matters because backfill is the ORDINARY path for a browser client — an evicted
// service worker takes the connection with it, so a reconnect asking for the whole ring is
// routine.
func TestBackfilledEdgeCarriesTheSamePayloadAsLive(t *testing.T) {
	at := time.Now().UTC()
	expires := at.Add(time.Hour)
	ev := api.ModelEvent{
		Type: EventLoadComplete, Model: "granite4.1:3b", At: at,
		DurationMs: 517, WeightsMs: 327, ContextMs: 190,
		SizeVRAM: 2915943054, SizeTotal: 2915943054, WeightsOnDisk: 2099501664,
		Memory:    &api.MemoryBreakdown{Weights: 2095935651, KVCache: 671088640},
		Placement: &api.ModelPlacement{NumLayers: 41},
		Timings:   &api.GenerationTimings{PromptTokens: 4098, PromptMs: 3.5, EvalMs: 170.6},
		Estimate:  &api.LoadEstimate{Predicted: 2915943054, Source: "calibration"},
		ExpiresAt: &expires,
		Reason:    "because",
	}

	// Live: what a connected subscriber receives.
	live := frameFromEvent(ev, at)

	// Backfilled: through the REAL publish path, not by building a retainedFrame here.
	//
	// A first version of this test constructed the retained frame directly and passed
	// against the bug, because the stripping happened in publishEvent -- the one line the
	// test skipped. Going through the scheduler is what makes it a regression test rather
	// than an assertion that the ring returns what it was handed.
	sched := &Scheduler{
		ring:            newFrameRing(10 * time.Minute),
		events:          newEventBus(),
		getGpuFn:        getGpuFn,
		getSystemInfoFn: getSystemInfoFn,
	}
	sched.publishEvent(ev)
	got, _ := sched.ring.since(10 * time.Minute)
	if len(got) != 1 {
		t.Fatalf("ring returned %d frames, want 1", len(got))
	}
	replayed := frameFromEvent(got[0].event, at)

	if !reflect.DeepEqual(live, replayed) {
		t.Errorf("a backfilled edge differs from the live one.\nlive:     %+v\nreplayed: %+v",
			live, replayed)
	}
	// Named explicitly as well, so a future change that drops one of these fails with the
	// field name rather than with a struct diff nobody reads.
	for name, ok := range map[string]bool{
		"weights_ms": replayed.WeightsMs == 327,
		"context_ms": replayed.ContextMs == 190,
		"size_vram":  replayed.SizeVRAM == 2915943054,
		"memory":     replayed.Memory != nil,
		"placement":  replayed.Placement != nil,
		"timings":    replayed.Timings != nil,
		"estimate":   replayed.Estimate != nil,
		"expires_at": replayed.ExpiresAt != nil,
	} {
		if !ok {
			t.Errorf("%s was lost on the way through the ring", name)
		}
	}
}

func cachedTokens(n int) *int { return &n }

// TestCachedPromptTokensDistinguishesColdFromUnreported is the regression for an ambiguity
// this field carried from the day it shipped until upstream's #17943 forced it into view.
//
// It was an int with omitempty, so a cold prefill -- nothing served from the prefix cache,
// a genuine zero -- was OMITTED, identical on the wire to an engine that does not report the
// figure at all. Every uncached request looked like a missing measurement. That is the exact
// failure the consumers of this stream asked us to design out: absence is load-bearing here,
// and it was being spent on the commonest case.
//
// llama-server always reports the count, so on this box the pointer is never nil in practice;
// MLX does not, which is why upstream made it optional and why the two states must differ.
func TestCachedPromptTokensDistinguishesColdFromUnreported(t *testing.T) {
	encode := func(ts api.GenerationTimings) string {
		b, err := json.Marshal(ts)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}

	cold := encode(api.GenerationTimings{PromptTokens: 4098, PromptTokensCached: cachedTokens(0)})
	if !strings.Contains(cold, `"prompt_tokens_cached":0`) {
		t.Errorf("a cold prefill must report 0, not vanish: %s", cold)
	}

	unreported := encode(api.GenerationTimings{PromptTokens: 4098})
	if strings.Contains(unreported, "prompt_tokens_cached") {
		t.Errorf("an engine that does not report the figure must omit it: %s", unreported)
	}

	if cold == unreported {
		t.Error("a cold prefill and an unreported figure encode identically; the field cannot " +
			"tell a consumer which one it is looking at")
	}
}
