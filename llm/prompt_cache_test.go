package llm

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
)

func mibBytes(v float64) uint64 { return uint64(v * 1024 * 1024) }

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// The fixtures are transcribed from a live runner (qwen3:32b, two conversations taking
// turns on one slot), not written from memory: one turn where the returning conversation
// was restored, and one where making room for the save evicted it.
func TestPromptCacheRestoredSwap(t *testing.T) {
	var tr promptCacheTracker
	tr.begin()
	tr.observe(readFixture(t, "prompt_cache_restore.log"))

	got := tr.take()
	want := &api.PromptCacheSwap{Ms: 620.61, SavedTokens: 6523, SavedBytes: mibBytes(1630.826), Restored: true}
	if got == nil || *got != *want {
		t.Fatalf("swap = %+v, want %+v", got, want)
	}

	state := tr.snapshot()
	wantState := api.PromptCacheState{Entries: 2, Tokens: 25 + 6523, Bytes: mibBytes(1637.078), LimitBytes: mibBytes(8192)}
	if state == nil || *state != wantState {
		t.Fatalf("state = %+v, want %+v", state, wantState)
	}
}

func TestPromptCacheThrashedSwap(t *testing.T) {
	var tr promptCacheTracker
	tr.begin()
	tr.observe(readFixture(t, "prompt_cache_thrash.log"))

	got := tr.take()
	want := &api.PromptCacheSwap{
		Ms: 1299.92, SavedTokens: 21763, SavedBytes: mibBytes(5441.001),
		Evicted: 3, EvictedBytes: mibBytes(1638.076) + mibBytes(6.252) + mibBytes(5434.5),
	}
	if got == nil || *got != *want {
		t.Fatalf("swap = %+v, want %+v", got, want)
	}
	if state := tr.snapshot(); state == nil || state.Entries != 1 || state.Tokens != 21763 {
		t.Fatalf("state = %+v, want the one entry just saved", state)
	}
}

// Transcribed from the first request a freshly loaded runner served: the engine runs an
// update into the empty slot that saves, evicts and restores nothing.
func TestPromptCacheEmptySlotIsNoSwap(t *testing.T) {
	var tr promptCacheTracker
	tr.begin()
	tr.observe(readFixture(t, "prompt_cache_empty_slot.log"))
	if got := tr.take(); got != nil {
		t.Fatalf("an update that moved nothing was reported as a swap: %+v", got)
	}
	if state := tr.snapshot(); state == nil || state.Entries != 0 || state.LimitBytes != mibBytes(8192) {
		t.Fatalf("state = %+v, want the empty cache and its limit", state)
	}
}

// The runner's stderr arrives in whatever chunks the pipe delivers. Order within the stream
// is what a swap is made of, so line-by-line delivery must read the same as one chunk.
func TestPromptCacheChunkingDoesNotMatter(t *testing.T) {
	fixture := readFixture(t, "prompt_cache_restore.log")
	var whole, lines promptCacheTracker
	whole.begin()
	whole.observe(fixture)
	lines.begin()
	for l := range strings.SplitSeq(string(fixture), "\n") {
		lines.observe([]byte(l + "\n"))
	}
	if a, b := whole.take(), lines.take(); a == nil || b == nil || *a != *b {
		t.Fatalf("whole chunk %+v, line by line %+v", a, b)
	}
}

// Lines no real run here produced, built from the format strings in llama.cpp's
// tools/server/server-task.cpp rather than from memory. Each arrives in a chunk of its own,
// which is what exercises the cheap pre-filter: the limit evictions never say "prompt".
func TestPromptCacheLimitEvictionsAndOversizedState(t *testing.T) {
	var tr promptCacheTracker
	tr.begin()
	for _, l := range []string{
		"srv  get_availabl: updating prompt cache\n",
		"srv   prompt_save:  - saving prompt with length 90000, total state size = 22500.000 MiB (draft: 0.000 MiB)\n",
		"srv         alloc:  - prompt state size 22500.000 MiB exceeds cache size limit 8192.000 MiB, skipping\n",
		"srv        update:  - cache size limit reached, removing oldest entry (size = 100.000 MiB)\n",
		"srv        update:  - cache token limit (32768, est: 32768) reached, removing oldest entry (size = 50.000 MiB)\n",
		"srv  get_availabl: prompt cache update took 3.00 ms\n",
	} {
		tr.observe([]byte(l))
	}
	got := tr.take()
	want := &api.PromptCacheSwap{Ms: 3, TooLarge: true, Evicted: 2, EvictedBytes: mibBytes(100) + mibBytes(50)}
	if got == nil || *got != *want {
		t.Fatalf("swap = %+v, want %+v", got, want)
	}
}

// A swap logged after its request completed must not be reported on the next request.
// begin discards it; the late request loses its swap, which is the right way round.
func TestPromptCacheLateSwapIsDroppedNotMisattributed(t *testing.T) {
	var tr promptCacheTracker
	tr.begin()
	if got := tr.take(); got != nil {
		t.Fatalf("a request that swapped nothing got %+v", got)
	}
	tr.observe(readFixture(t, "prompt_cache_restore.log")) // arrives after take
	tr.begin()                                             // the next request
	if got := tr.take(); got != nil {
		t.Fatalf("the next request inherited a swap it did not cause: %+v", got)
	}
	if tr.snapshot() == nil {
		t.Fatal("the occupancy it reported is still true and must be kept")
	}
}

func TestGenerationTimingsOmitsAbsentSwap(t *testing.T) {
	b, _ := json.Marshal(api.GenerationTimings{PromptTokens: 5})
	if strings.Contains(string(b), "prompt_cache_swap") {
		t.Fatalf("a request that swapped nothing encoded one: %s", b)
	}
	// Restored false is information -- the thrash case -- so it must survive encoding.
	b, _ = json.Marshal(api.PromptCacheSwap{Ms: 1})
	if !strings.Contains(string(b), `"restored":false`) {
		t.Fatalf("restored=false was dropped: %s", b)
	}
}

// End to end through Completion: the engine logs the swap while choosing a slot, before it
// answers, and the swap must reach the generation callback of that request and not the next.
func TestCompletionReportsThePromptCacheSwapItCaused(t *testing.T) {
	var runner *llamaServerRunner
	swapNext := true
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/slots":
			fmt.Fprint(w, `[{"id":0,"n_ctx":32768,"is_processing":false}]`)
		case "/completion":
			if swapNext {
				log := &memoryParsingWriter{inner: io.Discard, runner: runner}
				log.Write(readFixture(t, "prompt_cache_restore.log"))
			}
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"content":"","stop":true,"stop_type":"eos","timings":{"cache_n":6503,"prompt_n":21,"prompt_ms":24,"predicted_n":8,"predicted_ms":120}}`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	var port int
	fmt.Sscanf(srv.URL[strings.LastIndex(srv.URL, ":")+1:], "%d", &port)
	runner = &llamaServerRunner{
		port:    port,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 32768}},
		launch:  llamaServerLaunchConfig{numParallel: 1},
	}
	var got []api.GenerationTimings
	runner.SetOnGenerationDone(func(t api.GenerationTimings, _ *api.GenerationMeta) { got = append(got, t) })

	run := func() {
		opts := api.DefaultOptions()
		if err := runner.Completion(t.Context(), CompletionRequest{Prompt: "p", Options: &opts}, func(CompletionResponse) {}); err != nil {
			t.Fatal(err)
		}
	}
	run()
	// A swap logged late, after the first request completed and before the second began.
	// It belongs to neither and must not be pinned on the second.
	late := &memoryParsingWriter{inner: io.Discard, runner: runner}
	late.Write(readFixture(t, "prompt_cache_thrash.log"))
	swapNext = false
	run()

	if len(got) != 2 {
		t.Fatalf("got %d generations, want 2", len(got))
	}
	if s := got[0].PromptCacheSwap; s == nil || !s.Restored || s.Ms != 620.61 {
		t.Errorf("first request: swap = %+v, want the restore it caused", s)
	}
	if s := got[1].PromptCacheSwap; s != nil {
		t.Errorf("second request continued its conversation and swapped nothing, got %+v", s)
	}

	// Read the way /api/ps reads it, twice: the second answer comes from the /slots cache,
	// and the occupancy must be joined on that path too. The late thrash was the last
	// report, so one entry.
	for _, path := range []string{"fresh poll", "cached poll"} {
		activity := runner.Activity(t.Context(), false)
		if activity == nil || activity.PromptCache == nil || activity.PromptCache.Entries != 1 || activity.PromptCache.Tokens != 21763 {
			t.Errorf("%s: activity = %+v, want the occupancy the engine last reported", path, activity)
		}
	}
}

// With more than one slot the log cannot say whose swap is whose, so none is reported.
func TestCompletionReportsNoSwapWithParallelSlots(t *testing.T) {
	var runner *llamaServerRunner
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/completion":
			log := &memoryParsingWriter{inner: io.Discard, runner: runner}
			log.Write(readFixture(t, "prompt_cache_restore.log"))
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"content":"","stop":true,"stop_type":"eos","timings":{"cache_n":1,"prompt_n":1,"prompt_ms":1,"predicted_n":1,"predicted_ms":1}}`)
		}
	}))
	defer srv.Close()
	var port int
	fmt.Sscanf(srv.URL[strings.LastIndex(srv.URL, ":")+1:], "%d", &port)
	runner = &llamaServerRunner{
		port: port, cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(2),
		options: api.Options{Runner: api.Runner{NumCtx: 32768}},
		launch:  llamaServerLaunchConfig{numParallel: 2},
	}
	var got *api.PromptCacheSwap
	runner.SetOnGenerationDone(func(t api.GenerationTimings, _ *api.GenerationMeta) { got = t.PromptCacheSwap })
	opts := api.DefaultOptions()
	if err := runner.Completion(t.Context(), CompletionRequest{Prompt: "p", Options: &opts}, func(CompletionResponse) {}); err != nil {
		t.Fatal(err)
	}
	if got != nil {
		t.Fatalf("two slots, and a swap was attributed anyway: %+v", got)
	}
}

func TestAppendCacheRAMArgs(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  []string
	}{
		{"", nil},
		{"32768", []string{"--cache-ram", "32768"}},
		{"-1", []string{"--cache-ram", "-1"}},
		{"0", []string{"--cache-ram", "0"}},
		{"32GiB", nil}, // llama-server would refuse to start; drop it instead
		{"-2", nil},
	} {
		got := appendCacheRAMArgs(nil, tc.value)
		if strings.Join(got, " ") != strings.Join(tc.want, " ") {
			t.Errorf("OLLAMA_CACHE_RAM=%q: got %v, want %v", tc.value, got, tc.want)
		}
	}
}

func TestWithPromptCacheDoesNotMutateTheSharedReading(t *testing.T) {
	cached := &api.RunnerActivity{Phase: "idle"}
	out := withPromptCache(cached, &api.PromptCacheState{Entries: 1})
	if cached.PromptCache != nil || out.PromptCache == nil {
		t.Fatalf("cached = %+v, out = %+v", cached, out)
	}
	if withPromptCache(nil, &api.PromptCacheState{}) != nil {
		t.Fatal("no /slots reading must stay no reading")
	}
}

func TestLaunchPassesCacheRAMOnlyWhenSet(t *testing.T) {
	launch := llamaServerLaunchConfig{modelPath: "m", opts: api.DefaultOptions(), numParallel: 1}
	t.Setenv("OLLAMA_CACHE_RAM", "")
	if p := strings.Join(llamaServerParams(launch, 0), " "); strings.Contains(p, "--cache-ram") {
		t.Fatalf("unset must leave llama-server's default alone: %s", p)
	}
	t.Setenv("OLLAMA_CACHE_RAM", "32768")
	if p := strings.Join(llamaServerParams(launch, 0), " "); !strings.Contains(p, "--cache-ram 32768") {
		t.Fatalf("OLLAMA_CACHE_RAM=32768 did not reach the launch: %s", p)
	}
}

// The engine's own chat template path (Chat, not Completion) used to finish without firing the
// generation callback, so models served that way produced no gen.end at all. It must fire, with
// the prompt-cache swap it caused and the caller's hint.
func TestChatReportsGenerationDoneWithItsHint(t *testing.T) {
	var runner *llamaServerRunner
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/slots":
			fmt.Fprint(w, `[{"id":0,"n_ctx":32768,"is_processing":false}]`)
		case "/v1/chat/completions":
			log := &memoryParsingWriter{inner: io.Discard, runner: runner}
			log.Write(readFixture(t, "prompt_cache_restore.log"))
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"hi"}}]}`)
			fmt.Fprintln(w)
			fmt.Fprintln(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"cache_n":6503,"prompt_n":21,"prompt_ms":24,"predicted_n":8,"predicted_ms":120}}`)
			fmt.Fprintln(w)
			fmt.Fprintln(w, `data: [DONE]`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	var port int
	fmt.Sscanf(srv.URL[strings.LastIndex(srv.URL, ":")+1:], "%d", &port)
	runner = &llamaServerRunner{
		port:    port,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 32768}},
		launch:  llamaServerLaunchConfig{numParallel: 1},
	}
	var got []api.GenerationTimings
	var metas []*api.GenerationMeta
	runner.SetOnGenerationDone(func(t api.GenerationTimings, m *api.GenerationMeta) {
		got = append(got, t)
		metas = append(metas, m)
	})

	opts := api.DefaultOptions()
	meta := &api.GenerationMeta{Hint: &api.RequestHint{Use: "agent", Session: "run-7"}}
	if err := runner.Chat(t.Context(), ChatRequest{Messages: []api.Message{{Role: "user", Content: "hi"}}, Options: &opts, Meta: meta},
		func(ChatResponse) {}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("generation callback fired %d times, want 1", len(got))
	}
	if got[0].Decoded != 8 || got[0].PromptMs != 24 {
		t.Errorf("timings not carried: %+v", got[0])
	}
	if got[0].PromptCacheSwap == nil {
		t.Error("the swap this request caused was not attributed to it")
	}
	if metas[0] != meta {
		t.Errorf("meta = %+v, want the request's", metas[0])
	}
}
