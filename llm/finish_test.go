package llm

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
)

func finishTestRunner(t *testing.T, handler http.HandlerFunc) *llamaServerRunner {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/slots":
			fmt.Fprint(w, `[{"id":0,"n_ctx":32768,"is_processing":false}]`)
		default:
			w.Header().Set("Content-Type", "text/event-stream")
			handler(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	var port int
	fmt.Sscanf(srv.URL[strings.LastIndex(srv.URL, ":")+1:], "%d", &port)
	return &llamaServerRunner{
		port:    port,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 32768}},
		launch:  llamaServerLaunchConfig{numParallel: 1},
	}
}

// On the Completion path the tool calls are found by the handler's parser, and the last of
// them arrive with the final response. So gen.end must be announced after the handler has
// seen that response: announced before it, an agent turn read as a final answer, which is
// the one thing the finish shape exists to tell apart.
func TestCompletionFinishSeesToolCallsInTheFinalResponse(t *testing.T) {
	for _, tc := range []struct {
		name      string
		stopType  string
		toolCalls bool
		want      string
	}{
		{"tool calls", "eos", true, api.FinishToolCalls},
		{"answer", "eos", false, api.FinishAnswer},
		{"length", "limit", false, api.FinishLength},
		{"tool calls at the limit", "limit", true, api.FinishToolCalls},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runner := finishTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprintln(w, `data: {"content":"<tool_call>","stop":false}`)
				fmt.Fprintf(w, "data: {\"content\":\"</tool_call>\",\"stop\":true,\"stop_type\":%q,\"timings\":{\"prompt_n\":1,\"prompt_ms\":1,\"predicted_n\":2,\"predicted_ms\":2}}\n", tc.stopType)
			})
			var got []string
			runner.SetOnGenerationDone(func(_ api.GenerationTimings, m *api.GenerationMeta) { got = append(got, m.Finish()) })

			opts := api.DefaultOptions()
			meta := &api.GenerationMeta{}
			err := runner.Completion(t.Context(), CompletionRequest{Prompt: "p", Options: &opts, Meta: meta}, func(r CompletionResponse) {
				// What the chat handler does: its parser completes the tool call only once
				// the closing tag has arrived, which is in the final response.
				if tc.toolCalls && r.Done {
					meta.NoteToolCalls()
				}
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 1 || got[0] != tc.want {
				t.Fatalf("gen.end finish = %q, want [%q]", got, tc.want)
			}
		})
	}
}

// The engine's own chat template path accumulates tool calls itself, from the deltas.
func TestChatFinishFromAccumulatedToolCalls(t *testing.T) {
	for _, tc := range []struct {
		name   string
		delta  string
		reason string
		want   string
	}{
		{"tool calls", `{"tool_calls":[{"index":0,"id":"c1","function":{"name":"get_weather","arguments":"{}"}}]}`, "tool_calls", api.FinishToolCalls},
		{"answer", `{"content":"hi"}`, "stop", api.FinishAnswer},
		{"length", `{"content":"hi"}`, "length", api.FinishLength},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runner := finishTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":%s}]}\n\n", tc.delta)
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":%q}],\"timings\":{\"prompt_n\":1,\"prompt_ms\":1,\"predicted_n\":2,\"predicted_ms\":2}}\n\n", tc.reason)
				fmt.Fprintln(w, `data: [DONE]`)
			})
			var got []string
			runner.SetOnGenerationDone(func(_ api.GenerationTimings, m *api.GenerationMeta) { got = append(got, m.Finish()) })

			opts := api.DefaultOptions()
			err := runner.Chat(t.Context(), ChatRequest{Messages: []api.Message{{Role: "user", Content: "hi"}}, Options: &opts, Meta: &api.GenerationMeta{}},
				func(ChatResponse) {})
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 1 || got[0] != tc.want {
				t.Fatalf("gen.end finish = %q, want [%q]", got, tc.want)
			}
		})
	}
}
