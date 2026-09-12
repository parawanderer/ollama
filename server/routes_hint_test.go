package server

import (
	"context"
	"net/http"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
)

// The handler must hand the caller's hint to the runner, trimmed, or gen.end never carries it.
func TestChatHandlerPassesTheHintToTheRunner(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var got *api.RequestHint
	mock := mockRunner{
		CompletionFn: func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			got = r.Hint
			fn(llm.CompletionResponse{Content: "ok", Done: true, DoneReason: llm.DoneReasonStop})
			return nil
		},
	}
	s := newServerWithMockRunner(t, &mock)
	createParserModel(t, s, "hinted", "qwen3.5")

	stream := false
	w := createRequest(t, s.ChatHandler, api.ChatRequest{
		Model:    "hinted",
		Messages: []api.Message{{Role: "user", Content: "hello"}},
		Stream:   &stream,
		Hint:     &api.RequestHint{Use: " agent ", Session: "run-1"},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	if got == nil || got.Use != "agent" || got.Session != "run-1" {
		t.Fatalf("runner got hint %+v, want the request's, trimmed", got)
	}
}

// The engine's own chat-template path is a different call site, and the one that used to emit
// no gen.end at all.
func TestNativeChatPassesTheHintToTheRunner(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	t.Setenv("OLLAMA_GO_TEMPLATE", "")
	gin.SetMode(gin.TestMode)
	mock := mockRunner{
		ChatFn: func(_ context.Context, req llm.ChatRequest, fn func(llm.ChatResponse)) error {
			fn(llm.ChatResponse{Message: api.Message{Role: "assistant", Content: "ok"}, Done: true, DoneReason: llm.DoneReasonStop})
			return nil
		},
	}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "native-hinted", ggml.KV{"tokenizer.chat_template": "{{ messages[0]['content'] }}"}, "", nil)

	stream := false
	w := createRequest(t, s.ChatHandler, api.ChatRequest{
		Model: "native-hinted", Messages: []api.Message{{Role: "user", Content: "hello"}}, Stream: &stream,
		Hint: &api.RequestHint{Use: "utility", Session: "chat-9"},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	if h := mock.ChatRequest.Hint; h == nil || h.Use != "utility" || h.Session != "chat-9" {
		t.Fatalf("runner got hint %+v", h)
	}
}

func TestGenerateHandlerPassesTheHintToTheRunner(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var got *api.RequestHint
	mock := mockRunner{
		CompletionFn: func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			got = r.Hint
			fn(llm.CompletionResponse{Content: "ok", Done: true, DoneReason: llm.DoneReasonStop})
			return nil
		},
	}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "gen-hinted", ggml.KV{}, "{{ .Prompt }}", nil)

	stream := false
	w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
		Model: "gen-hinted", Prompt: "hello", Stream: &stream, Hint: &api.RequestHint{Use: "batch"},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	if got == nil || got.Use != "batch" {
		t.Fatalf("runner got hint %+v", got)
	}
}
