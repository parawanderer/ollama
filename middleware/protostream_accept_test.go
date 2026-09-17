package middleware

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

// The cases the negotiation has to get right, including the two a substring check gets
// wrong: an explicit refusal, and a client that ranks SSE above protobuf.
func TestAcceptsProtoStream(t *testing.T) {
	cases := []struct {
		name   string
		accept []string
		want   bool
	}{
		{"absent", nil, false},
		{"empty", []string{""}, false},
		{"anything", []string{"*/*"}, false},
		{"subtype wildcard", []string{"application/*"}, false},
		{"sse only", []string{"text/event-stream"}, false},
		{"protobuf alone", []string{"application/protobuf"}, true},
		{"what window.ml sends", []string{"application/protobuf, text/event-stream;q=0.9"}, true},
		{"protobuf with its framing parameter", []string{"application/protobuf; delimited=varint"}, true},
		{"explicit refusal", []string{"application/protobuf;q=0"}, false},
		{"refusal among others", []string{"text/event-stream, application/protobuf;q=0"}, false},
		{"sse preferred", []string{"text/event-stream, application/protobuf;q=0.1"}, false},
		{"sse preferred, reversed order", []string{"application/protobuf;q=0.1, text/event-stream"}, false},
		{"protobuf preferred", []string{"text/event-stream;q=0.2, application/protobuf;q=0.8"}, true},
		{"a tie goes to protobuf", []string{"application/protobuf;q=0.5, text/event-stream;q=0.5"}, true},
		{"protobuf named alongside a wildcard", []string{"application/protobuf, */*"}, true},
		{"case and spacing", []string{"  APPLICATION/Protobuf ;  Q=0.9 , text/event-stream;q=0.1"}, true},
		{"two header lines", []string{"text/event-stream;q=0.9", "application/protobuf"}, true},
		{"sse by wildcard outranks a downweighted protobuf", []string{"*/*, application/protobuf;q=0.5"}, false},
		{"text wildcard is more specific than */*", []string{"*/*;q=0.1, text/*;q=0.9, application/protobuf;q=0.5"}, false},
		{"malformed q on protobuf drops it", []string{"application/protobuf;q=high"}, false},
		{"malformed q on sse drops it", []string{"text/event-stream;q=high, application/protobuf;q=0.4"}, true},
		{"q out of range drops it", []string{"application/protobuf;q=2"}, false},
		{"a substring is not a match", []string{"application/protobuf-mangled"}, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := AcceptsProtoStream(tc.accept); got != tc.want {
				t.Errorf("AcceptsProtoStream(%q) = %v, want %v", tc.accept, got, tc.want)
			}
		})
	}
}

// End to end through the middleware, because what the negotiation decides is the response's
// Content-Type, and a unit test of the parser alone would not notice the wiring being lost.
func TestChatMiddlewareNegotiatesProtoStream(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cases := []struct {
		name   string
		accept string
		stream bool
		want   string
	}{
		{"no Accept stays on SSE", "", true, "text/event-stream"},
		{"protobuf preferred", "application/protobuf, text/event-stream;q=0.9", true, protoStreamContentType},
		{"sse preferred", "text/event-stream, application/protobuf;q=0.1", true, "text/event-stream"},
		{"explicit refusal", "application/protobuf;q=0", true, "text/event-stream"},
		{"not streaming", "application/protobuf", false, ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			router := gin.New()
			router.POST("/v1/chat/completions", ChatMiddleware(), func(c *gin.Context) {
				body, _ := json.Marshal(api.ChatResponse{
					Model:      "test-model",
					Message:    api.Message{Role: "assistant", Content: "hi"},
					Done:       true,
					DoneReason: "stop",
				})
				c.Writer.Write(body)
			})

			body := `{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":` +
				map[bool]string{true: "true", false: "false"}[tc.stream] + `}`
			req, _ := http.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
			req.Header.Set("Content-Type", "application/json")
			if tc.accept != "" {
				req.Header.Set("Accept", tc.accept)
			}
			router.ServeHTTP(recorder, req)

			if recorder.Code != http.StatusOK {
				t.Fatalf("status %d: %s", recorder.Code, recorder.Body.String())
			}
			got := recorder.Header().Get("Content-Type")
			if tc.want == "" {
				if got == protoStreamContentType {
					t.Errorf("a non-streaming request was answered with %q", got)
				}
				return
			}
			if got != tc.want {
				t.Errorf("Content-Type = %q, want %q", got, tc.want)
			}
		})
	}
}
