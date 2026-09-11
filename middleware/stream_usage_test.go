package middleware

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"regexp"
	"testing"

	"github.com/gin-gonic/gin"
	"google.golang.org/protobuf/encoding/protowire"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

// A stream as the chat handler produces it with stream_metrics on: the engine's running
// count on every chunk, and the final totals on the last.
func runningCountStream() []api.ChatResponse {
	return []api.ChatResponse{
		{Model: "m", Message: api.Message{Thinking: "weighing it"}, Metrics: api.Metrics{PromptEvalCount: 40, EvalCount: 3}},
		{Model: "m", Message: api.Message{Content: "The answer"}, Metrics: api.Metrics{PromptEvalCount: 40, EvalCount: 7}},
		{Model: "m", Message: api.Message{Content: " is 42."}, Done: true, DoneReason: "stop", Metrics: api.Metrics{PromptEvalCount: 40, EvalCount: 9}},
	}
}

func writeStream(t *testing.T, w *ChatWriter, responses []api.ChatResponse) {
	t.Helper()
	for _, r := range responses {
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(b); err != nil {
			t.Fatal(err)
		}
	}
}

func newChatWriter(proto bool, opts *openai.StreamOptions) (*ChatWriter, *httptest.ResponseRecorder) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	return &ChatWriter{
		stream: true, proto: proto, streamOptions: opts, id: "chatcmpl-test",
		BaseWriter: BaseWriter{ResponseWriter: c.Writer},
	}, recorder
}

func TestContinuousUsageStatsPutsTheRunningCountOnEveryChunk(t *testing.T) {
	w, rec := newChatWriter(false, &openai.StreamOptions{IncludeUsage: true, ContinuousUsageStats: true})
	writeStream(t, w, runningCountStream())

	var counts []int
	for _, frame := range sseDataFrames(rec.Body.String()) {
		if frame == "[DONE]" {
			continue
		}
		var chunk openai.ChatCompletionChunk
		if err := json.Unmarshal([]byte(frame), &chunk); err != nil {
			t.Fatal(err)
		}
		if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Content == nil && chunk.Choices[0].Delta.Reasoning == "" {
			continue // finish and final-usage chunks: covered by the existing include_usage tests
		}
		if chunk.Usage == nil {
			t.Fatalf("a content chunk carried no usage: %s", frame)
		}
		if chunk.Usage.PromptTokens != 40 || chunk.Usage.TotalTokens != 40+chunk.Usage.CompletionTokens {
			t.Errorf("usage = %+v, want prompt 40 and a consistent total", chunk.Usage)
		}
		counts = append(counts, chunk.Usage.CompletionTokens)
	}
	if want := []int{3, 7, 9}; len(counts) != len(want) || counts[0] != 3 || counts[1] != 7 || counts[2] != 9 {
		t.Fatalf("running counts = %v, want %v", counts, want)
	}
}

func TestIncludeUsageAloneKeepsContentChunksBare(t *testing.T) {
	w, rec := newChatWriter(false, &openai.StreamOptions{IncludeUsage: true})
	writeStream(t, w, runningCountStream())
	for _, frame := range sseDataFrames(rec.Body.String()) {
		var chunk openai.ChatCompletionChunk
		if frame == "[DONE]" || json.Unmarshal([]byte(frame), &chunk) != nil {
			continue
		}
		if len(chunk.Choices) > 0 && chunk.Usage != nil {
			t.Fatalf("a client that did not ask got usage on a content chunk: %s", frame)
		}
	}
}

// deltaCounts decodes every frame in a protobuf stream with protowire and returns the
// completion_tokens field of each Delta: -1 where the field is absent.
func deltaCounts(t *testing.T, b []byte) []int {
	t.Helper()
	var out []int
	for len(b) > 0 {
		n, l := protowire.ConsumeVarint(b)
		if l < 0 || int(n) > len(b)-l {
			t.Fatalf("bad length prefix")
		}
		frame := b[l : l+int(n)]
		b = b[l+int(n):]
		num, typ, l := protowire.ConsumeTag(frame)
		if l < 0 || typ != protowire.BytesType {
			t.Fatal("bad frame tag")
		}
		if num != frameDelta {
			continue
		}
		msg, _ := protowire.ConsumeBytes(frame[l:])
		count := -1
		for len(msg) > 0 {
			fnum, ftyp, l := protowire.ConsumeTag(msg)
			msg = msg[l:]
			if fnum == deltaCompletionTokens && ftyp == protowire.VarintType {
				v, l := protowire.ConsumeVarint(msg)
				count, msg = int(v), msg[l:]
				continue
			}
			msg = msg[protowire.ConsumeFieldValue(fnum, ftyp, msg):]
		}
		out = append(out, count)
	}
	return out
}

func TestProtoDeltaCarriesTheRunningCountOnlyWhenAsked(t *testing.T) {
	w, rec := newChatWriter(true, &openai.StreamOptions{ContinuousUsageStats: true})
	writeStream(t, w, runningCountStream())
	if got := deltaCounts(t, rec.Body.Bytes()); len(got) != 3 || got[0] != 3 || got[1] != 7 || got[2] != 9 {
		t.Fatalf("delta counts = %v, want [3 7 9]", got)
	}

	w, rec = newChatWriter(true, nil)
	writeStream(t, w, runningCountStream())
	for _, c := range deltaCounts(t, rec.Body.Bytes()) {
		if c != -1 {
			t.Fatalf("a client that did not ask got a count on a delta: %v", c)
		}
	}
}

// Declared optional so that absent means "not asked for"; a zero -- a delta before the first
// generated token is counted -- must still be written.
func TestDeltaFrameWritesAZeroCount(t *testing.T) {
	zero := 0
	if got := deltaCounts(t, DeltaFrame(0, "x", "", nil, nil, &zero)); len(got) != 1 || got[0] != 0 {
		t.Fatalf("counts = %v, want an explicit 0", got)
	}
}

// Same tie as TestEndFrameMatchesChatProto, for the Delta field clients will generate from.
func TestDeltaCompletionTokensMatchesChatProto(t *testing.T) {
	src, err := os.ReadFile("chat.proto")
	if err != nil {
		t.Fatal(err)
	}
	body := regexp.MustCompile(`(?s)message Delta \{(.*?)\n\}`).FindSubmatch(src)
	if body == nil {
		t.Fatal("no Delta message in chat.proto")
	}
	m := regexp.MustCompile(`(?m)^\s*optional uint32 completion_tokens = (\d+);`).FindSubmatch(body[1])
	if m == nil || string(m[1]) != "6" || deltaCompletionTokens != 6 {
		t.Fatalf("chat.proto Delta.completion_tokens = %q, encoder writes field %d; want optional uint32 = 6 on both", m, deltaCompletionTokens)
	}
}
