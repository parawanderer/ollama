package api

import (
	"encoding/json"
	"strings"
	"testing"
)

// A hint is copied into every gen.end event, so what a caller can put in it is bounded, and an
// empty hint is the same as none.
func TestRequestHintSanitized(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   *RequestHint
		want *RequestHint
	}{
		{"nil", nil, nil},
		{"empty is none", &RequestHint{Use: "  ", Session: ""}, nil},
		{"kept as sent, trimmed", &RequestHint{Use: " agent ", Session: " s1 "}, &RequestHint{Use: "agent", Session: "s1"}},
		// Unknown values are recorded, not rejected: a newer client must not break an older server.
		{"unknown use kept", &RequestHint{Use: "rerank"}, &RequestHint{Use: "rerank"}},
		{"bounded", &RequestHint{Use: strings.Repeat("u", 100), Session: strings.Repeat("s", 500),
			Request: strings.Repeat("r", 100), After: strings.Repeat("a", 100)},
			&RequestHint{Use: strings.Repeat("u", hintUseMax), Session: strings.Repeat("s", hintSessionMax),
				Request: strings.Repeat("r", hintRequestMax), After: strings.Repeat("a", hintAfterMax)}},
		// Synthetic alone is still a hint: it keeps benchmark traffic out of learned usage.
		{"synthetic alone", &RequestHint{Synthetic: true}, &RequestHint{Synthetic: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.in.Sanitized()
			if (got == nil) != (tc.want == nil) || (got != nil && *got != *tc.want) {
				t.Fatalf("Sanitized() = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestRequestHintOnTheWire(t *testing.T) {
	var req ChatRequest
	if err := json.Unmarshal([]byte(`{"model":"m","hint":{"use":"agent","session":"abc"}}`), &req); err != nil {
		t.Fatal(err)
	}
	if req.Hint == nil || req.Hint.Use != "agent" || req.Hint.Session != "abc" {
		t.Fatalf("hint not decoded: %+v", req.Hint)
	}
	b, _ := json.Marshal(ChatRequest{Model: "m"})
	if strings.Contains(string(b), "hint") {
		t.Fatalf("an absent hint must not appear on the wire: %s", b)
	}
}
