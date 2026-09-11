package middleware

import (
	"os"
	"regexp"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"
)

// endFields decodes an End frame with a real protobuf parser rather than the encoder's own
// helpers, so the test cannot agree with the encoder merely by sharing its mistakes.
func endFields(t *testing.T, b []byte) map[protowire.Number]uint64 {
	t.Helper()
	n, l := protowire.ConsumeVarint(b)
	if l < 0 || int(n) != len(b)-l {
		t.Fatalf("bad length prefix: %d over %d bytes", n, len(b)-l)
	}
	b = b[l:]
	num, typ, l := protowire.ConsumeTag(b)
	if l < 0 || num != frameEnd || typ != protowire.BytesType {
		t.Fatalf("frame is field %d type %d, want End (%d) as bytes", num, typ, frameEnd)
	}
	msg, l := protowire.ConsumeBytes(b[l:])
	if l < 0 {
		t.Fatal("truncated End message")
	}
	out := make(map[protowire.Number]uint64)
	for len(msg) > 0 {
		num, typ, l := protowire.ConsumeTag(msg)
		if l < 0 {
			t.Fatal("bad tag")
		}
		msg = msg[l:]
		switch typ {
		case protowire.VarintType:
			v, l := protowire.ConsumeVarint(msg)
			out[num], msg = v, msg[l:]
		default:
			l := protowire.ConsumeFieldValue(num, typ, msg)
			msg = msg[l:]
		}
	}
	return out
}

// A cold prefill must put an explicit 0 on the wire; an unreported count must be absent. A
// plain proto3 uint32 cannot tell them apart, which is why the field is declared optional.
func TestEndFrameCarriesCachedTokensWithPresence(t *testing.T) {
	zero, some := 0, 16
	cold := endFields(t, EndFrame("stop", 17, 20, &zero))
	if v, ok := cold[endCachedTokens]; !ok || v != 0 {
		t.Errorf("cold prefill: cached_tokens present=%v value=%d, want an explicit 0", ok, v)
	}
	warm := endFields(t, EndFrame("stop", 17, 20, &some))
	if warm[endCachedTokens] != 16 || warm[endPromptTokens] != 17 || warm[endCompletionTokens] != 20 {
		t.Errorf("warm: %v", warm)
	}
	if _, ok := endFields(t, EndFrame("stop", 17, 20, nil))[endCachedTokens]; ok {
		t.Error("an unreported count was written; absent must mean not reported")
	}
}

// The encoder is hand-written beside a .proto that clients generate their decoders from, and
// nothing else ties the two together. This checks every End field number the encoder writes
// against the schema's, so a field added on one side only fails here instead of in a client.
func TestEndFrameMatchesChatProto(t *testing.T) {
	src, err := os.ReadFile("chat.proto")
	if err != nil {
		t.Fatal(err)
	}
	body := regexp.MustCompile(`(?s)message End \{(.*?)\}`).FindSubmatch(src)
	if body == nil {
		t.Fatal("no End message in chat.proto")
	}
	fields := map[string]string{}
	for _, m := range regexp.MustCompile(`(?m)^\s*((?:optional )?\w+) (\w+) = (\d+);`).FindAllSubmatch(body[1], -1) {
		fields[string(m[2])] = string(m[1]) + "=" + string(m[3])
	}
	for name, want := range map[string]string{
		"finish_reason":     "string=1",
		"prompt_tokens":     "uint32=2",
		"completion_tokens": "uint32=3",
		"cached_tokens":     "optional uint32=4",
	} {
		if fields[name] != want {
			t.Errorf("chat.proto End.%s = %q, want %q to match the encoder", name, fields[name], want)
		}
	}
	if len(fields) != 4 {
		t.Errorf("chat.proto End has %d fields, the encoder writes 4: %v", len(fields), fields)
	}
}
