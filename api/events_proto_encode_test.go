package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/types/descriptorpb"
)

// testdata/event-frames.ndjson is one real frame of each of the 17 kinds this server has been
// observed to emit, taken verbatim from captures of it (the vectors in
// reports/ui-api/captures/ in the slop-zone repository). Real frames rather than frames built
// here, on purpose: a fixture written from memory encodes what its author thinks the server
// sends, which is the thing under test.
func realFrames(t *testing.T) []EventFrame {
	t.Helper()
	b, err := os.ReadFile("testdata/event-frames.ndjson")
	if err != nil {
		t.Fatal(err)
	}
	var frames []EventFrame
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		var f EventFrame
		if err := json.Unmarshal([]byte(line), &f); err != nil {
			t.Fatalf("testdata: %v", err)
		}
		frames = append(frames, f)
	}
	if len(frames) < 17 {
		t.Fatalf("testdata holds %d frames, want one of every kind", len(frames))
	}
	return frames
}

// The binary encoding must carry what the JSON carries -- not merely encode without error.
// Each frame goes out as protobuf, comes back as JSON, and every value the server sent has to
// still be there, because the failure that matters is a field quietly present in one encoding
// and missing from the other.
func TestEventFrameProtoCarriesEveryValue(t *testing.T) {
	for _, f := range realFrames(t) {
		body, err := MarshalEventFrameProto(f)
		if err != nil {
			t.Fatalf("%s: %v", f.Kind, err)
		}
		back, err := UnmarshalEventFrameProtoJSON(body)
		if err != nil {
			t.Fatalf("%s: %v", f.Kind, err)
		}

		sent, err := json.Marshal(f)
		if err != nil {
			t.Fatal(err)
		}
		var a, b map[string]any
		if err := json.Unmarshal(sent, &a); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(back, &b); err != nil {
			t.Fatal(err)
		}
		flatA, flatB := flattenJSON(a, ""), flattenJSON(b, "")

		for path, v := range flatA {
			got, ok := flatB[path]
			switch {
			case !ok && (v == nil || isEmptyContainer(v)):
				// A JSON null, an empty list and an absent key all mean "nothing to
				// report" on this stream, and protobuf has one way to say that.
			case !ok:
				t.Errorf("%s: %s = %v was sent and did not survive the round trip", f.Kind, path, v)
			case !sameJSONValue(v, got):
				t.Errorf("%s: %s went out as %v and came back as %v", f.Kind, path, v, got)
			}
		}
		for path, v := range flatB {
			if _, ok := flatA[path]; ok || isEmptyContainer(v) {
				// An empty container on either side says nothing: a frame that sent
				// `"ps":{"models":[]}` comes back as `"ps":{}`, because protobuf keeps
				// the message's presence and has no empty list to put in it.
				continue
			}
			t.Errorf("%s: %s = %v came back and was never sent — an absence became a fact",
				f.Kind, path, v)
		}
	}
}

// The stream is a sequence, so the framing has to be readable as one: each message preceded by
// its length, which is what every protobuf runtime's parseDelimitedFrom expects.
func TestEventFrameProtoStreamIsDelimited(t *testing.T) {
	frames := realFrames(t)
	var buf bytes.Buffer
	for _, f := range frames {
		if err := WriteEventFrameProto(&buf, f); err != nil {
			t.Fatal(err)
		}
	}

	r := bytes.NewReader(buf.Bytes())
	var kinds, want []string
	for range frames {
		body, err := ReadEventFrameProto(r)
		if err != nil {
			t.Fatalf("reading back: %v", err)
		}
		back, err := UnmarshalEventFrameProtoJSON(body)
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(back, &m); err != nil {
			t.Fatal(err)
		}
		kinds = append(kinds, fmt.Sprint(m["kind"]))
	}
	if r.Len() != 0 {
		t.Errorf("%d bytes left over after %d frames", r.Len(), len(frames))
	}
	for _, f := range frames {
		want = append(want, f.Kind)
	}
	if !reflect.DeepEqual(kinds, want) {
		t.Errorf("kinds came back as %v, want %v", kinds, want)
	}
}

// A field the schema does not declare must fail loudly. Silence is the failure mode that
// matters: a consumer cannot tell a field the server omitted from one the encoder dropped, so
// the encoder must never be the one that decides. This is the runtime backstop for the case
// TestEventsProtoLockstep catches at build time.
func TestEventFrameProtoRefusesAnUndeclaredField(t *testing.T) {
	md, err := eventFrameDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	if md.Fields().ByJSONName("slop_not_a_field") != nil {
		t.Skip("the schema grew a field named like the probe")
	}

	_, err = unmarshalFrameJSON([]byte(`{"v":1,"kind":"sample","t":1,"slop_not_a_field":"surprise"}`), md)
	if err == nil {
		t.Fatal("an undeclared field was accepted; it must be an error, not a silent drop")
	}
	if !strings.Contains(err.Error(), "slop_not_a_field") {
		t.Errorf("the error must name the field, got: %v", err)
	}

	// And the same JSON without that field must encode, so the test is about the field and
	// not about the frame.
	if _, err := unmarshalFrameJSON([]byte(`{"v":1,"kind":"sample","t":1}`), md); err != nil {
		t.Fatalf("a declared frame must encode: %v", err)
	}
}

// The committed descriptor is generated from events.proto, so it can fall behind it -- and a
// stale descriptor is invisible at runtime: the encoder keeps working and quietly serves the
// old shape. Regenerate with:
//
//	protoc -Iapi --include_imports --descriptor_set_out=api/events.pb.desc api/events.proto
func TestEventFrameDescriptorMatchesProto(t *testing.T) {
	var set descriptorpb.FileDescriptorSet
	if err := proto.Unmarshal(eventFrameDescriptorSet, &set); err != nil {
		t.Fatal(err)
	}
	files, err := protodesc.NewFiles(&set)
	if err != nil {
		t.Fatal(err)
	}

	for msgName, fields := range parseProto(t, "events.proto") {
		d, err := files.FindDescriptorByName(protoreflect.FullName("slop.events.v1." + msgName))
		if err != nil {
			t.Errorf("events.proto declares message %s, events.pb.desc does not — regenerate it", msgName)
			continue
		}
		md, ok := d.(protoreflect.MessageDescriptor)
		if !ok {
			t.Errorf("%s is a %T in events.pb.desc, not a message", msgName, d)
			continue
		}
		if md.Fields().Len() != len(fields) {
			t.Errorf("%s: events.proto has %d fields, events.pb.desc has %d — regenerate it",
				msgName, len(fields), md.Fields().Len())
		}
		for name, f := range fields {
			fd := md.Fields().ByName(protoreflect.Name(name))
			if fd == nil {
				t.Errorf("%s.%s is in events.proto and not in events.pb.desc — regenerate it", msgName, name)
				continue
			}
			if fd.HasOptionalKeyword() != f.optional {
				t.Errorf("%s.%s: events.proto says optional=%v, events.pb.desc says %v — regenerate it",
					msgName, name, f.optional, fd.HasOptionalKeyword())
			}
			if fd.IsList() != f.repeated {
				t.Errorf("%s.%s: events.proto says repeated=%v, events.pb.desc says %v — regenerate it",
					msgName, name, f.repeated, fd.IsList())
			}
		}
	}
}

func flattenJSON(v any, prefix string) map[string]any {
	out := map[string]any{}
	switch t := v.(type) {
	case map[string]any:
		for k, vv := range t {
			for p, leaf := range flattenJSON(vv, prefix+k+".") {
				out[p] = leaf
			}
		}
		if len(t) == 0 {
			out[strings.TrimSuffix(prefix, ".")] = t
		}
	case []any:
		for i, vv := range t {
			for p, leaf := range flattenJSON(vv, fmt.Sprintf("%s%d.", prefix, i)) {
				out[p] = leaf
			}
		}
		if len(t) == 0 {
			out[strings.TrimSuffix(prefix, ".")] = t
		}
	default:
		out[strings.TrimSuffix(prefix, ".")] = v
	}
	return out
}

func isEmptyContainer(v any) bool {
	switch t := v.(type) {
	case []any:
		return len(t) == 0
	case map[string]any:
		return len(t) == 0
	}
	return false
}

// sameJSONValue compares two decoded JSON values across the two forms protobuf's JSON mapping
// legitimately uses: a 64-bit integer comes back as a string, and a timestamp is re-formatted
// with a different number of fractional digits.
func sameJSONValue(a, b any) bool {
	if a == nil || b == nil {
		return a == b
	}
	as, bs := jsonScalarString(a), jsonScalarString(b)
	if as == bs {
		return true
	}
	af, aok := jsonNumber(a)
	bf, bok := jsonNumber(b)
	if aok && bok {
		return af == bf
	}
	if strings.HasSuffix(as, "Z") && strings.HasSuffix(bs, "Z") {
		norm := func(s string) string {
			s = strings.TrimSuffix(s, "Z")
			if strings.Contains(s, ".") {
				s = strings.TrimRight(s, "0")
				s = strings.TrimSuffix(s, ".")
			}
			return s
		}
		return norm(as) == norm(bs)
	}
	return false
}

func jsonScalarString(v any) string {
	b, _ := json.Marshal(v)
	return strings.Trim(string(b), `"`)
}

func jsonNumber(v any) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case string:
		var f float64
		if err := json.Unmarshal([]byte(t), &f); err == nil {
			return f, true
		}
	}
	return 0, false
}
