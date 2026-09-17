package api

import (
	"fmt"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"
)

// events.proto describes the /api/events NDJSON, and nothing enforces that a field added
// here is added there — a schema that silently falls behind the encoder is worse than no
// schema, because a consumer generates code from it and believes it.
//
// So this walks EventFrame by reflection, walks the proto, and requires them to agree on
// every JSON path. Reflection rather than a regex over the source: the encoder is
// encoding/json, and reflection is what it reads, so this cannot disagree with the wire
// about a tag, an embedded struct or a type it does not recognise.
//
// Adding a field to a struct here is meant to turn this red. To see what to add:
//
//	SLOP_EVENTS_PROTO_SKELETON=1 go test ./api -run TestEventsProtoLockstep
//
// prints the proto the current Go types describe, to diff into events.proto by hand. The
// file stays hand-written — a generated one could not catch drift, since it would be
// regenerated from whatever the Go said at the time.

// protoField is one field as either side declares it.
type protoField struct {
	typeName string // proto type: a scalar, a message name, or a well-known type
	repeated bool
	// optional is explicit presence. The rule, which the test enforces in both
	// directions, is that a field is optional exactly when the JSON key can appear
	// carrying a zero: a Go pointer (absent means not reported), or a Go value without
	// omitempty (always written, including as 0). A Go value WITH omitempty is the
	// remaining case, where the encoder omits the zero so absent and zero are the same
	// fact, and proto3's implicit presence says exactly that.
	//
	// Getting this wrong is not cosmetic. With the field implicit, protobuf's JSON
	// mapping drops it whenever it is zero, so a frame that said `slots_busy: 0`
	// round-trips to a frame that does not mention slots_busy -- which on this stream
	// reads as "the runner could not be asked". The first version of this test used
	// "pointer" alone as the rule and the conformance notebook found 1221 such paths
	// across 117 real frames.
	optional bool
}

// goProtoType maps a Go type to the proto type that describes the JSON encoding/json
// produces for it. The mapping is deliberately narrow: an unrecognised type fails the test
// rather than being described as something vague, because "this field is a Struct" is how a
// schema stops saying anything useful.
func goProtoType(t reflect.Type) (string, bool) {
	if t == reflect.TypeOf(time.Time{}) {
		return "google.protobuf.Timestamp", true
	}
	switch t.Kind() {
	case reflect.Bool:
		return "bool", true
	case reflect.Int, reflect.Int32:
		return "int32", true
	case reflect.Int64:
		return "int64", true
	case reflect.Uint, reflect.Uint32:
		return "uint32", true
	case reflect.Uint64:
		return "uint64", true
	case reflect.Float32:
		return "float", true
	case reflect.Float64:
		return "double", true
	case reflect.String:
		return "string", true
	case reflect.Struct:
		return t.Name(), true
	case reflect.Map:
		if t.Key().Kind() == reflect.String && t.Elem().Kind() == reflect.String {
			return "map<string, string>", true
		}
		return "google.protobuf.Struct", true
	case reflect.Interface:
		return "google.protobuf.Value", true
	}
	return "", false
}

// goSurface returns every JSON path reachable from a struct type, and the messages those
// paths pass through. Cycles are guarded per branch rather than globally: the same type
// legitimately appears at two paths (MemoryBreakdown hangs off both the model row and each
// GPU) and both are places a consumer has to read.
func goSurface(t reflect.Type) (map[string]protoField, map[string]map[string]protoField) {
	paths := map[string]protoField{}
	msgs := map[string]map[string]protoField{}
	order := map[string][]string{}

	var walk func(t reflect.Type, prefix string, seen []reflect.Type)
	walk = func(t reflect.Type, prefix string, seen []reflect.Type) {
		msgName := t.Name()
		if msgs[msgName] == nil {
			msgs[msgName] = map[string]protoField{}
		}
		for i := range t.NumField() {
			f := t.Field(i)
			if !f.IsExported() {
				continue
			}
			tag := f.Tag.Get("json")
			if tag == "-" {
				continue
			}
			name, _, _ := strings.Cut(tag, ",")
			if name == "" {
				name = f.Name
			}

			ft := f.Type
			pf := protoField{optional: !strings.Contains(tag, ",omitempty")}
			if ft.Kind() == reflect.Pointer {
				pf.optional, ft = true, ft.Elem()
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() != reflect.Uint8 {
				pf.repeated, ft = true, ft.Elem()
				if ft.Kind() == reflect.Pointer {
					ft = ft.Elem()
				}
				pf.optional = false // repeated has no presence, on either side
			}
			typeName, ok := goProtoType(ft)
			if !ok {
				typeName = "UNMAPPED:" + ft.String()
			}
			pf.typeName = typeName

			path := prefix + name
			if pf.repeated {
				path += "[]"
			}
			paths[path] = pf
			if _, dup := msgs[msgName][name]; !dup {
				order[msgName] = append(order[msgName], name)
			}
			msgs[msgName][name] = pf

			if ft.Kind() == reflect.Struct && ft != reflect.TypeOf(time.Time{}) {
				cycle := false
				for _, s := range seen {
					if s == ft {
						cycle = true
					}
				}
				if !cycle {
					walk(ft, path+".", append(seen, t))
				}
			}
		}
	}

	walk(t, "", nil)
	goFieldOrder = order
	return paths, msgs
}

// goFieldOrder is the order the fields are declared in Go, which the skeleton follows so it
// can be read beside types.go. Set by goSurface.
var goFieldOrder map[string][]string

var (
	protoMessageRE = regexp.MustCompile(`^message (\w+) \{`)
	protoFieldRE   = regexp.MustCompile(`^\s*(optional |repeated )?([\w\.]+|map<[^>]+>) (\w+) = (\d+);`)
)

// parseProto reads the messages and fields out of a .proto file. It is a few lines because
// the file is a few constructs: this is a reader for the schema this repository writes, not
// a protobuf parser, and it fails loudly on anything it does not recognise.
func parseProto(t *testing.T, path string) map[string]map[string]protoField {
	t.Helper()
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	msgs := map[string]map[string]protoField{}
	var current string
	for i, line := range strings.Split(string(src), "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "//") {
			continue
		}
		if m := protoMessageRE.FindStringSubmatch(line); m != nil {
			current = m[1]
			if msgs[current] == nil {
				msgs[current] = map[string]protoField{}
			}
			continue
		}
		if trimmed == "}" {
			current = ""
			continue
		}
		if current == "" {
			continue // syntax, package, import, option
		}
		m := protoFieldRE.FindStringSubmatch(line)
		if m == nil {
			t.Fatalf("%s:%d: cannot read as a field declaration: %q", path, i+1, trimmed)
		}
		msgs[current][m[3]] = protoField{
			typeName: m[2],
			repeated: strings.TrimSpace(m[1]) == "repeated",
			optional: strings.TrimSpace(m[1]) == "optional",
		}
	}
	return msgs
}

// protoSurface expands the proto's messages into the same JSON paths goSurface produces, so
// the two can be compared as sets rather than message by message.
func protoSurface(msgs map[string]map[string]protoField, root string) map[string]protoField {
	paths := map[string]protoField{}
	var walk func(msg, prefix string, seen []string)
	walk = func(msg, prefix string, seen []string) {
		for name, f := range msgs[msg] {
			path := prefix + name
			if f.repeated {
				path += "[]"
			}
			paths[path] = f
			if _, nested := msgs[f.typeName]; nested {
				cycle := false
				for _, s := range seen {
					if s == f.typeName {
						cycle = true
					}
				}
				if !cycle {
					walk(f.typeName, path+".", append(seen, msg))
				}
			}
		}
	}
	walk(root, "", nil)
	return paths
}

func TestEventsProtoLockstep(t *testing.T) {
	goPaths, goMsgs := goSurface(reflect.TypeOf(EventFrame{}))

	if os.Getenv("SLOP_EVENTS_PROTO_SKELETON") != "" {
		t.Log("\n" + protoSkeleton(goMsgs))
	}

	for path, f := range goPaths {
		if strings.HasPrefix(f.typeName, "UNMAPPED:") {
			t.Errorf("%s is a %s, which events.proto has no mapping for: add one to goProtoType, "+
				"or the schema cannot describe this field", path, strings.TrimPrefix(f.typeName, "UNMAPPED:"))
		}
	}

	protoPaths := protoSurface(parseProto(t, "events.proto"), "EventFrame")

	var missing, extra, mismatched []string
	for path, g := range goPaths {
		p, ok := protoPaths[path]
		if !ok {
			missing = append(missing, fmt.Sprintf("%s (%s)", path, g.typeName))
			continue
		}
		switch {
		case p.typeName != g.typeName:
			mismatched = append(mismatched, fmt.Sprintf("%s: proto says %s, Go encodes %s", path, p.typeName, g.typeName))
		case p.repeated != g.repeated:
			mismatched = append(mismatched, fmt.Sprintf("%s: proto repeated=%v, Go repeated=%v", path, p.repeated, g.repeated))
		case p.optional != g.optional && !g.repeated && !isMessage(g.typeName):
			// See protoField.optional: optional exactly when the key can carry a zero.
			mismatched = append(mismatched, fmt.Sprintf(
				"%s: proto optional=%v, encoder can send this key carrying a zero=%v",
				path, p.optional, g.optional))
		}
	}
	for path := range protoPaths {
		if _, ok := goPaths[path]; !ok {
			extra = append(extra, path)
		}
	}
	sort.Strings(missing)
	sort.Strings(extra)
	sort.Strings(mismatched)

	if len(missing) > 0 {
		t.Errorf("events.proto does not declare %d path(s) the stream can carry:\n  %s\n"+
			"Regenerate with SLOP_EVENTS_PROTO_SKELETON=1 and diff it in.",
			len(missing), strings.Join(missing, "\n  "))
	}
	if len(extra) > 0 {
		t.Errorf("events.proto declares %d path(s) the stream no longer carries:\n  %s",
			len(extra), strings.Join(extra, "\n  "))
	}
	if len(mismatched) > 0 {
		t.Errorf("events.proto disagrees with the encoder on %d path(s):\n  %s",
			len(mismatched), strings.Join(mismatched, "\n  "))
	}
	if t.Failed() {
		return
	}
	if len(goPaths) < 100 {
		t.Fatalf("walked only %d paths, which is too few to be the real surface", len(goPaths))
	}
	t.Logf("events.proto matches the encoder on %d JSON paths", len(goPaths))
}

func isMessage(typeName string) bool {
	return typeName != "" && strings.ToUpper(typeName[:1]) == typeName[:1] &&
		!strings.HasPrefix(typeName, "map<")
}

// protoSkeleton prints the messages the Go types describe. Field numbers restart at 1 per
// message and will not match the committed file; it is a list to diff, not a file to copy
// over, because the numbers in the real file are a wire contract.
func protoSkeleton(msgs map[string]map[string]protoField) string {
	var b strings.Builder
	names := make([]string, 0, len(msgs))
	for name := range msgs {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		fmt.Fprintf(&b, "message %s {\n", name)
		fields := goFieldOrder[name]
		for i, f := range fields {
			pf := msgs[name][f]
			prefix := ""
			if pf.repeated {
				prefix = "repeated "
			} else if pf.optional {
				prefix = "optional "
			}
			fmt.Fprintf(&b, "  %s%s %s = %d;\n", prefix, pf.typeName, f, i+1)
		}
		b.WriteString("}\n\n")
	}
	return b.String()
}
