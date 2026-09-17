package api

// Binary protobuf for /api/events, offered by content negotiation beside the NDJSON.
//
// Why it exists, since the bytes are not the reason: a relay that has to decode JSON and
// re-encode it is a place the data can change shape, and a phone that has to parse JSON pays
// for it in battery. Passing one encoding end to end, and handing a mobile client something
// its runtime decodes natively, are both worth more than the size. Measured over 194 real
// frames, protobuf+gzip is 0.563x the gzipped NDJSON we already serve -- 29 bytes a frame,
// which is not a reason on its own (reports/ui-api/event-stream-findings.md in slop-zone).
//
// THE ENCODER IS DRIVEN BY THE JSON, deliberately. It marshals the frame with encoding/json,
// exactly as the NDJSON path does, then parses that into a message built from events.proto's
// descriptor and marshals it to binary. Two marshals per frame, which at this stream's rate
// costs nothing measurable, and in exchange there is no second hand-written mapping of 301
// fields -- so a field added to EventFrame cannot reach one encoding and silently miss the
// other. It cannot even reach this one quietly: protojson is told to reject unknown fields,
// so a field the schema does not declare is an error at the first frame that carries it,
// which is the same moment TestEventsProtoLockstep would have gone red at build time.
//
// The descriptor is generated from events.proto and committed as events.pb.desc.
// TestEventFrameDescriptorMatchesProto fails if the two drift.

import (
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"sync"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/dynamicpb"
)

// EventFrameProtoContentType is the Content-Type of the binary stream, and
// EventFrameProtoAccept is what a client puts in Accept to ask for it. The framing is a
// varint byte count before each message -- protobuf's own delimited convention, what
// writeDelimitedTo and parseDelimitedFrom read and write -- stated in a parameter because
// there is no registered media type for a sequence of protobuf messages. Same pair the chat
// stream uses; see middleware/protostream.go, which explains the choice at length.
const (
	EventFrameProtoContentType = "application/protobuf; delimited=varint"
	EventFrameProtoAccept      = "application/protobuf"
)

//go:embed events.pb.desc
var eventFrameDescriptorSet []byte

// eventFrameDescriptor resolves the EventFrame message out of the embedded descriptor once.
// A failure here is a build problem, not a request problem, so it is kept and returned to
// every caller rather than retried per frame.
var eventFrameDescriptor = sync.OnceValues(func() (protoreflect.MessageDescriptor, error) {
	var set descriptorpb.FileDescriptorSet
	if err := proto.Unmarshal(eventFrameDescriptorSet, &set); err != nil {
		return nil, fmt.Errorf("events.pb.desc is not a FileDescriptorSet: %w", err)
	}
	files, err := protodesc.NewFiles(&set)
	if err != nil {
		return nil, fmt.Errorf("events.pb.desc does not resolve: %w", err)
	}
	d, err := files.FindDescriptorByName("slop.events.v1.EventFrame")
	if err != nil {
		return nil, fmt.Errorf("slop.events.v1.EventFrame not in events.pb.desc: %w", err)
	}
	md, ok := d.(protoreflect.MessageDescriptor)
	if !ok {
		return nil, fmt.Errorf("slop.events.v1.EventFrame is a %T, not a message", d)
	}
	return md, nil
})

// MarshalEventFrameProto encodes one frame as a protobuf message, without the length prefix.
func MarshalEventFrameProto(f EventFrame) ([]byte, error) {
	md, err := eventFrameDescriptor()
	if err != nil {
		return nil, err
	}
	b, err := json.Marshal(f)
	if err != nil {
		return nil, err
	}
	msg, err := unmarshalFrameJSON(b, md)
	if err != nil {
		return nil, err
	}
	return proto.Marshal(msg)
}

// unmarshalFrameJSON parses the frame's JSON into a message of the schema's shape.
//
// DiscardUnknown stays false, which is the whole point: a field events.proto does not declare
// must be an error rather than a silent omission, because a client reading this stream cannot
// tell a field the server omitted from one an encoder dropped.
func unmarshalFrameJSON(b []byte, md protoreflect.MessageDescriptor) (*dynamicpb.Message, error) {
	msg := dynamicpb.NewMessage(md)
	opts := protojson.UnmarshalOptions{Resolver: protoregistry.GlobalTypes}
	if err := opts.Unmarshal(b, msg); err != nil {
		return nil, fmt.Errorf("frame does not match events.proto: %w", err)
	}
	return msg, nil
}

// WriteEventFrameProto writes one frame, length-delimited.
func WriteEventFrameProto(w io.Writer, f EventFrame) error {
	body, err := MarshalEventFrameProto(f)
	if err != nil {
		return err
	}
	_, err = w.Write(append(protowire.AppendVarint(nil, uint64(len(body))), body...))
	return err
}

// UnmarshalEventFrameProtoJSON decodes one message back to the JSON the frame was built
// from. It exists for the tests and for a client written in Go; a client in another language
// generates its own code from events.proto and does not need this.
func UnmarshalEventFrameProtoJSON(body []byte) ([]byte, error) {
	md, err := eventFrameDescriptor()
	if err != nil {
		return nil, err
	}
	msg := dynamicpb.NewMessage(md)
	if err := proto.Unmarshal(body, msg); err != nil {
		return nil, err
	}
	return protojson.MarshalOptions{UseProtoNames: true}.Marshal(msg)
}

// UnmarshalEventFrameProto decodes one message back into an EventFrame.
//
// It exists because protobuf's JSON mapping writes 64-bit integers as strings -- `"t":"1"` --
// which is correct and is what a generated client handles natively, but which encoding/json
// will not put into an int64 field. So the descriptor is walked alongside the decoded JSON and
// those fields are turned back into numbers. A client in another language never sees this: in
// the binary form they are varints, and its generated code gives it an integer.
func UnmarshalEventFrameProto(body []byte) (EventFrame, error) {
	var f EventFrame
	md, err := eventFrameDescriptor()
	if err != nil {
		return f, err
	}
	asJSON, err := UnmarshalEventFrameProtoJSON(body)
	if err != nil {
		return f, err
	}
	var tree any
	if err := json.Unmarshal(asJSON, &tree); err != nil {
		return f, err
	}
	fixed, err := json.Marshal(numbersFromStrings(tree, md))
	if err != nil {
		return f, err
	}
	return f, json.Unmarshal(fixed, &f)
}

// numbersFromStrings walks a decoded proto3-JSON tree with its descriptor and unquotes the
// fields protobuf renders as strings because they are 64 bits wide.
func numbersFromStrings(v any, md protoreflect.MessageDescriptor) any {
	m, ok := v.(map[string]any)
	if !ok || md == nil {
		return v
	}
	out := make(map[string]any, len(m))
	for k, vv := range m {
		fd := md.Fields().ByName(protoreflect.Name(k))
		if fd == nil {
			fd = md.Fields().ByJSONName(k)
		}
		out[k] = vv
		if fd == nil {
			continue
		}
		wide := fd.Kind() == protoreflect.Int64Kind || fd.Kind() == protoreflect.Uint64Kind ||
			fd.Kind() == protoreflect.Sint64Kind || fd.Kind() == protoreflect.Fixed64Kind ||
			fd.Kind() == protoreflect.Sfixed64Kind
		switch {
		case wide:
			out[k] = unquoteNumber(vv)
		case fd.Kind() == protoreflect.MessageKind:
			if list, isList := vv.([]any); isList {
				items := make([]any, len(list))
				for i, item := range list {
					items[i] = numbersFromStrings(item, fd.Message())
				}
				out[k] = items
			} else {
				out[k] = numbersFromStrings(vv, fd.Message())
			}
		}
	}
	return out
}

func unquoteNumber(v any) any {
	switch t := v.(type) {
	case string:
		var n json.Number
		if err := json.Unmarshal([]byte(t), &n); err == nil {
			return n
		}
	case []any:
		out := make([]any, len(t))
		for i, item := range t {
			out[i] = unquoteNumber(item)
		}
		return out
	}
	return v
}

// ReadEventFrameProto reads one length-delimited message from r, returning its bytes. The
// varint is read a byte at a time rather than through a buffered reader, so this consumes
// exactly one frame and leaves the rest of the stream for the next call -- which is what a
// caller reading a live response body needs.
func ReadEventFrameProto(r io.Reader) ([]byte, error) {
	var head []byte
	var one [1]byte
	for {
		if _, err := io.ReadFull(r, one[:]); err != nil {
			return nil, err
		}
		head = append(head, one[0])
		if one[0] < 0x80 {
			break
		}
		if len(head) > binary.MaxVarintLen64 {
			return nil, fmt.Errorf("length prefix does not end")
		}
	}
	n, consumed := protowire.ConsumeVarint(head)
	if consumed < 0 {
		return nil, fmt.Errorf("bad length prefix")
	}
	body := make([]byte, n)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, err
	}
	return body, nil
}
