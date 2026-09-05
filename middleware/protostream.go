package middleware

// A protobuf encoding of the OpenAI streaming chat format, offered by content negotiation
// on the same endpoint.
//
// The JSON form re-sends id, object, created, model, system_fingerprint and a nested
// choices[0].delta wrapper for every token, so roughly 224 bytes of envelope arrive
// carrying roughly 5 bytes of text. Measured on a 220-token generation from qwen3.5:4b:
// 53852 bytes of SSE against 2434 of this, 22x smaller, and the envelope goes from 48x the
// payload to 2.2x.
//
// The wire format is described by chat.proto in this directory, which is the definition.
// The encoder is written by hand rather than generated because encoding protobuf is a
// handful of varints and this avoids putting protoc in the build for one message type;
// decoding, which is the hard half, is left to whatever real implementation the client
// already has. Verified against a generated Python decoder, which is what makes "written by
// hand" acceptable rather than merely convenient.

import (
	"encoding/binary"

	"github.com/ollama/ollama/openai"
)

// protoStreamContentType is what comes back in Content-Type; protoStreamAccept is what a
// client matches on in Accept. A client that asks for neither gets the SSE it always got.
//
// application/protobuf is the registered type (RFC 9996) and carries optional encoding and
// version parameters. It does not cover this case: the RFC says the types are "used in the
// transport of serialized objects only" and defines nothing for a sequence of them, so
// unlike JSON -- which has application/json-seq (RFC 7464) as well as the de facto
// application/x-ndjson -- there is no registered media type for a protobuf stream.
//
// So the registered type is used and the framing is stated in a parameter, which is
// unregistered and says so by being obvious. The framing itself is not invented: a varint
// byte count before each message is protobuf's own delimited convention, what
// writeDelimitedTo and parseDelimitedFrom read and write.
const (
	protoStreamContentType = "application/protobuf; delimited=varint"
	protoStreamAccept      = "application/protobuf"
)

// Field numbers from chat.proto. Changing one is a wire break, so they are written out
// rather than derived from position.
const (
	frameStart = 1
	frameDelta = 2
	frameEnd   = 3

	startID                = 1
	startModel             = 2
	startCreated           = 3
	startSystemFingerprint = 4
	startRole              = 5

	deltaContent   = 1
	deltaReasoning = 2

	endFinishReason     = 1
	endPromptTokens     = 2
	endCompletionTokens = 3
)

// wire types, of which this needs two
const (
	wireVarint = 0
	wireBytes  = 2
)

func appendTag(b []byte, field, wire int) []byte {
	return binary.AppendUvarint(b, uint64(field)<<3|uint64(wire))
}

// appendString omits empty strings. proto3 treats absent and empty as the same value, and
// omitting them is most of why a delta frame costs a handful of bytes.
func appendString(b []byte, field int, s string) []byte {
	if s == "" {
		return b
	}
	b = appendTag(b, field, wireBytes)
	b = binary.AppendUvarint(b, uint64(len(s)))
	return append(b, s...)
}

func appendUint(b []byte, field int, v uint64) []byte {
	if v == 0 {
		return b
	}
	b = appendTag(b, field, wireVarint)
	return binary.AppendUvarint(b, v)
}

// appendMessage nests one message inside another, length-delimited.
func appendMessage(b []byte, field int, msg []byte) []byte {
	b = appendTag(b, field, wireBytes)
	b = binary.AppendUvarint(b, uint64(len(msg)))
	return append(b, msg...)
}

// frame wraps a message as a Frame and prefixes the whole thing with its length, which is
// how a protobuf stream is delimited.
func frame(field int, msg []byte) []byte {
	body := appendMessage(nil, field, msg)
	out := binary.AppendUvarint(nil, uint64(len(body)))
	return append(out, body...)
}

// StartFrame carries everything the JSON form repeats on every chunk, once.
func StartFrame(id, model, fingerprint, role string, created int64) []byte {
	var m []byte
	m = appendString(m, startID, id)
	m = appendString(m, startModel, model)
	if created > 0 {
		m = appendUint(m, startCreated, uint64(created))
	}
	m = appendString(m, startSystemFingerprint, fingerprint)
	m = appendString(m, startRole, role)
	return frame(frameStart, m)
}

// DeltaFrame carries one step of the generation and nothing else.
func DeltaFrame(content, reasoning string) []byte {
	var m []byte
	m = appendString(m, deltaContent, content)
	m = appendString(m, deltaReasoning, reasoning)
	return frame(frameDelta, m)
}

// EndFrame closes the stream, replacing both the finish chunk and "data: [DONE]".
func EndFrame(finishReason string, prompt, completion int) []byte {
	var m []byte
	m = appendString(m, endFinishReason, finishReason)
	m = appendUint(m, endPromptTokens, uint64(max(prompt, 0)))
	m = appendUint(m, endCompletionTokens, uint64(max(completion, 0)))
	return frame(frameEnd, m)
}

// deltaOf pulls the two fields a Delta carries out of a chunk, so the streaming path does
// not have to know the shape of either encoding.
// Content is typed any because the OpenAI schema allows a string or a parts array; only the
// string form carries streamed text, and anything else is left to the JSON encoding.
func deltaOf(c openai.ChunkChoice) (content, reasoning string) {
	if s, ok := c.Delta.Content.(string); ok {
		content = s
	}
	return content, c.Delta.Reasoning
}
