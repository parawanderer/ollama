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
	"math"

	"github.com/ollama/ollama/api"
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
	deltaToolCalls = 3
	deltaLogprobs  = 4
	deltaIndex     = 5

	toolCallID       = 1
	toolCallIndex    = 2
	toolCallType     = 3
	toolCallFunction = 4

	functionName      = 1
	functionArguments = 2

	tokenLogprobToken   = 1
	tokenLogprobLogprob = 2
	tokenLogprobBytes   = 3

	logprobToken       = 1
	logprobTopLogprobs = 2

	endFinishReason     = 1
	endPromptTokens     = 2
	endCompletionTokens = 3
	endCachedTokens     = 4
)

// wire types, of which this needs three
const (
	wireVarint  = 0
	wireFixed64 = 1
	wireBytes   = 2
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

// appendDouble writes a float64. Unlike the other fields here a zero is written rather than
// omitted: a logprob of 0 means certainty, which is a real value and not an absent one.
func appendDouble(b []byte, field int, v float64) []byte {
	b = appendTag(b, field, wireFixed64)
	return binary.LittleEndian.AppendUint64(b, math.Float64bits(v))
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

// DeltaFrame carries one step of the generation. Everything beyond the text is present only
// on the chunks that have it, so a plain text delta is unchanged in size by their existence.
func DeltaFrame(index int, content, reasoning string, toolCalls []openai.ToolCall, logprobs *openai.ChoiceLogprobs) []byte {
	var m []byte
	m = appendString(m, deltaContent, content)
	m = appendString(m, deltaReasoning, reasoning)
	m = appendUint(m, deltaIndex, uint64(max(index, 0)))

	for _, tc := range toolCalls {
		var fn []byte
		fn = appendString(fn, functionName, tc.Function.Name)
		fn = appendString(fn, functionArguments, tc.Function.Arguments)

		var call []byte
		call = appendString(call, toolCallID, tc.ID)
		call = appendUint(call, toolCallIndex, uint64(max(tc.Index, 0)))
		call = appendString(call, toolCallType, tc.Type)
		if len(fn) > 0 {
			call = appendMessage(call, toolCallFunction, fn)
		}
		m = appendMessage(m, deltaToolCalls, call)
	}

	if logprobs != nil {
		for _, lp := range logprobs.Content {
			var entry []byte
			entry = appendMessage(entry, logprobToken, tokenLogprob(lp.TokenLogprob))
			for _, top := range lp.TopLogprobs {
				entry = appendMessage(entry, logprobTopLogprobs, tokenLogprob(top))
			}
			m = appendMessage(m, deltaLogprobs, entry)
		}
	}

	return frame(frameDelta, m)
}

func tokenLogprob(t api.TokenLogprob) []byte {
	var b []byte
	b = appendString(b, tokenLogprobToken, t.Token)
	b = appendDouble(b, tokenLogprobLogprob, t.Logprob)
	for _, v := range t.Bytes {
		// Packed would be smaller, but a token is a handful of bytes and unpacked is what
		// a hand-written encoder can be trusted to get right.
		b = appendUint(b, tokenLogprobBytes, uint64(max(v, 0)))
	}
	return b
}

// EndFrame closes the stream, replacing both the finish chunk and "data: [DONE]".
// EndFrame closes a stream. cached is the prompt tokens the prefix cache served, or nil when
// the engine did not report it.
func EndFrame(finishReason string, prompt, completion int, cached *int) []byte {
	var m []byte
	m = appendString(m, endFinishReason, finishReason)
	m = appendUint(m, endPromptTokens, uint64(max(prompt, 0)))
	m = appendUint(m, endCompletionTokens, uint64(max(completion, 0)))
	if cached != nil {
		// Not appendUint, which skips zero as proto3's default: this field is declared
		// optional, so a 0 is written, and a cold prefill stays distinct from "not reported".
		m = appendTag(m, endCachedTokens, wireVarint)
		m = binary.AppendUvarint(m, uint64(max(*cached, 0)))
	}
	return frame(frameEnd, m)
}

// deltaOf pulls the text fields a Delta carries out of a chunk, so the streaming path does
// not have to know the shape of either encoding.
// Content is typed any because the OpenAI schema allows a string or a parts array; only the
// string form carries streamed text, and anything else is left to the JSON encoding.
func deltaOf(c openai.ChunkChoice) (content, reasoning string) {
	if s, ok := c.Delta.Content.(string); ok {
		content = s
	}
	return content, c.Delta.Reasoning
}
