package llm

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/fs/ggml"
)

// SyntheticModel is the shape of a llama-architecture model whose weights are all zero. It
// exists to time the hardware: decode reads every weight once per token whatever its value, so
// a model of zeros exercises the memory system exactly as a real model of the same shape does.
// Measured on 2026-09-12: Q8_0 zeros and Q8_0 random values of one shape decoded within 1.1%
// of each other (notebooks/placement/box-profile.ipynb in slop-zone).
//
// Zero weights make every activation zero, so the output is meaningless. Time it with a fixed
// number of forced tokens; never read what it says.
type SyntheticModel struct {
	Embedding int // n_embd; a multiple of 256
	Layers    int
}

// syntheticVocab is small enough to cost nothing and valid for llama.cpp's sentencepiece
// loader: unk, bos, eos, the ASCII byte tokens, then filler. Byte tokens stop at 0x7F because
// a lone byte above it is invalid UTF-8, and llama-server fails a request whose reply it cannot
// serialise. Every token here decodes to valid text on its own.
func syntheticVocab() (tokens []string, scores []float32, types []int32) {
	const n = 512
	tokens = []string{"<unk>", "<s>", "</s>"}
	types = []int32{2, 3, 3}
	for b := range 0x80 {
		tokens = append(tokens, fmt.Sprintf("<0x%02X>", b))
		types = append(types, 6)
	}
	for len(tokens) < n {
		tokens = append(tokens, fmt.Sprintf("▁t%d", len(tokens)))
		types = append(types, 1)
	}
	return tokens, make([]float32, n), types
}

// zeros writes nothing, which leaves the tensor's data as a hole in a sparse file.
type zeros struct{}

func (zeros) WriteTo(io.Writer) (int64, error) { return 0, nil }

func (m SyntheticModel) tensors() []*ggml.Tensor {
	n := uint64(m.Embedding)
	ff := n * 7 / 2
	kv := n / 4 // head size 128, one KV head per four query heads
	const vocab = 512
	q4 := uint32(ggml.TensorTypeQ4_K)
	f32 := uint32(ggml.TensorTypeF32)

	ts := []*ggml.Tensor{
		{Name: "token_embd.weight", Kind: q4, Shape: []uint64{n, vocab}},
		{Name: "output_norm.weight", Kind: f32, Shape: []uint64{n}},
		{Name: "output.weight", Kind: q4, Shape: []uint64{n, vocab}},
	}
	for i := range m.Layers {
		p := fmt.Sprintf("blk.%d.", i)
		ts = append(ts,
			&ggml.Tensor{Name: p + "attn_norm.weight", Kind: f32, Shape: []uint64{n}},
			&ggml.Tensor{Name: p + "attn_q.weight", Kind: q4, Shape: []uint64{n, n}},
			&ggml.Tensor{Name: p + "attn_k.weight", Kind: q4, Shape: []uint64{n, kv}},
			&ggml.Tensor{Name: p + "attn_v.weight", Kind: q4, Shape: []uint64{n, kv}},
			&ggml.Tensor{Name: p + "attn_output.weight", Kind: q4, Shape: []uint64{n, n}},
			&ggml.Tensor{Name: p + "ffn_norm.weight", Kind: f32, Shape: []uint64{n}},
			&ggml.Tensor{Name: p + "ffn_gate.weight", Kind: q4, Shape: []uint64{n, ff}},
			&ggml.Tensor{Name: p + "ffn_up.weight", Kind: q4, Shape: []uint64{n, ff}},
			&ggml.Tensor{Name: p + "ffn_down.weight", Kind: q4, Shape: []uint64{ff, n}},
		)
	}
	for _, t := range ts {
		t.WriterTo = zeros{}
	}
	return ts
}

// ReadPerToken is the bytes one decoded token reads: every weight but the token embedding,
// of which decode reads one row. The KV cache is left out; at the few dozen tokens of context
// used to time decode it is a fraction of a percent.
func (m SyntheticModel) ReadPerToken() uint64 {
	var total uint64
	for _, t := range m.tensors() {
		if t.Name != "token_embd.weight" {
			total += t.Size()
		}
	}
	return total
}

// Write writes the model to path as a sparse file: the header is real and the data is a hole,
// so a model of many GiB costs a few KiB of disk. An all-zero block is a valid Q4_K block (its
// scales are zero too), so no values need writing.
func (m SyntheticModel) Write(path string) error {
	if m.Embedding <= 0 || m.Embedding%256 != 0 || m.Layers <= 0 {
		return fmt.Errorf("synthetic model: embedding %d must be a positive multiple of 256, layers %d positive", m.Embedding, m.Layers)
	}
	tokens, scores, types := syntheticVocab()
	n := uint32(m.Embedding)
	kv := ggml.KV{
		"general.architecture":                   "llama",
		"general.name":                           fmt.Sprintf("synthetic-%dx%d", m.Layers, m.Embedding),
		"llama.context_length":                   uint32(8192),
		"llama.embedding_length":                 n,
		"llama.block_count":                      uint32(m.Layers),
		"llama.feed_forward_length":              n * 7 / 2,
		"llama.attention.head_count":             n / 128,
		"llama.attention.head_count_kv":          max(1, n/512),
		"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
		"tokenizer.ggml.model":                   "llama",
		"tokenizer.ggml.tokens":                  tokens,
		"tokenizer.ggml.scores":                  scores,
		"tokenizer.ggml.token_type":              types,
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := markSparse(f); err != nil {
		os.Remove(path)
		return fmt.Errorf("synthetic model: the filesystem at %s cannot hold sparse files, so the model would take its full size on disk: %w", filepath.Dir(path), err)
	}
	ts := m.tensors()
	if err := ggml.WriteGGUF(f, kv, ts); err != nil {
		return err
	}

	// WriteGGUF wrote the header and nothing of the data, and left ts sorted with their
	// offsets set. Extending the file to where the last tensor ends makes the data region a
	// hole that reads as zeros.
	header, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	const alignment = 32
	header += (alignment - header%alignment) % alignment
	var end uint64
	for _, t := range ts {
		end = max(end, t.Offset+t.Size())
	}
	if err := f.Truncate(header + int64(end)); err != nil {
		return err
	}
	// A filesystem that does not keep holes allocates the whole length. Refuse rather than fill
	// the disk: the profile records the reason as a failure.
	if used, ok := allocatedBytes(path); ok && used > sparseLimit {
		f.Close()
		os.Remove(path)
		return fmt.Errorf("synthetic model: the filesystem at %s does not keep holes (%d bytes allocated for a %d-byte file), so it cannot be used", filepath.Dir(path), used, header+int64(end))
	}
	return nil
}

// sparseLimit is more than a synthetic model's header ever needs: the real ones take 20-50 KiB.
const sparseLimit = 64 << 20

// allocatedBytes is AllocatedBytes, replaceable in tests: no filesystem here lacks holes.
var allocatedBytes = AllocatedBytes
