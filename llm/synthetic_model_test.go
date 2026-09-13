package llm

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

// The file must be a GGUF that describes the whole model, with a data region that is a hole:
// that is what lets the profile time a model of many GiB without writing it.
func TestSyntheticModelIsASparseGGUF(t *testing.T) {
	m := SyntheticModel{Embedding: 2048, Layers: 8}
	path := filepath.Join(t.TempDir(), "synthetic.gguf")
	if err := m.Write(path); err != nil {
		t.Fatal(err)
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	g, err := ggml.Decode(f, -1)
	if err != nil {
		t.Fatal(err)
	}
	if got := g.KV().BlockCount(); got != 8 {
		t.Errorf("block_count = %d, want 8", got)
	}
	items := g.Tensors().Items()
	if len(items) != 3+9*8 {
		t.Fatalf("%d tensors, want %d", len(items), 3+9*8)
	}

	var total, read uint64
	for _, ti := range items {
		total += ti.Size()
		if ti.Name != "token_embd.weight" {
			read += ti.Size()
		}
	}
	if read != m.ReadPerToken() {
		t.Errorf("ReadPerToken = %d, the file's tensors say %d", m.ReadPerToken(), read)
	}

	st, _ := f.Stat()
	if want := int64(g.Tensors().Offset) + int64(total); st.Size() < want {
		t.Errorf("file is %d bytes, the tensors need %d: llama.cpp would reject it as truncated", st.Size(), want)
	}
	if used, ok := AllocatedBytes(path); ok && used > 1<<20 {
		t.Errorf("file occupies %d bytes of disk for %d of data: the data was written, not left a hole", used, total)
	}
}

// On a filesystem that allocates the whole length (no holes), writing must fail and leave
// nothing behind rather than fill the disk with a model of many GiB.
func TestSyntheticModelRefusesAFilesystemWithoutHoles(t *testing.T) {
	defer func(f func(string) (int64, bool)) { allocatedBytes = f }(allocatedBytes)
	allocatedBytes = func(string) (int64, bool) { return 11 << 30, true }

	path := filepath.Join(t.TempDir(), "synthetic.gguf")
	if err := (SyntheticModel{Embedding: 2048, Layers: 8}).Write(path); err == nil {
		t.Fatal("wrote a model the filesystem allocated in full")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error("the refused model was left on disk")
	}
}
