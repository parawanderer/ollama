package discover

import (
	"os"
	"path/filepath"
	"testing"
)

func writeCgroup(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

const gib = uint64(1 << 30)

// Transcribed from this box's ollama container on 2026-09-12: no limit, and memory.current
// almost entirely file cache from reading model blobs. The host had 114 GiB available.
func TestUnlimitedCgroupKeepsHostAvailableMemory(t *testing.T) {
	root := writeCgroup(t, map[string]string{
		"memory.max":     "max\n",
		"memory.current": "111296495616\n",
		"memory.stat":    "anon 272805888\nfile 111023689728\ninactive_file 65896132608\n",
	})
	host := memInfo{TotalMemory: 121 * gib, FreeMemory: 114 * gib}
	got := getCPUMemByCgroupsAt(root, host)
	if got.FreeMemory != host.FreeMemory || got.TotalMemory != host.TotalMemory {
		t.Fatalf("free %d GiB total %d GiB, want the host's 114/121: page cache is not usage",
			got.FreeMemory/gib, got.TotalMemory/gib)
	}
}

func TestLimitedCgroupDoesNotCountFileCache(t *testing.T) {
	root := writeCgroup(t, map[string]string{
		"memory.max":     "68719476736\n", // 64 GiB
		"memory.current": "53687091200\n", // 50 GiB, 40 of it file cache
		"memory.stat":    "anon 10737418240\nfile 42949672960\n",
	})
	got := getCPUMemByCgroupsAt(root, memInfo{TotalMemory: 121 * gib, FreeMemory: 114 * gib})
	if got.TotalMemory != 64*gib || got.FreeMemory != 54*gib {
		t.Fatalf("free %d GiB total %d GiB, want 54 of 64", got.FreeMemory/gib, got.TotalMemory/gib)
	}
}

func TestNoCgroupKeepsHostMemory(t *testing.T) {
	host := memInfo{TotalMemory: 8 * gib, FreeMemory: 5 * gib}
	if got := getCPUMemByCgroupsAt(t.TempDir(), host); got != host {
		t.Fatalf("got %+v, want the host figures unchanged", got)
	}
}
