//go:build linux

package discover

import (
	"os"
	"path/filepath"
	"testing"
)

func TestProcessesScopeReadsThePIDNamespace(t *testing.T) {
	for _, tc := range []struct {
		link, want string
	}{
		{"pid:[4026531836]", "all"},           // the host's namespace, as read on this box
		{"pid:[4026533851]", "pid_namespace"}, // the ollama container's, as read on this box
		{"", ""},                              // unreadable
	} {
		root := t.TempDir()
		if tc.link != "" {
			if err := os.MkdirAll(filepath.Join(root, "self", "ns"), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(tc.link, filepath.Join(root, "self", "ns", "pid")); err != nil {
				t.Fatal(err)
			}
		}
		if got := processesScopeAt(root); got != tc.want {
			t.Errorf("ns %q: scope = %q, want %q", tc.link, got, tc.want)
		}
	}
}

func TestDescribeProcess(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "960")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Transcribed from the runner in the ollama container: comm, and status's PPid line.
	os.WriteFile(filepath.Join(dir, "comm"), []byte("llama-server\n"), 0o644)
	os.WriteFile(filepath.Join(dir, "status"), []byte("Name:\tllama-server\nUmask:\t0022\nState:\tS (sleeping)\nTgid:\t960\nNgid:\t0\nPid:\t960\nPPid:\t1\n"), 0o644)

	if name, child := describeProcess(root, 960, 1); name != "llama-server" || !child {
		t.Errorf("our runner: name=%q child=%v, want llama-server, true", name, child)
	}
	if _, child := describeProcess(root, 960, 2); child {
		t.Error("a process whose parent is not this ollama was called its child")
	}
	if name, child := describeProcess(root, 4242, 1); name != "" || child {
		t.Errorf("a process /proc cannot see: name=%q child=%v, want nothing", name, child)
	}
}
