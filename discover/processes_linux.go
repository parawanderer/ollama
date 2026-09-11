//go:build linux

package discover

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const procRoot = "/proc"

// initPIDNamespace is how the kernel names the initial PID namespace: its inode is fixed
// (PROC_PID_INIT_INO, 0xEFFFFFFC), so a process can tell whether it is inside a child
// namespace without guessing from container markers.
const initPIDNamespace = "pid:[4026531836]"

// ProcessesScope says which processes ComputeProcesses can see.
//
// "all" when ollama runs in the host's PID namespace: every process holding memory on a
// device is listed. "pid_namespace" when it runs in a child namespace, as in a container:
// the driver lists only processes in that same namespace, so another container's or the
// host's process on the card is not listed at all -- its memory still counts as used, and
// a caller must not read an empty list as "nothing else is on this card". Measured on this
// box with a torch process holding 2.6 GB on a card: the ollama container's NVML listed
// nothing for that card. Empty when it cannot be determined.
func ProcessesScope() string {
	return processesScopeAt(procRoot)
}

func processesScopeAt(root string) string {
	ns, err := os.Readlink(filepath.Join(root, "self", "ns", "pid"))
	if err != nil {
		return ""
	}
	if ns == initPIDNamespace {
		return "all"
	}
	return "pid_namespace"
}

// describeProcess reads a process's name and whether this ollama started it. Either is
// left empty when /proc cannot answer -- a process that exited, or one in a namespace this
// one cannot see.
func describeProcess(root string, pid, self int) (name string, ollamaChild bool) {
	dir := filepath.Join(root, strconv.Itoa(pid))
	if b, err := os.ReadFile(filepath.Join(dir, "comm")); err == nil {
		name = strings.TrimSpace(string(b))
	}
	f, err := os.Open(filepath.Join(dir, "status"))
	if err != nil {
		return name, false
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		if v, ok := strings.CutPrefix(s.Text(), "PPid:"); ok {
			ppid, err := strconv.Atoi(strings.TrimSpace(v))
			return name, err == nil && ppid == self
		}
	}
	return name, false
}
