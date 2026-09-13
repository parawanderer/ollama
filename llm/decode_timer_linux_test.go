package llm

import (
	"bufio"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// If ollama is killed while the profile is timing a synthetic model, its llama-server must not
// outlive it holding GPU memory. A helper process stands in for ollama: it starts a child the
// way TimeDecode does and is then killed outright.
func TestTimedServerDiesWithOllama(t *testing.T) {
	if os.Getenv("SLOP_DIE_WITH_PARENT_HELPER") == "1" {
		cmd := exec.Command("sleep", "60")
		unlock := dieWithParent(cmd)
		defer unlock()
		if err := cmd.Start(); err != nil {
			os.Exit(2)
		}
		os.Stdout.WriteString(strconv.Itoa(cmd.Process.Pid) + "\n")
		time.Sleep(time.Minute)
		return
	}

	helper := exec.Command(os.Args[0], "-test.run=^TestTimedServerDiesWithOllama$")
	helper.Env = append(os.Environ(), "SLOP_DIE_WITH_PARENT_HELPER=1")
	out, _ := helper.StdoutPipe()
	if err := helper.Start(); err != nil {
		t.Fatal(err)
	}
	line, err := bufio.NewReader(out).ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	child, _ := strconv.Atoi(strings.TrimSpace(line))
	defer syscall.Kill(child, syscall.SIGKILL) //nolint:errcheck // in case the test fails

	_ = helper.Process.Kill() // SIGKILL: no deferred cleanup runs in the helper
	_ = helper.Wait()
	deadline := time.Now().Add(3 * time.Second)
	for running(child) {
		if time.Now().After(deadline) {
			t.Fatal("the timed llama-server outlived the ollama that started it")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// running reports whether pid is a live process. A killed child that nothing has reaped yet is
// a zombie: it still has a pid, but it has stopped.
func running(pid int) bool {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return false
	}
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	return i >= 0 && i+2 < len(s) && s[i+2] != 'Z'
}
