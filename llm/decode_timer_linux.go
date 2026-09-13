package llm

import (
	"os/exec"
	"runtime"
	"syscall"
)

// dieWithParent makes the timed llama-server receive SIGKILL if ollama dies, so a server killed
// mid-measurement does not leave one holding GPU memory. Linux sends the signal when the
// *thread* that started the child exits, not the process, so the calling goroutine is locked to
// its thread until the returned function runs, which must be after the child is reaped.
func dieWithParent(cmd *exec.Cmd) (unlock func()) {
	runtime.LockOSThread()
	attr := *LlamaServerSysProcAttr
	attr.Pdeathsig = syscall.SIGKILL
	cmd.SysProcAttr = &attr
	return runtime.UnlockOSThread
}
