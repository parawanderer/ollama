//go:build !linux

package llm

import "os/exec"

func dieWithParent(cmd *exec.Cmd) (unlock func()) {
	cmd.SysProcAttr = LlamaServerSysProcAttr
	return func() {}
}
