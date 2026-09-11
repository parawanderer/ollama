package server

import (
	"encoding/json"
	"runtime"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/ml"
)

func TestRunnerPIDsNamesModelsAsPsDoes(t *testing.T) {
	s := &Scheduler{loaded: map[string]*runnerRef{
		"a": {llama: &mockLlm{pid: 960}, name: "registry.ollama.ai/library/granite4.1:3b"},
		"b": {llama: &mockLlm{}, name: "registry.ollama.ai/library/qwen3:32b"}, // no process
	}}
	got := s.runnerPIDs()
	if len(got) != 1 || got[960] != "granite4.1:3b" {
		t.Fatalf("runnerPIDs = %v, want {960: granite4.1:3b}", got)
	}
}

func TestGPUProcessesMarksRunnersAndHelpers(t *testing.T) {
	procs := []ml.DeviceProcess{
		{PID: 960, UsedMemory: 14298382336, Name: "llama-server", OllamaChild: true},
		{PID: 1010, UsedMemory: 600 << 20, Name: "llama-server", OllamaChild: true}, // a fit probe
		{PID: 7, UsedMemory: 2 << 30, Name: "python3"},                              // someone else's
	}
	got := gpuProcesses(procs, map[int]string{960: "granite4.1:3b"})

	if got[0].Runner == nil || got[0].Runner.Model != "granite4.1:3b" || got[0].OllamaHelper {
		t.Errorf("runner: %+v", got[0])
	}
	if got[1].Runner != nil || !got[1].OllamaHelper {
		t.Errorf("ollama's own non-runner child must be a helper, not a tenant: %+v", got[1])
	}
	if got[2].Runner != nil || got[2].OllamaHelper || got[2].Name != "python3" {
		t.Errorf("another tenant must carry neither mark: %+v", got[2])
	}

	b, _ := json.Marshal(got[2])
	if strings.Contains(string(b), "runner") || strings.Contains(string(b), "ollama_helper") {
		t.Errorf("absent marks must be absent on the wire: %s", b)
	}
	b, _ = json.Marshal(api.GPUProcess{PID: 1, Runner: &api.GPUProcessRunner{Model: "m"}})
	if !strings.Contains(string(b), `"runner":{"model":"m"}`) {
		t.Errorf("runner shape: %s", b)
	}
}

// The scope travels on every device, whatever the process list holds, because it is what
// says whether an empty list means "nothing else here" or "nothing else visible".
func TestInfoHandlerReportsProcessesScope(t *testing.T) {
	gpu := infoResponse(t, infoTestServer(t)).ComputeInfo.SupportedGPUs[0]
	if want := discover.ProcessesScope(); gpu.ProcessesScope != want || (runtime.GOOS == "linux" && want == "") {
		t.Errorf("processes_scope = %q, want %q (and known on linux)", gpu.ProcessesScope, want)
	}
}
