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
	finishing := &runnerRef{llama: &mockLlm{pid: 970}, name: "registry.ollama.ai/library/gemma4:31b"}
	finishing.stillLoading.Store(true)
	s := &Scheduler{loaded: map[string]*runnerRef{
		"a": {llama: &mockLlm{pid: 960}, name: "registry.ollama.ai/library/granite4.1:3b"},
		"b": {llama: &mockLlm{}, name: "registry.ollama.ai/library/qwen3:32b"}, // no process
		"c": finishing,
	}}
	got := s.runnerPIDs()
	want := map[int]runnerMark{960: {model: "granite4.1:3b"}, 970: {model: "gemma4:31b", loading: true}}
	if len(got) != len(want) || got[960] != want[960] || got[970] != want[970] {
		t.Fatalf("runnerPIDs = %v, want %v", got, want)
	}
}

// A runner exists, and holds memory, for the whole of its load before it joins s.loaded.
func TestRunnerPIDsIncludesTheRunnerBeingLoaded(t *testing.T) {
	s := &Scheduler{loaded: map[string]*runnerRef{}}
	s.setLoadingModel("registry.ollama.ai/library/qwen3.5:0.8b")
	s.loadingPID.Store(384)
	if got := s.runnerPIDs(); got[384] != (runnerMark{model: "qwen3.5:0.8b", loading: true}) {
		t.Fatalf("runnerPIDs = %v, want 384 marked as qwen3.5:0.8b, loading", got)
	}
	s.clearLoadingModel()
	if got := s.runnerPIDs(); len(got) != 0 {
		t.Fatalf("after the load ends the pid must not linger: %v", got)
	}
	// The next load names its model at load.start, before its process exists. A pid left
	// over from the last load would be pinned on the new model in that window.
	s.setLoadingModel("registry.ollama.ai/library/gemma4:31b")
	if got := s.runnerPIDs(); len(got) != 0 {
		t.Fatalf("the previous load's pid was named as the next model: %v", got)
	}
}

func TestGPUProcessesMarksRunnersAndHelpers(t *testing.T) {
	procs := []ml.DeviceProcess{
		{PID: 960, UsedMemory: 14298382336, Name: "llama-server", OllamaChild: true},
		{PID: 1010, UsedMemory: 600 << 20, Name: "llama-server", OllamaChild: true}, // a fit probe
		{PID: 7, UsedMemory: 2 << 30, Name: "python3"},                              // someone else's
	}
	got := gpuProcesses(procs, map[int]runnerMark{960: {model: "granite4.1:3b"}})

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
