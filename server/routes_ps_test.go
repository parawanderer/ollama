package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func psResponse(t *testing.T, runner *runnerRef) (api.ProcessResponse, string) {
	t.Helper()

	s := &Server{sched: &Scheduler{loaded: map[string]*runnerRef{"m": runner}}}

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/api/ps", nil)

	s.PsHandler(c)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var got api.ProcessResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("decoding: %v (body %q)", err, w.Body.String())
	}
	return got, w.Body.String()
}

// A model split across two cards names both, so a client can attribute residency
// per device by joining these ids against /api/info's supported_gpus.
func TestPsHandlerReportsDevicesForSplitModel(t *testing.T) {
	dev0 := ml.DeviceID{ID: "0", Library: "CUDA"}
	dev1 := ml.DeviceID{ID: "1", Library: "CUDA"}

	got, _ := psResponse(t, &runnerRef{
		model:     &Model{ShortName: "llama3:70b"},
		gpus:      []ml.DeviceID{dev0, dev1},
		totalSize: 42 * format.GigaByte,
		vramSize:  40 * format.GigaByte,
		expiresAt: time.Now().Add(5 * time.Minute),
		llama: &fakeRunner{vram: map[ml.DeviceID]uint64{
			dev0: 25 * format.GigaByte,
			dev1: 15 * format.GigaByte,
		}, total: 42 * format.GigaByte, gpuTotal: 40 * format.GigaByte},
	})

	if len(got.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(got.Models))
	}

	gpus := got.Models[0].GPUs
	if len(gpus) != 2 {
		t.Fatalf("expected both devices, got %d: %+v", len(gpus), gpus)
	}
	for i, want := range []string{"0", "1"} {
		if gpus[i].ID != want {
			t.Errorf("gpu %d: id %q, want %q", i, gpus[i].ID, want)
		}
		if gpus[i].Runner != "CUDA" {
			t.Errorf("gpu %d: runner %q, want CUDA", i, gpus[i].Runner)
		}
	}

	// an uneven split is reported as it is, not divided evenly
	if gpus[0].SizeVRAM != int64(25*format.GigaByte) {
		t.Errorf("gpu 0 size_vram: got %d, want %d", gpus[0].SizeVRAM, 25*format.GigaByte)
	}
	if gpus[1].SizeVRAM != int64(15*format.GigaByte) {
		t.Errorf("gpu 1 size_vram: got %d, want %d", gpus[1].SizeVRAM, 15*format.GigaByte)
	}

	// the model-level figure remains the total across devices
	if got.Models[0].SizeVRAM != int64(40*format.GigaByte) {
		t.Errorf("size_vram: got %d, want %d", got.Models[0].SizeVRAM, 40*format.GigaByte)
	}
}

// A CPU-resident model has no devices, and the field is omitted rather than
// serialised as null.
func TestPsHandlerOmitsDevicesOnCPU(t *testing.T) {
	got, raw := psResponse(t, &runnerRef{
		model:     &Model{ShortName: "smol:1b"},
		totalSize: 2 * format.GigaByte,
		expiresAt: time.Now().Add(5 * time.Minute),
	})

	if len(got.Models[0].GPUs) != 0 {
		t.Errorf("expected no devices, got %+v", got.Models[0].GPUs)
	}
	if strings.Contains(raw, "\"gpus\"") {
		t.Errorf("gpus should be omitted for a CPU-resident model, got %s", raw)
	}
}

// fakeRunner is the narrowest llm.LlamaServer that PsHandler exercises: the
// memory accessors. Everything else panics if the handler ever grows a call.
type fakeRunner struct {
	llm.LlamaServer
	vram          map[ml.DeviceID]uint64
	total         uint64
	gpuTotal      uint64
	memVRAM       api.MemoryBreakdown
	memHost       api.MemoryBreakdown
	memByGPU      map[ml.DeviceID]api.MemoryBreakdown
	weightsOnDisk int64
	placement     *api.ModelPlacement
	activity      *api.RunnerActivity
}

func (f *fakeRunner) MemorySize() (uint64, uint64)    { return f.total, f.gpuTotal }
func (f *fakeRunner) VRAMByGPU(id ml.DeviceID) uint64 { return f.vram[id] }
func (f *fakeRunner) ContextLength() int              { return 4096 }

func (f *fakeRunner) MemoryBreakdownTotals() (vram, host api.MemoryBreakdown) {
	return f.memVRAM, f.memHost
}

func (f *fakeRunner) MemoryBreakdownByGPU(id ml.DeviceID) api.MemoryBreakdown {
	return f.memByGPU[id]
}

func (f *fakeRunner) WeightsOnDisk() int64 { return f.weightsOnDisk }

func (f *fakeRunner) LayerPlacement() *api.ModelPlacement { return f.placement }

func (f *fakeRunner) Activity(ctx context.Context, busy bool) *api.RunnerActivity { return f.activity }

// The split is reported per device and in aggregate, and both sum to the size_vram they
// sit beside. That property is the whole reason a client can trust the breakdown: a UI
// drawing weights-vs-cache as parts of a bar needs the parts to fill it.
func TestPsHandlerReportsMemoryBreakdown(t *testing.T) {
	dev0 := ml.DeviceID{ID: "0", Library: "CUDA"}
	dev1 := ml.DeviceID{ID: "1", Library: "CUDA"}

	perGPU := map[ml.DeviceID]api.MemoryBreakdown{
		dev0: {Weights: 20 * format.GigaByte, KVCache: 4 * format.GigaByte, Compute: format.GigaByte},
		dev1: {Weights: 12 * format.GigaByte, KVCache: 2 * format.GigaByte, Compute: format.GigaByte},
	}
	total := api.MemoryBreakdown{
		Weights: 32 * format.GigaByte, KVCache: 6 * format.GigaByte, Compute: 2 * format.GigaByte,
	}

	got, body := psResponse(t, &runnerRef{
		model:     &Model{ShortName: "llama3:70b"},
		gpus:      []ml.DeviceID{dev0, dev1},
		totalSize: 40 * format.GigaByte,
		vramSize:  40 * format.GigaByte,
		expiresAt: time.Now().Add(5 * time.Minute),
		llama: &fakeRunner{
			vram: map[ml.DeviceID]uint64{
				dev0: 25 * format.GigaByte,
				dev1: 15 * format.GigaByte,
			},
			total: 40 * format.GigaByte, gpuTotal: 40 * format.GigaByte,
			memVRAM: total, memByGPU: perGPU,
			weightsOnDisk: 38 * format.GigaByte,
		},
	})

	if len(got.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(got.Models))
	}
	m := got.Models[0]

	if m.Memory == nil {
		t.Fatalf("no aggregate breakdown on the model row; body %s", body)
	}
	if m.Memory.Total() != m.SizeVRAM {
		t.Errorf("aggregate breakdown totals %d but size_vram is %d", m.Memory.Total(), m.SizeVRAM)
	}
	if m.WeightsOnDisk != 38*format.GigaByte {
		t.Errorf("weights_on_disk = %d, want %d", m.WeightsOnDisk, 38*format.GigaByte)
	}

	for _, g := range m.GPUs {
		if g.Memory == nil {
			t.Fatalf("device %s carries no breakdown; body %s", g.ID, body)
		}
		if g.Memory.Total() != g.SizeVRAM {
			t.Errorf("device %s: breakdown totals %d but size_vram is %d",
				g.ID, g.Memory.Total(), g.SizeVRAM)
		}
	}

	// The field has to reach the wire under the name a client reads, not merely exist on
	// the struct: a breakdown that marshals to nothing is invisible however correct it is.
	for _, want := range []string{`"memory"`, `"weights"`, `"kv_cache"`, `"weights_on_disk"`} {
		if !strings.Contains(body, want) {
			t.Errorf("%s missing from the response body: %s", want, body)
		}
	}
}
