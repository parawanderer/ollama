package server

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// fmtSscanShape reads a shape back from the synthetic model's file name.
func fmtSscanShape(name string, s *profileShape) (int, error) {
	return fmt.Sscanf(name, "synthetic-%dx%d.gguf", &s.width, &s.layers)
}

func profileGPU(id, pci string) ml.DeviceInfo {
	return ml.DeviceInfo{
		DeviceID:    ml.DeviceID{ID: id, Library: "CUDA"},
		PCIID:       pci,
		Description: "test card",
		TotalMemory: 96 * format.GigaByte,
		FreeMemory:  95 * format.GigaByte,
	}
}

// A machine that behaves exactly as the model says: a + c·L + bytes/bw on one device, and
// under tensor split only the bandwidth part divides, plus two reductions a layer.
type fakeMachine struct {
	a, c, bw, ell float64 // ms, ms per layer, bytes per ms, ms per reduction
	tensorErr     error
}

func (m fakeMachine) time(_ context.Context, gpus []ml.DeviceInfo, path string, tensor bool) (float64, error) {
	var s profileShape
	if _, err := fmtSscanShape(filepath.Base(path), &s); err != nil {
		return 0, err
	}
	one := m.a + m.c*float64(s.layers) + float64(modelBytes(s))/m.bw
	if !tensor {
		return one, nil
	}
	if m.tensorErr != nil {
		return 0, m.tensorErr
	}
	n := float64(len(gpus))
	return one - float64(modelBytes(s))/m.bw*(1-1/n) + 2*float64(s.layers)*m.ell, nil
}

func newTestProfiler(t *testing.T, m fakeMachine) (*boxProfiler, *time.Time) {
	p := newBoxProfiler(filepath.Join(t.TempDir(), "box-profile.json"))
	p.timeFn = m.time
	p.engineFn = func() string { return "engine-1" }
	now := time.Unix(1_800_000_000, 0)
	p.now = func() time.Time { return now }
	p.lastBusy = now.Add(-time.Hour)
	return p, &now
}

// The fit and the link cost come back as the numbers the machine was built from, which is what
// makes the profile's figures mean what api.BoxProfile says they mean.
func TestProfileRecoversTheMachine(t *testing.T) {
	m := fakeMachine{a: 0.067, c: 0.0255, bw: 1.611e9, ell: 0.011}
	p, _ := newTestProfiler(t, m)
	gpus := []ml.DeviceInfo{profileGPU("0", "0000:01:00.0"), profileGPU("1", "0000:03:00.0"), profileGPU("2", "0000:05:00.0")}

	if !p.tick(t.Context(), func() ([]ml.DeviceInfo, string) { return gpus, "" }) {
		t.Fatal("a quiet machine with no profile was not measured")
	}
	r := p.report(gpus)
	if r.State != "measured" || len(r.Devices) != 3 || len(r.Failures) != 0 {
		t.Fatalf("report = %+v", r)
	}
	for _, d := range r.Devices {
		if d.ID == "" {
			t.Errorf("device %s reported without its id", d.PCIID)
		}
		if math.Abs(float64(d.BandwidthBytesPerSec)/1.611e12-1) > 1e-6 ||
			math.Abs(d.TokenOverheadMs-0.067) > 1e-3 || math.Abs(d.LayerOverheadUs-25.5) > 1e-3 {
			t.Errorf("device %s = %+v, want 1.611 TB/s, 0.067 ms, 25.5 µs", d.PCIID, d)
		}
	}
	// Three devices: every pair, and all three together.
	if len(r.Links) != 4 {
		t.Fatalf("%d links, want 4: %+v", len(r.Links), r.Links)
	}
	for _, l := range r.Links {
		if len(l.Reductions) != 2 || l.Reductions[0].Width != 4096 || l.Reductions[1].Width != 8192 {
			t.Fatalf("link %v reductions = %+v", l.PCIIDs, l.Reductions)
		}
		for _, red := range l.Reductions {
			if math.Abs(red.Us-11) > 1e-3 {
				t.Errorf("link %v at width %d = %.3f µs, want 11", l.PCIIDs, red.Width, red.Us)
			}
		}
	}
}

// A backend without tensor split still gets its devices measured, and says why the links are
// missing rather than leaving them out silently.
func TestProfileRecordsAFailedTensorSplit(t *testing.T) {
	m := fakeMachine{a: 0.1, c: 0.02, bw: 1e9, tensorErr: errors.New("split mode tensor not supported")}
	p, _ := newTestProfiler(t, m)
	gpus := []ml.DeviceInfo{profileGPU("0", "0000:01:00.0"), profileGPU("1", "0000:03:00.0")}
	if !p.tick(t.Context(), func() ([]ml.DeviceInfo, string) { return gpus, "" }) {
		t.Fatal("not measured")
	}
	r := p.report(gpus)
	if len(r.Devices) != 2 || len(r.Links) != 0 || len(r.Failures) != 1 || r.Failures[0].What != "tensor_split" {
		t.Fatalf("report = %+v", r)
	}
	// A failure is a result: the machine is not measured again on the next quiet tick.
	if p.tick(t.Context(), func() ([]ml.DeviceInfo, string) { return gpus, "" }) {
		t.Error("a profile with a recorded failure was measured again")
	}
}

func TestProfileWaitsForQuiet(t *testing.T) {
	m := fakeMachine{a: 0.1, c: 0.02, bw: 1e9}
	gpus := []ml.DeviceInfo{profileGPU("0", "0000:01:00.0")}

	p, now := newTestProfiler(t, m)
	if p.tick(t.Context(), func() ([]ml.DeviceInfo, string) { return gpus, "a model is loaded" }) {
		t.Error("measured while the scheduler said the GPUs were busy")
	}
	p.lastBusy = now.Add(-profileIdleFor / 2)
	if p.tick(t.Context(), func() ([]ml.DeviceInfo, string) { return gpus, "" }) {
		t.Error("measured before the server had been quiet for profileIdleFor")
	}
	p.lastBusy = now.Add(-profileIdleFor)
	if !p.tick(t.Context(), func() ([]ml.DeviceInfo, string) { return gpus, "" }) {
		t.Error("not measured once quiet")
	}
}

// preempt stops a measurement and returns only once it has stopped, and nothing half-measured
// is kept.
func TestPreemptStopsAMeasurement(t *testing.T) {
	p, _ := newTestProfiler(t, fakeMachine{})
	entered := make(chan struct{})
	p.timeFn = func(ctx context.Context, _ []ml.DeviceInfo, _ string, _ bool) (float64, error) {
		close(entered)
		<-ctx.Done()
		return 0, ctx.Err()
	}
	gpus := []ml.DeviceInfo{profileGPU("0", "0000:01:00.0")}
	measured := make(chan bool)
	go func() { measured <- p.tick(t.Context(), func() ([]ml.DeviceInfo, string) { return gpus, "" }) }()
	<-entered
	if got := p.report(gpus).State; got != "measuring" {
		t.Errorf("state while measuring = %q", got)
	}
	p.preempt()
	select {
	case ok := <-measured:
		if ok {
			t.Error("an interrupted measurement was reported as done")
		}
	default:
		t.Fatal("preempt returned before the measurement stopped")
	}
	if got := p.report(gpus).State; got != "pending" {
		t.Errorf("state after preempt = %q, want pending", got)
	}
}

// A request must not be placed while the profile's llama-server still holds memory: the
// scheduler preempts before it places anything.
func TestSchedulerPreemptsTheProfileBeforeLoading(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 2*time.Second)
	defer done()
	s := InitScheduler(ctx)
	s.getGpuFn = getGpuFn
	s.getSystemInfoFn = getSystemInfoFn

	p, _ := newTestProfiler(t, fakeMachine{})
	var stopped atomic.Bool
	entered := make(chan struct{})
	p.timeFn = func(ctx context.Context, _ []ml.DeviceInfo, _ string, _ bool) (float64, error) {
		close(entered)
		<-ctx.Done()
		stopped.Store(true)
		return 0, ctx.Err()
	}
	s.profiler = p
	gpus := []ml.DeviceInfo{profileGPU("0", "0000:01:00.0")}
	go p.tick(ctx, func() ([]ml.DeviceInfo, string) { return gpus, "" })
	<-entered

	a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: time.Minute}, nil)
	var stoppedAtLoad atomic.Bool
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, config llm.LlamaServerConfig) (llm.LlamaServer, error) {
		stoppedAtLoad.Store(stopped.Load())
		return a.newServer(systemInfo, gpus, model, f, adapters, projectors, opts, numParallel, config)
	}
	s.pendingReqCh <- a.req
	s.Run(ctx)
	select {
	case <-a.req.successCh:
	case err := <-a.req.errCh:
		t.Fatal(err)
	case <-ctx.Done():
		t.Fatal("timeout")
	}
	if !stoppedAtLoad.Load() {
		t.Error("a model was loaded while the profile was still measuring")
	}
}

func TestProfileIdentity(t *testing.T) {
	p, _ := newTestProfiler(t, fakeMachine{})
	a, b := profileGPU("0", "0000:01:00.0"), profileGPU("1", "0000:03:00.0")
	base := p.identity([]ml.DeviceInfo{a, b})
	if p.identity([]ml.DeviceInfo{b, a}) != base {
		t.Error("the order discovery lists the devices in changed the identity")
	}
	renumbered := b
	renumbered.ID = "0"
	if p.identity([]ml.DeviceInfo{renumbered, a}) != base {
		t.Error("a renumbered device changed the identity; ids are not stable across rediscovery")
	}
	driver := b
	driver.DriverMajor = 13
	swapped := b
	swapped.PCIID = "0000:05:00.0"
	for name, set := range map[string][]ml.DeviceInfo{"driver": {a, driver}, "slot": {a, swapped}, "one card": {a}} {
		if p.identity(set) == base {
			t.Errorf("a different %s kept the identity", name)
		}
	}
	q, _ := newTestProfiler(t, fakeMachine{})
	q.engineFn = func() string { return "engine-2" }
	if q.identity([]ml.DeviceInfo{a, b}) == base {
		t.Error("a new engine build kept the identity")
	}
}

// Measured once, kept across restarts, and reported with the ids the devices have now.
func TestProfileSurvivesARestart(t *testing.T) {
	m := fakeMachine{a: 0.1, c: 0.02, bw: 1e9}
	p, _ := newTestProfiler(t, m)
	gpus := []ml.DeviceInfo{profileGPU("0", "0000:01:00.0")}
	if !p.tick(t.Context(), func() ([]ml.DeviceInfo, string) { return gpus, "" }) {
		t.Fatal("not measured")
	}

	q := newBoxProfiler(p.path)
	q.engineFn = p.engineFn
	renumbered := gpus[0]
	renumbered.ID = "7"
	r := q.report([]ml.DeviceInfo{renumbered})
	if r.State != "measured" || len(r.Devices) != 1 || r.Devices[0].ID != "7" {
		t.Fatalf("after restart = %+v", r)
	}
	q.timeFn = func(context.Context, []ml.DeviceInfo, string, bool) (float64, error) {
		t.Fatal("a stored profile was measured again")
		return 0, nil
	}
	q.lastBusy = time.Time{}
	q.tick(t.Context(), func() ([]ml.DeviceInfo, string) { return []ml.DeviceInfo{renumbered}, "" })
}

func TestProfileCandidates(t *testing.T) {
	s := &Scheduler{pendingReqCh: make(chan *LlmRequest, 1), loaded: map[string]*runnerRef{}}
	g := profileGPU("0", "0000:01:00.0")
	free := g.FreeMemory
	s.freeMemoryFn = func([]string) map[string]uint64 { return map[string]uint64{g.PCIID: free} }
	if gpus, _ := s.profileCandidates(); gpus != nil {
		t.Error("candidates offered before discovery ran; the profile must not start discovery itself")
	}
	s.deviceCache = []ml.DeviceInfo{g}
	if _, why := s.profileCandidates(); why != "" {
		t.Fatalf("an idle machine was refused: %s", why)
	}

	free = g.TotalMemory / 2
	if _, why := s.profileCandidates(); why == "" {
		t.Error("measured with half the card held by something else")
	}
	free = g.FreeMemory

	s.loaded["m"] = &runnerRef{}
	if _, why := s.profileCandidates(); why == "" {
		t.Error("measured with a model loaded")
	}
	delete(s.loaded, "m")

	l := s.leases.add("job", map[ml.DeviceID]string{g.DeviceID: g.PCIID})
	if _, why := s.profileCandidates(); why == "" {
		t.Error("measured a leased GPU")
	}
	s.leases.remove(l)
}
