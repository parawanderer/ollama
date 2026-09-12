package server

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

var (
	leaseGPU0 = ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, PCIID: "0000:01:00.0"}
	leaseGPU1 = ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}, PCIID: "0000:03:00.0"}
)

// gpu-run sends what nvidia-smi prints, which pads the PCI domain to eight digits.
func TestResolveLeaseDevices(t *testing.T) {
	devs := []ml.DeviceInfo{leaseGPU0, leaseGPU1}
	got, err := resolveLeaseDevices([]string{"00000000:03:00.0"}, devs)
	if err != nil || len(got) != 1 || got[leaseGPU1.DeviceID] != "0000:03:00.0" {
		t.Fatalf("nvidia-smi spelling: %v %v", got, err)
	}
	if got, _ := resolveLeaseDevices(nil, devs); len(got) != 2 {
		t.Fatalf("no devices named means all, got %v", got)
	}
	if _, err := resolveLeaseDevices([]string{"0000:09:00.0"}, devs); err == nil {
		t.Fatal("an unknown address must be refused, not silently ignored")
	}
}

func TestLeasedDevicesAreNotPlacedOn(t *testing.T) {
	s := &Scheduler{}
	l := s.leases.add("train.py", map[ml.DeviceID]string{leaseGPU1.DeviceID: leaseGPU1.PCIID})
	if got := s.withoutLeased([]ml.DeviceInfo{leaseGPU0, leaseGPU1}); len(got) != 1 || got[0].DeviceID != leaseGPU0.DeviceID {
		t.Fatalf("placement list = %v, want only the unleased card", got)
	}
	if !s.onLeasedDevice(&runnerRef{gpus: []ml.DeviceID{leaseGPU1.DeviceID}}) {
		t.Error("a runner on a leased card must not be reused")
	}
	s.leases.remove(l)
	if got := s.withoutLeased([]ml.DeviceInfo{leaseGPU0, leaseGPU1}); len(got) != 2 {
		t.Fatalf("after release, placement list = %v", got)
	}
}

// Idle runners on leased cards unload now; busy ones finish their request first.
func TestVacateUnloadsIdleAndDrainsBusy(t *testing.T) {
	s := &Scheduler{expiredCh: make(chan *runnerRef, 4), loaded: map[string]*runnerRef{}}
	idle := &runnerRef{name: "idle", gpus: []ml.DeviceID{leaseGPU1.DeviceID}, sessionDuration: time.Hour}
	busy := &runnerRef{name: "busy", gpus: []ml.DeviceID{leaseGPU1.DeviceID}, sessionDuration: time.Hour, refCount: 1}
	other := &runnerRef{name: "other", gpus: []ml.DeviceID{leaseGPU0.DeviceID}, sessionDuration: time.Hour}
	s.loaded["idle"], s.loaded["busy"], s.loaded["other"] = idle, busy, other
	l := s.leases.add("job", map[ml.DeviceID]string{leaseGPU1.DeviceID: leaseGPU1.PCIID})

	holding := s.vacate(l)
	if len(holding) != 2 {
		t.Fatalf("holding = %v, want the two runners on the leased card", holding)
	}
	select {
	case r := <-s.expiredCh:
		if r != idle {
			t.Fatalf("expired %s, want idle", r.name)
		}
	default:
		t.Fatal("the idle runner was not unloaded")
	}
	select {
	case r := <-s.expiredCh:
		t.Fatalf("%s was unloaded mid-request", r.name)
	default:
	}
	if busy.sessionDuration != 0 {
		t.Error("the busy runner must unload as soon as its request finishes")
	}
	if other.sessionDuration != time.Hour {
		t.Error("a runner on another card was touched")
	}
	// Polling again must not ask for the same unload twice.
	s.vacate(l)
	if len(s.expiredCh) != 0 {
		t.Error("vacate asked for the idle runner's unload twice")
	}
}

// With every GPU lent out, a request that needs one fails with the reason instead of
// quietly taking the CPU the job needs.
func TestAllGPUsLeasedFailsFast(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.getGpuFn = func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := leaseGPU0
		g.TotalMemory, g.FreeMemory = 24*format.GigaByte, 12*format.GigaByte
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn
	s.leases.add("job", map[ml.DeviceID]string{leaseGPU0.DeviceID: leaseGPU0.PCIID})
	a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: 5 * time.Millisecond}, nil)
	s.newServerFn = a.newServer
	s.pendingReqCh <- a.req
	s.Run(ctx)
	select {
	case err := <-a.req.errCh:
		if !errors.Is(err, errAllGPUsLeased) {
			t.Fatalf("err = %v, want errAllGPUsLeased", err)
		}
	case <-a.req.successCh:
		t.Fatal("a model was loaded while every GPU was leased")
	case <-ctx.Done():
		t.Fatal("timeout")
	}
}

// The whole lifecycle over a real connection: granted, marked leased, released on disconnect.
func TestLeaseHandlerLifecycle(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ctx, done := context.WithCancel(t.Context())
	defer done()
	s := &Server{sched: InitScheduler(ctx)}
	s.sched.getGpuFn = func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		return []ml.DeviceInfo{leaseGPU0, leaseGPU1}
	}
	r := gin.New()
	r.POST("/api/lease", s.LeaseHandler)
	srv := httptest.NewServer(r)
	defer srv.Close()

	reqCtx, cancel := context.WithCancel(t.Context())
	req, _ := http.NewRequestWithContext(reqCtx, http.MethodPost, srv.URL+"/api/lease",
		strings.NewReader(`{"devices":["00000000:03:00.0"],"holder":"train.py"}`))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	var first map[string]any
	if err := json.Unmarshal([]byte(line), &first); err != nil || first["status"] != "granted" {
		t.Fatalf("first line = %q, want granted (nothing was loaded)", line)
	}
	if held := s.sched.leases.holders(); held[leaseGPU1.DeviceID] != "train.py" || len(held) != 1 {
		t.Fatalf("leases while held = %v", held)
	}

	cancel() // the job ended: gpu-run closes the connection
	deadline := time.Now().Add(2 * time.Second)
	for len(s.sched.leases.holders()) != 0 {
		if time.Now().After(deadline) {
			t.Fatal("the lease outlived its connection")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// A model already loaded on a card that then gets leased must not be reused: the next request
// for it goes back through placement, which no longer offers that card. Each path is shown to
// reuse the runner without a lease first, so the test cannot pass on a reload for another
// reason.
func TestRunnerOnLeasedCardIsNotReused(t *testing.T) {
	for _, path := range []string{"fast", "pending"} {
		t.Run(path, func(t *testing.T) {
			ctx, done := context.WithTimeout(t.Context(), 2*time.Second)
			defer done()
			s := InitScheduler(ctx)
			s.waitForRecovery = 10 * time.Millisecond
			s.getGpuFn = getGpuFn
			s.getSystemInfoFn = getSystemInfoFn
			a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: time.Minute}, nil)
			// As getRunner would set them, so a second identical request matches the runner.
			a.req.opts.NumCtx = 32 // the scenario model is trained at 32; more would be clamped and never match
			a.req.contextShift = resolveContextShift(a.req.shift, a.req.model)
			s.newServerFn = a.newServer
			s.pendingReqCh <- a.req
			s.Run(ctx)
			var runner *runnerRef
			select {
			case runner = <-a.req.successCh:
			case err := <-a.req.errCh:
				t.Fatal(err)
			case <-ctx.Done():
				t.Fatal("timeout")
			}

			ask := func() (chan *runnerRef, chan error) {
				if path == "fast" {
					return s.getRunner(ctx, a.req.model, a.req.opts, &api.Duration{Duration: time.Minute},
						a.req.numCtxAuto, a.req.numBatchAuto, a.req.shift)
				}
				b := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: time.Minute}, nil)
				b.req.model, b.req.opts = a.req.model, a.req.opts
				s.pendingReqCh <- b.req
				return b.req.successCh, b.req.errCh
			}

			// Control: without a lease the same request reuses the loaded runner.
			okCh, errCh := ask()
			select {
			case r := <-okCh:
				if r != runner {
					t.Fatal("control: the runner was not reused without a lease, so this test proves nothing")
				}
			case err := <-errCh:
				t.Fatalf("control: %v", err)
			case <-ctx.Done():
				t.Fatal("control: timeout")
			}

			s.leases.add("job", map[ml.DeviceID]string{runner.gpus[0]: ""})
			okCh, errCh = ask()
			select {
			case r := <-okCh:
				if r == runner {
					t.Fatal("the runner on the leased card was reused")
				}
			case err := <-errCh:
				if !errors.Is(err, errAllGPUsLeased) {
					t.Fatalf("err = %v", err)
				}
			case <-ctx.Done():
				// Waiting for the busy runner to finish before reloading is also correct:
				// the control request above still holds it. What must not happen is reuse.
			}
		})
	}
}
