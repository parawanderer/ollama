package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

// A lease gives GPUs to a job outside ollama (training, vLLM, a simulation) for exactly as
// long as the job runs. ollama unloads what it has on those devices, places nothing there
// while the lease is held, and goes back to using them when it ends.
//
// A lease is held by an open HTTP connection, not by a process id: ollama usually runs in a
// container and cannot see the job's pid. `gpu-run` opens the connection, waits for "granted",
// runs the job and closes the connection when the job exits -- and if the wrapper is killed,
// the connection closes anyway. So a lease cannot outlive its job.
//
// Freeing memory at the start is only half of it. The dangerous direction is re-taking memory
// mid-job: a request arriving an hour into a training run would otherwise load a model into
// memory the trainer's allocator was about to grow into, and the run fails out of memory.
type lease struct {
	id      uint64
	holder  string
	devices map[ml.DeviceID]string // device -> PCI address
	since   time.Time
}

type leaseTable struct {
	mu     sync.Mutex
	seq    uint64
	leases map[uint64]*lease
}

func (t *leaseTable) add(holder string, devices map[ml.DeviceID]string) *lease {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.leases == nil {
		t.leases = map[uint64]*lease{}
	}
	t.seq++
	l := &lease{id: t.seq, holder: holder, devices: devices, since: time.Now()}
	t.leases[l.id] = l
	return l
}

func (t *leaseTable) remove(l *lease) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.leases, l.id)
}

// holders maps each leased device to who holds it.
func (t *leaseTable) holders() map[ml.DeviceID]string {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := map[ml.DeviceID]string{}
	for _, l := range t.leases {
		for d := range l.devices {
			out[d] = l.holder
		}
	}
	return out
}

// withoutLeased drops leased devices from a list about to be placed on.
func (s *Scheduler) withoutLeased(gpus []ml.DeviceInfo) []ml.DeviceInfo {
	held := s.leases.holders()
	if len(held) == 0 {
		return gpus
	}
	out := make([]ml.DeviceInfo, 0, len(gpus))
	for _, g := range gpus {
		if _, leased := held[g.DeviceID]; !leased {
			out = append(out, g)
		}
	}
	return out
}

// onLeasedDevice reports whether a loaded runner occupies a leased device. Such a runner is
// never reused: a request for its model reloads it elsewhere.
func (s *Scheduler) onLeasedDevice(r *runnerRef) bool {
	held := s.leases.holders()
	if len(held) == 0 {
		return false
	}
	for _, d := range r.gpus {
		if _, leased := held[d]; leased {
			return true
		}
	}
	return false
}

// vacate unloads every runner on the lease's devices: idle ones now, busy ones as soon as
// their current requests finish (a generation is never cut off). It returns the names of the
// runners still holding a leased device.
func (s *Scheduler) vacate(l *lease) []string {
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()

	var holding []string
	for _, r := range runners {
		if !slices.ContainsFunc(r.gpus, func(d ml.DeviceID) bool { _, ok := l.devices[d]; return ok }) {
			continue
		}
		holding = append(holding, r.name)
		r.refMu.Lock()
		if r.expireTimer != nil {
			r.expireTimer.Stop()
			r.expireTimer = nil
		}
		r.sessionDuration = 0
		if r.refCount == 0 && !r.leaseExpiring {
			r.leaseExpiring = true
			s.expiredCh <- r
		}
		r.refMu.Unlock()
	}
	return holding
}

// resolveLeaseDevices turns what a client named into devices: PCI addresses in either common
// spelling ("0000:03:00.0" or nvidia-smi's "00000000:03:00.0"), or "all".
func resolveLeaseDevices(requested []string, devices []ml.DeviceInfo) (map[ml.DeviceID]string, error) {
	out := map[ml.DeviceID]string{}
	if len(requested) == 0 || slices.Contains(requested, "all") {
		for _, d := range devices {
			out[d.DeviceID] = d.PCIID
		}
		return out, nil
	}
	for _, want := range requested {
		norm := normalizePCI(want)
		found := false
		for _, d := range devices {
			if normalizePCI(d.PCIID) == norm {
				out[d.DeviceID] = d.PCIID
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("no usable GPU at %q", want)
		}
	}
	return out, nil
}

// normalizePCI writes a PCI address as domain:bus:device.function with a four-digit domain.
func normalizePCI(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	parts := strings.Split(s, ":")
	if len(parts) != 3 {
		return s
	}
	domain, err := strconv.ParseUint(parts[0], 16, 32)
	if err != nil {
		return s
	}
	return fmt.Sprintf("%04x:%s:%s", domain, parts[1], parts[2])
}

// errAllGPUsLeased fails a request that needs a GPU while every GPU is lent to a job.
var errAllGPUsLeased = errors.New("every GPU is leased to a job running outside ollama (gpu-run); try again when it finishes, or request the CPU with num_gpu 0")

// LeaseRequest asks for GPUs to be given to a job.
type LeaseRequest struct {
	// Devices are PCI addresses, or "all" (the default).
	Devices []string `json:"devices"`
	// Holder names the job, for /api/info and the event stream.
	Holder string `json:"holder"`
}

// LeaseHandler holds a lease for as long as the request stays open. It streams NDJSON:
// {"status":"waiting","models":[...]} while models on those devices finish, then
// {"status":"granted","devices":[...]}, then {"status":"held"} every 15 s until the client
// disconnects, which ends the lease.
func (s *Server) LeaseHandler(c *gin.Context) {
	var req LeaseRequest
	if err := c.ShouldBindJSON(&req); err != nil && !errors.Is(err, io.EOF) {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	holder := strings.TrimSpace(req.Holder)
	if holder == "" {
		holder = "gpu-run"
	}
	if len(holder) > 64 {
		holder = holder[:64]
	}
	devices, err := resolveLeaseDevices(req.Devices, s.sched.cachedDevices(c.Request.Context()))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if len(devices) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "no GPUs to lease"})
		return
	}

	l := s.sched.leases.add(holder, devices)
	pcis := make([]string, 0, len(devices))
	for _, p := range devices {
		pcis = append(pcis, p)
	}
	slices.Sort(pcis)
	info := &api.LeaseInfo{Holder: holder, Devices: pcis}
	s.sched.publishEvent(api.ModelEvent{Type: EventLeaseStart, Lease: info})
	s.sched.invalidateDeviceCache()
	defer func() {
		s.sched.leases.remove(l)
		s.sched.publishEvent(api.ModelEvent{Type: EventLeaseEnd, Lease: info})
		s.sched.invalidateDeviceCache()
		slog.Info("GPU lease ended", "holder", holder, "devices", pcis, "held", time.Since(l.since).Round(time.Second))
	}()

	c.Header("Content-Type", "application/x-ndjson")
	c.Status(http.StatusOK)
	enc := json.NewEncoder(c.Writer)
	send := func(v any) bool {
		if err := enc.Encode(v); err != nil {
			return false
		}
		c.Writer.Flush()
		return true
	}

	ctx := c.Request.Context()
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	lastWaiting := ""
	for {
		holding := s.sched.vacate(l)
		if len(holding) == 0 {
			break
		}
		if key := strings.Join(holding, ","); key != lastWaiting {
			lastWaiting = key
			if !send(gin.H{"status": "waiting", "models": holding}) {
				return
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
	slog.Info("GPU lease granted", "holder", holder, "devices", pcis)
	s.sched.publishEvent(api.ModelEvent{Type: EventLeaseGranted, Lease: info})
	if !send(gin.H{"status": "granted", "devices": pcis}) {
		return
	}
	held := time.NewTicker(15 * time.Second)
	defer held.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-held.C:
			if !send(gin.H{"status": "held"}) {
				return
			}
		}
	}
}
