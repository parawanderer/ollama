package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/mlxrunner"
)

type LlmRequest struct {
	ctx             context.Context //nolint:containedctx
	model           *Model
	opts            api.Options
	sessionDuration *api.Duration
	successCh       chan *runnerRef
	errCh           chan error
	schedAttempts   uint

	// oomRetryAttempted is set after a llama-server load crash triggers an
	// evict-all-and-retry. Prevents infinite retry on persistent load failures.
	oomRetryAttempted bool

	// numCtxAuto is true when NumCtx came from Ollama's automatic VRAM-tier
	// default rather than explicit request, model, or environment config.
	numCtxAuto bool

	// numBatchAuto is true when NumBatch came from Ollama's default options
	// rather than an explicit request or model option.
	numBatchAuto bool

	// useMMapAuto is true when UseMMap was derived by the scheduler rather than
	// explicitly requested.
	useMMapAuto bool

	// contextShift is a llama-server launch attribute resolved from the
	// request-level shift option before scheduling.
	contextShift bool
	shift        *bool
}

type Scheduler struct {
	pendingReqCh  chan *LlmRequest
	finishedReqCh chan *LlmRequest
	expiredCh     chan *runnerRef
	unloadedCh    chan any

	// loadedMu protects loaded and activeLoading
	loadedMu sync.Mutex

	// activeLoading is the model that we are currently working on loading,
	// including by evicting one or more other models. We can only load
	// one model at a time but new requests to models that already loaded can
	// happen in parallel
	activeLoading llm.LlamaServer
	loaded        map[string]*runnerRef

	// vramCalibration remembers what earlier loads actually used, so a repeat load of a
	// model is predicted from measurement instead of from metadata alone.
	vramCalibration *llm.VRAMCalibration

	// loadsInFlight counts loads that have started and not finished. A runner object does
	// not exist for most of a load, so the runner list cannot answer "is anything loading"
	// -- which is exactly the period the sampler most needs to run fast.
	loadsInFlight atomic.Int64

	// loadingModel is the display name of the model whose load is in flight, or empty.
	//
	// It exists because the runner object is built only after the load returns, so /api/ps
	// had nothing to report for the whole of a long load -- measured, an empty list for 64 s
	// while a 142 GB model loaded. A client polling it saw "no models" while its own request
	// hung, which is indistinguishable from a dead server. The scheduler knows the name from
	// the moment it publishes load.start; this is that same knowledge, readable by a poll.
	loadingModel atomic.Pointer[string]

	// loadingPID is the process of the runner being loaded, from the moment it exists until
	// it is resident. The runner joins s.loaded only after its load returns, so without this
	// a process holding gigabytes on a card for the whole load could not be named. An atomic
	// rather than a read of activeLoading, which is written outside loadedMu.
	loadingPID atomic.Int64

	// deviceNames remembers what each device was called while healthy, for naming it after
	// it faults. Nil in tests that do not set it; every method accepts that.
	deviceNames *deviceNames

	// events publishes model lifecycle transitions to /api/events subscribers, and ring
	// retains them so a client that reconnects can be told what it missed rather than
	// having a gap drawn over.
	events *eventBus
	ring   *frameRing

	// samplerWake carries a nudge from anything that moved memory to the sampler, so the
	// first reading after an edge does not wait for the next tick. Buffered by one and
	// written non-blocking; see wakeSampler.
	samplerWake chan struct{}

	// psFn and infoFn build the bodies the stream embeds. They are supplied by the server
	// rather than reached for, so the scheduler does not depend on the HTTP layer and the
	// sampler stays testable.
	psFn   func() *api.ProcessResponse
	infoFn func() *api.InfoResponse

	// deviceCacheMu guards a recent discovery result. Discovery can refresh free memory
	// through runners that already hold a device, but only when the runners it is given
	// account for every device -- so on a host with an idle GPU it always falls back to
	// spawning a probe, which costs hundreds of milliseconds. Reporting endpoints serve a
	// recent result instead of probing per request.
	deviceCacheMu  sync.Mutex
	deviceCache    []ml.DeviceInfo
	deviceCacheAt  time.Time
	deviceCacheTTL time.Duration
	// deviceCacheLiveTTL replaces deviceCacheTTL while free memory is being read live from
	// the driver, which is what the short TTL existed to keep current.
	deviceCacheLiveTTL time.Duration
	liveFreeMemory     atomic.Bool
	// faultRediscoveredAt limits rediscovery for a fault to once per health verdict, so a
	// faulted device that discovery goes on returning cannot make every read rediscover.
	faultRediscoveredAt time.Time
	// unavailableFn reports devices that exist and cannot be used; nil means
	// discover.CachedUnavailableDevices. Swapped only by tests.
	unavailableFn func(known []string) []ml.UnavailableDevice
	// freeMemoryFn reads free memory live from the driver; nil means
	// discover.FreeMemoryByPCI. Swapped only by tests.
	freeMemoryFn func(pciIDs []string) map[string]uint64

	loadFn func(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool
	// db is the server database (server_db.go): usage, calibration, device names and the box
	// profile. nil keeps everything in memory.
	db *serverDB

	// leases are GPUs given to jobs outside ollama (server/lease.go).
	leases leaseTable

	// profiler measures the machine's decode speed once, when it is quiet
	// (server/box_profile.go); nil measures nothing.
	profiler *boxProfiler

	// fitProbe measures a load without performing it; nil means llm.ProbeFitVRAM. Swapped
	// only by tests, which cannot run llama-server.
	fitProbe        func(ctx context.Context, gpus []ml.DeviceInfo, modelPath string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int, kvCacheType string, config llm.LlamaServerConfig, numCtx int) (uint64, int, error)
	newServerFn     func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, config llm.LlamaServerConfig) (llm.LlamaServer, error)
	getGpuFn        func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo
	getSystemInfoFn func() ml.SystemInfo
	waitForRecovery time.Duration
}

// Default automatic value for number of models we allow per GPU
// Model will still need to fit in VRAM, but loading many small models
// on a large GPU can cause stalling
var defaultModelsPerGPU = 3

var ErrMaxQueue = errors.New("server busy, please try again.  maximum pending requests exceeded")

func InitScheduler(ctx context.Context) *Scheduler {
	maxQueue := envconfig.MaxQueue()
	sched := &Scheduler{
		pendingReqCh:    make(chan *LlmRequest, maxQueue),
		finishedReqCh:   make(chan *LlmRequest, maxQueue),
		expiredCh:       make(chan *runnerRef, maxQueue),
		unloadedCh:      make(chan any, maxQueue),
		loaded:          make(map[string]*runnerRef),
		newServerFn:     llm.NewLlamaServer,
		getGpuFn:        discover.GPUDevices,
		getSystemInfoFn: discover.GetSystemInfo,
		waitForRecovery: 5 * time.Second,
		vramCalibration: llm.NewVRAMCalibration(),
		events:          newEventBus(),
		ring:            newFrameRing(retainedWindow),
		samplerWake:     make(chan struct{}, 1),
		deviceCacheTTL:  2 * time.Second,
		// Ten minutes rather than never: a safety net for changes nothing here signals.
		deviceCacheLiveTTL: 10 * time.Minute,
	}
	sched.loadFn = sched.load
	return sched
}

// publishEvent emits a lifecycle event and brings the cached capacity snapshot up to date,
// since every event this publishes is something that changed how much device memory is free.
//
// Updating beats invalidating here. Dropping the cache makes the next reader re-run
// discovery, which costs a subprocess whenever a device has no runner to ask -- and the
// next reader is the sampler, woken by this very event, so the reading that should be the
// promptest of the load was instead the slowest: measured at 840ms after load.start.
// Refreshing free memory from the driver is microseconds and is the only field an event
// can have changed; the device set itself is not affected by a model loading or leaving.
func (s *Scheduler) publishEvent(ev api.ModelEvent) {
	if !s.refreshFreeMemory() {
		s.invalidateDeviceCache()
	}

	// The body is deliberately NOT built here. publishEvent is called from goroutines that
	// hold a runner's refMu for the duration of a load, and building the body takes that
	// same lock for every runner -- so fetching it here deadlocks the load against itself.
	// The stream attaches the body when it serialises the frame instead.
	if ev.At.IsZero() {
		ev.At = time.Now().UTC()
	}
	s.ring.add(retainedFrame{at: ev.At, event: ev})
	s.events.Publish(ev)
	s.wakeSampler()
}

// wakeSampler asks the sampler to take a reading now rather than at its next tick.
//
// The cadence alone is not enough. It is chosen when a tick fires, so a load that starts
// and finishes inside one idle interval is never seen: measured here, weights landed at
// t=3.0s and the first sample of that load arrived at t=6.0s, by which point the memory
// had already been there for three seconds. Every edge that moves memory publishes an
// event, so the events are what should drive the first reading after something happens.
//
// The send is non-blocking and the channel holds one, so a burst of edges coalesces into a
// single extra reading and no publisher ever waits on the sampler.
func (s *Scheduler) wakeSampler() {
	if s.samplerWake == nil {
		return
	}
	select {
	case s.samplerWake <- struct{}{}:
	default:
	}
}

// publishExpiry reports a keep-alive deadline moving, which happens only as a request
// finishes. A client cannot infer this from /api/ps: the field is there, but nothing says
// whether it is being held or approaching.
func (s *Scheduler) publishExpiry(model string, at time.Time) {
	deadline := at
	s.publishEvent(api.ModelEvent{Type: EventExpires, Model: model, ExpiresAt: &deadline})
}

// publishSample emits a periodic level snapshot, which corresponds to no discrete event:
// a KV cache filling during a generation moves the number with nothing to announce it.
func (s *Scheduler) publishSample(ps *api.ProcessResponse, info *api.InfoResponse) {
	at := time.Now().UTC()
	s.ring.add(retainedFrame{at: at, event: api.ModelEvent{Type: "sample", At: at, PS: ps, Info: info}})
	s.events.Publish(api.ModelEvent{Type: "sample", At: at, PS: ps, Info: info})
}

// encodeInfoForCompare renders capacity for change detection only.
func encodeInfoForCompare(info *api.InfoResponse) string {
	if info == nil {
		return ""
	}
	b, err := json.Marshal(info)
	if err != nil {
		return ""
	}
	return string(b)
}

// encodeForCompare renders a body for change detection only.
func encodeForCompare(ps *api.ProcessResponse) string {
	if ps == nil {
		return ""
	}
	b, err := json.Marshal(ps)
	if err != nil {
		return ""
	}
	return string(b)
}

// cachedDevices returns device information for reporting, refreshing it only when the
// cached copy has aged out or a model lifecycle event has invalidated it.
//
// The cache exists because discovery is expensive whenever any device lacks a runner to
// ask, which is the normal state of a multi-GPU host. Free memory is the only field that
// moves, and the things that move it -- a load, an unload, an eviction -- all publish an
// event, so the cache is dropped exactly when it stops being true rather than merely when
// it gets old.
//
// What ollama does not control, such as another process on the same card, is covered by
// reading free memory live from the driver on every read. Where that works the short TTL
// has nothing left to do, and it was expensive: discovery spawns a llama-server that
// initialises every GPU, and the sampler's 15 s idle tick always found a 2 s cache stale,
// so an idle box started one on both cards every ~16 s -- about 5,400 a day (measured
// 2026-09-12). Both GSP faults on this box were a llama-server's first allocation on an
// idle card. So the TTL is long while the live read works, and short only where it does not.
//
// The one thing the old churn did catch is a device that faults while cached, because
// discovery stops returning it. That is now asked of the driver directly, through the same
// 30 s health verdict unavailable_gpus uses, and a cached device that has faulted drops
// the cache at once.
func (s *Scheduler) cachedDevices(ctx context.Context) []ml.DeviceInfo {
	ttl := s.deviceCacheTTL
	if s.liveFreeMemory.Load() && s.deviceCacheLiveTTL > ttl {
		ttl = s.deviceCacheLiveTTL
	}
	s.deviceCacheMu.Lock()
	cached := s.deviceCache
	fresh := cached != nil && time.Since(s.deviceCacheAt) < ttl
	mayCheckFaults := time.Since(s.faultRediscoveredAt) >= deviceFaultRecheck
	s.deviceCacheMu.Unlock()
	if fresh && (!mayCheckFaults || !s.anyFaulted(cached)) {
		return s.devicesWithLiveFreeMemory()
	}
	if fresh {
		s.deviceCacheMu.Lock()
		s.faultRediscoveredAt = time.Now()
		s.deviceCacheMu.Unlock()
	}

	devices := s.getGpuFn(ctx, s.runnerDiscoverySnapshot())
	s.deviceNames.observe(devices)

	s.deviceCacheMu.Lock()
	s.deviceCache = devices
	s.deviceCacheAt = time.Now()
	s.deviceCacheMu.Unlock()
	return s.devicesWithLiveFreeMemory()
}

// deviceFaultRecheck is how often a fault may trigger rediscovery: the life of one health
// verdict (discover.unavailableTTL), since a sooner check would read the same verdict.
const deviceFaultRecheck = 30 * time.Second

// anyFaulted reports whether one of these devices, all of which discovery offered, can no
// longer compute.
func (s *Scheduler) anyFaulted(devices []ml.DeviceInfo) bool {
	unavailable := s.unavailableFn
	if unavailable == nil {
		unavailable = discover.CachedUnavailableDevices
	}
	known := make([]string, 0, len(devices))
	for _, d := range devices {
		if d.PCIID != "" {
			known = append(known, d.PCIID)
		}
	}
	if len(known) == 0 {
		return false
	}
	for _, u := range unavailable(known) {
		for _, pci := range known {
			if strings.EqualFold(u.PCIID, pci) {
				slog.Warn("a GPU faulted while in use; re-discovering devices", "pci_id", u.PCIID, "reason", u.Reason)
				return true
			}
		}
	}
	return false
}

// devicesWithLiveFreeMemory returns the cached devices with free memory read from the
// driver, and is what every reported figure goes through.
//
// It is on the read path rather than only on the paths that care about liveness, because
// the two sources do not agree: discovery's free memory and the driver's differ by the
// memory the driver reserves for itself, about 1.1 GiB across two cards here. Refreshing
// only sometimes made a series that switched sources mid-flight, which drew a step of that
// size at every lifecycle event -- a change in where the number came from, rendered as if
// memory had moved. One source for every reading is worth more than the microseconds.
func (s *Scheduler) devicesWithLiveFreeMemory() []ml.DeviceInfo {
	s.refreshFreeMemory()

	s.deviceCacheMu.Lock()
	defer s.deviceCacheMu.Unlock()
	return s.deviceCache
}

// refreshFreeMemory updates the free-memory figure on the cached devices, in place, from
// the driver.
//
// This is deliberately not a discovery refresh. Discovery enumerates devices and costs a
// subprocess whenever any of them lacks a runner to ask, which during a load is all of
// them -- the runner being loaded is not serving yet. Re-running it at a sampling cadence
// would spawn a probe every tick. Reading free memory alone is a handful of microseconds
// and answers the only question that moves during a load.
//
// The device list itself is left alone: nothing is added, removed or reordered, so a
// caller holding the cache sees the same devices with a newer number. If the driver cannot
// be read the cached figures stand, which is the pre-existing behaviour.
// It reports whether the cached figures were actually updated, so a caller that needs them
// current can fall back to dropping the cache when the driver cannot be read.
func (s *Scheduler) refreshFreeMemory() bool {
	s.deviceCacheMu.Lock()
	devices := s.deviceCache
	s.deviceCacheMu.Unlock()
	if len(devices) == 0 {
		return false
	}

	ids := make([]string, 0, len(devices))
	for _, d := range devices {
		if d.PCIID != "" {
			ids = append(ids, d.PCIID)
		}
	}

	readFree := s.freeMemoryFn
	if readFree == nil {
		readFree = discover.FreeMemoryByPCI
	}
	free := readFree(ids)
	s.liveFreeMemory.Store(len(free) > 0)
	if len(free) == 0 {
		return false
	}
	// Read at the same moment and from the same held-open session, so the two figures on a
	// device describe one instant and the read costs nothing extra to open.
	util := discover.UtilizationByPCI(ids)

	s.deviceCacheMu.Lock()
	defer s.deviceCacheMu.Unlock()
	// The cache may have been dropped or replaced while the driver was being read; the
	// figures below describe the list that was read, so only apply them to that list.
	if len(s.deviceCache) != len(devices) {
		return false
	}
	for i := range s.deviceCache {
		if got, ok := free[s.deviceCache[i].PCIID]; ok {
			s.deviceCache[i].FreeMemory = got
		}
		// Replaced, not merged: a device that stopped answering loses its old figure rather
		// than showing a stale one as current.
		if u, ok := util[s.deviceCache[i].PCIID]; ok {
			s.deviceCache[i].Utilization = &u
		} else {
			s.deviceCache[i].Utilization = nil
		}
	}
	s.deviceCacheAt = time.Now()
	return true
}

// seedDeviceCache stores a discovery made outside cachedDevices, as if cachedDevices had made
// it, unless the cache already holds one.
func (s *Scheduler) seedDeviceCache(devices []ml.DeviceInfo) {
	s.deviceCacheMu.Lock()
	seeded := s.deviceCache == nil && len(devices) > 0
	if seeded {
		s.deviceCache, s.deviceCacheAt = devices, time.Now()
	}
	s.deviceCacheMu.Unlock()
	if seeded {
		s.deviceNames.observe(devices)
	}
}

// invalidateDeviceCache drops the cached discovery so the next reader re-reads. Called
// when something ollama did has changed how much memory is free.
func (s *Scheduler) invalidateDeviceCache() {
	s.deviceCacheMu.Lock()
	s.deviceCache = nil
	s.deviceCacheMu.Unlock()
}

// runnerDiscoverySnapshot returns the loaded runners as discovery participants.
//
// Device discovery has two paths: it can ask the runners that already hold a device how
// much memory is free, which is cheap, or it can spawn a bootstrap probe, which is not.
// It takes the first only when the runners it is given can account for every device, so a
// caller that passes none forces the expensive path every time.
func (s *Scheduler) runnerDiscoverySnapshot() []ml.FilteredRunnerDiscovery {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()

	runners := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	return runners
}

// vramCalibrationKey describes the inputs to a placement decision, so that a later
// decision made from the same inputs can reuse what the earlier one measured.
//
// It deliberately describes the inputs rather than what the load ended up doing. The rest
// of the pipeline is deterministic given these, so keying on them keeps the sample that a
// load records addressable by the next load that would make the same decision — and any
// change to them selects a different bucket, which is what stops a stale sample being
// applied to a load it no longer describes.
func vramCalibrationKey(req *LlmRequest, gpus []ml.DeviceInfo, numParallel int) llm.CalibrationKey {
	return llm.CalibrationKey{
		Model:          req.model.ModelPath,
		ModelSize:      modelFileSize(req.model.ModelPath),
		Projectors:     strings.Join(req.model.ProjectorPaths, ","),
		KVCacheType:    envconfig.KvCacheType(),
		FlashAttention: llm.LlamaServerFlashAttention(gpus) == ml.FlashAttentionEnabled,
		NumBatch:       req.opts.NumBatch,
		NumParallel:    numParallel,
		NumGPU:         len(gpus),
	}
}

// warnIfPredictionIsALowerBound reports a placement being decided from an estimate that
// cannot account for this architecture, and that has no measurement to fall back on.
//
// It is the one case where the prediction is not merely imprecise but known-incomplete in
// a known direction: too low, by more with more context. Saying so makes a placement that
// later fails attributable, rather than looking like the model misbehaving.
func (s *Scheduler) warnIfPredictionIsALowerBound(key llm.CalibrationKey, f *ggml.GGML, modelPath string, numCtx int, predicted uint64) {
	if f.KV().KVCacheModelIsComplete() {
		return
	}
	if _, calibrated := s.vramCalibration.Predict(key, numCtx, 0, 0); calibrated {
		return
	}
	slog.Warn("placing a model whose memory use cannot be derived from its metadata, and which has not been measured yet",
		"model", modelPath,
		"architecture", f.KV().Architecture(),
		"predicted", format.HumanBytes2(predicted),
		"num_ctx", numCtx,
		"note", "this estimate is a lower bound; it is corrected by measurement once this model has loaded once")
}

// probeContexts are the two context lengths a cold probe measures at. Any two distinct
// points determine the line, and these are cheap: the pass never reads tensors, so its
// cost is flat in model size. They bracket ordinary use rather than sitting at the
// extremes, because the line is fitted, not interpolated.
var probeContexts = [2]int{8192, 131072}

// probeMinContext is the floor a halved probe point will not go below. Under it the two
// points sit so close together that the slope is dominated by the 256-token rounding
// rather than by the model.
const probeMinContext = 256

// probePoints picks the two contexts to measure at, or reports that no usable pair exists.
//
// The ordinary pair brackets normal use. Both are clamped to what the model supports, and
// on a model whose ceiling is at or below the lower point they clamp onto the *same* value
// -- two coincident points, no line, and until this the probe simply gave up. That cost
// about a tenth of the models on this box, and not a random tenth: the short-context models
// here are disproportionately sliding-window architectures (cohere2, gemma2, gemma3), which
// is exactly where the metadata fallback is weakest.
//
// Nothing about the pair has to be 8192 and 131072 -- any two distinct contexts determine
// the line. So when the ordinary pair collapses, put the second point *below* the ceiling
// instead of clamping both onto it. Rounding down to a multiple of 256 keeps the requested
// and granted contexts equal, since llama.cpp silently rounds a context *up* to a multiple
// of 256 and a sample must be filed against the context it actually ran at.
//
// Models with a normal ceiling keep exactly the contexts they use today, so samples already
// recorded stay comparable.
// Takes the model's trained context rather than the model, because it is arithmetic and
// nothing else -- which is also what makes it testable without building a GGML.
// probeContextsFor adds the context being loaded to the fixed probe points when it lies
// beyond them.
//
// The fixed points stop at 131072, and the most common trained context on this box is
// 262144 -- 26 of 70 models -- which is exactly what an automatic context loads them at. So
// the figure every one of those loads was placed from was the line extrapolated to twice
// the furthest point anyone measured. Probing the requested context turns that into a
// measurement, for one more ~0.5 s probe, once per model.
//
// Measured afterwards on all 31 models here trained past 131072 (slop-zone
// notebooks/context-vs-vram.ipynb): the extrapolation was never off by more than 590 MiB
// (0.4%, dsv4-flash), and by exactly 0 for 19 of them, because allocation is close to
// linear in context. So the third point turned an assumption into a measurement rather than
// correcting a real error. It pays past 262144, where the line's error grows with distance:
// -2.2 GiB at 1M on dsv4-flash.
//
// Beyond the points it is added rather than substituted. A requested context too large to
// fit in the memory free right now fails to probe, and substituting it would then leave one
// point -- which probeCalibration rightly discards -- where the fixed pair would have
// produced a line.
//
// Between the points it replaces the nearer one, so the line is exact where this load uses
// it and still spans the range. The worry above does not apply there: a context inside the
// pair fits whenever the upper point does. This is also what keeps the placement decision
// free. settlePlacement measures the load's own context to decide where the load goes, and
// that measurement is then one of these points rather than an extra probe.
func probeContextsFor(points [2]int, numCtx int) []int {
	switch {
	case numCtx > points[1]:
		return []int{points[0], points[1], numCtx}
	case numCtx <= 0 || numCtx == points[0] || numCtx == points[1]:
		return []int{points[0], points[1]}
	case numCtx-points[0] > points[1]-numCtx:
		return []int{points[0], numCtx}
	default:
		return []int{numCtx, points[1]}
	}
}

func probePoints(trainCtx, numParallel int) ([2]int, bool) {
	contexts := probeContexts
	for i := range contexts {
		contexts[i] = effectiveContext(contexts[i], trainCtx) * max(numParallel, 1)
	}
	if contexts[0] != contexts[1] {
		return contexts, true
	}

	ceiling := contexts[1]
	lower := (ceiling / 2 / 256) * 256
	if lower < probeMinContext || lower >= ceiling {
		// A ceiling too small to split. Rare, and the metadata estimate is at its most
		// accurate here anyway: at 256 tokens the KV cache is a rounding error beside
		// the weights, so what it gets wrong barely matters.
		return contexts, false
	}
	return [2]int{lower, ceiling}, true
}

// probeCalibration measures a model at two or three context lengths without loading it,
// whenever the load's invocation has not been measured here yet.
//
// Without this the first load of a model is placed from its metadata, which is off by
// hundreds of GiB for the architectures it cannot describe. The probe costs about a second
// per point and turns the first load into a calibrated one.
//
// Samples are recorded exactly as a completed load's are, because they are the same
// quantity: llama-server's fit pass reports what the load would hold, and replaying the
// real invocation with only the context changed reproduced two real measurements to
// within 0.01 GiB. That equivalence is the whole reason this is worth doing, and it is
// entirely dependent on the invocation being identical -- see ProbeFitVRAM.
//
// Every model is measured, not only those whose metadata admits it cannot describe them.
// That gate used to decide from metadata keys whether the metadata could be trusted, and it
// was the last place the estimate was still believed: measured on 2026-09-11 across the 31
// models here trained past 131072 (slop-zone notebooks/context-vs-vram.ipynb), the one model
// it let through, qwen3:235b, was placed 9.3 GiB under what it uses at 262144, while every
// probed line landed within 590 MiB. The metadata line is now only the first guess that
// placement starts from, and settlePlacement corrects the placement when the measurement
// disagrees with it.
//
// premeasured holds samples already taken for this key -- the one settlePlacement took to
// decide the placement -- keyed by the context they were measured at. They count toward the
// line and their contexts are not probed again.
func (s *Scheduler) probeCalibration(ctx context.Context, key llm.CalibrationKey, req *LlmRequest, f *ggml.GGML, launchOpts api.Options, gpus []ml.DeviceInfo, numParallel int, numCtx int, premeasured map[int]uint64) bool {
	// Two distinct samples are what it takes to stop consulting the metadata: below that
	// the slope still comes from the prior, and for these architectures the prior slope is
	// the broken part. So the bar is a fitted line, not merely a calibrated answer -- one
	// sample plus a wrong slope measured worse than no samples at all.
	if s.vramCalibration.SampleCount(key) >= 2 {
		return false
	}

	// Probing at one context tells us nothing a metadata estimate does not: the point is
	// the slope, and a slope needs two points.
	points, ok := probePoints(modelTrainContext(f), numParallel)
	if !ok {
		return false
	}
	contexts := probeContextsFor(points, numCtx)

	started := time.Now()
	var recorded int
	var probes []string
	for c, vram := range premeasured {
		if vram > 0 {
			s.recordCalibration(key, c, vram, "probe")
			recorded++
		}
	}
	for _, probeCtx := range contexts {
		// A context already measured is not asked again, including one that failed: the
		// memory free has not changed in the moments since.
		if _, done := premeasured[probeCtx]; done {
			continue
		}
		probeStarted := time.Now()
		vram, measuredCtx, ok := s.probeOnce(ctx, req, f, launchOpts, gpus, numParallel, probeCtx)
		probes = append(probes, fmt.Sprintf("%d:%s", probeCtx, time.Since(probeStarted).Round(time.Millisecond)))
		if !ok {
			continue
		}
		s.recordCalibration(key, measuredCtx, vram, "probe")
		recorded++
	}

	// One sample is worse than none here. It fixes the intercept but leaves the slope
	// coming from the metadata prior, which for this architecture is the thing known to
	// be wrong -- so the result would look calibrated while carrying the same error.
	if recorded < 2 {
		s.forgetCalibration(key)
		return false
	}

	slog.Info("measured a model's memory use without loading it",
		"model", req.model.ModelPath,
		"architecture", f.KV().Architecture(),
		"contexts", contexts,
		"reused", len(premeasured),
		"probes", probes,
		"took", time.Since(started).Round(time.Millisecond),
		"reason", "no measurement of this model in this configuration yet")
	return true
}

// probeOnce measures one context without loading the model, and reports the context the
// measurement is filed at.
func (s *Scheduler) probeOnce(ctx context.Context, req *LlmRequest, f *ggml.GGML, launchOpts api.Options, gpus []ml.DeviceInfo, numParallel, numCtx int) (vram uint64, measuredCtx int, ok bool) {
	// launchOpts, not req.opts: the placement decision writes the split mode, the main
	// device and the offload count into it, and those are part of the invocation being
	// measured. Probing with the request's own options measures a load that was never
	// going to run -- on qwen3.8:27b at 128k that reported 15.86 GiB against 23.86 for
	// the same model probed as it would actually be launched.
	probe := s.fitProbe
	if probe == nil {
		probe = llm.ProbeFitVRAM
	}
	vram, measuredCtx, err := probe(ctx, gpus, req.model.ModelPath, f, req.model.AdapterPaths, req.model.ProjectorPaths,
		launchOpts, numParallel, envconfig.KvCacheType(), llamaServerConfigForModel(req.model), numCtx)
	if err != nil {
		// Not fitting is worth saying out loud: it is the reason this load is about
		// to be placed from a lower bound, and it is recoverable -- the probe runs
		// before the pre-flight eviction, so a model resident right now can be what
		// denies it. The next load, once that model has gone, will measure normally.
		if errors.Is(err, llm.ErrFitProbeWouldNotFit) {
			slog.Info("could not measure this model without loading it: it does not fit in the memory free right now",
				"model", req.model.ModelPath, "num_ctx", numCtx)
		} else {
			slog.Debug("could not measure model memory without loading it", "model", req.model.ModelPath, "num_ctx", numCtx, "error", err)
		}
		return 0, numCtx, false
	}
	// Filed at the context the probe measured, not the one it asked for: llama.cpp
	// rounds each slot up to a multiple of 256, so a probe point off that grid measures
	// a larger cache than its label says. No model on this box trains at an unaligned
	// context, so today the two agree -- the point is not to depend on that.
	if measuredCtx <= 0 {
		measuredCtx = numCtx
	}
	return vram, measuredCtx, true
}

// settlePlacement chooses where a load runs, measures what it costs there, and moves it once
// if the measurement disagrees with the estimate the choice was made from.
//
// The first choice has to come from an estimate: the placement is part of the invocation a
// probe measures, so the measurement cannot come first. On a model's first load that
// estimate is its metadata, and the placement used to stand on it whatever the probe then
// said. Both directions of that are on record here. The calibration store holds gemma4:31b
// measured on two cards because a metadata estimate hundreds of GiB high split a 43 GiB
// model; and an estimate that is low pins a model to one card, where it can only spill to the
// CPU, because a single-card load does not see the other card.
//
// So the measurement gets one chance to move the placement, and the new placement is
// measured in turn. One move, not a loop: a second disagreement is left to the fit check
// that follows, rather than letting two estimates chase each other.
//
// The first placement is only asked what the load itself costs (decideOnly), not measured
// in full, because it may be about to be abandoned. Measured in full, the abandoned
// placement cost 11.8 s of a 22 s load (gemma4:26b at 262144, 2026-09-12): three probes on
// two cards, all thrown away. The one probe it does take is not wasted when the placement
// stands, because it is at one of the contexts the full measurement uses (probeContextsFor).
func settlePlacement(
	estimate uint64,
	place func(estimate uint64) ([]ml.DeviceInfo, api.Options),
	measure func(placed []ml.DeviceInfo, opts api.Options, estimate uint64, decideOnly bool) (api.Options, llm.CalibrationKey, uint64, bool),
) (gpus []ml.DeviceInfo, opts api.Options, key llm.CalibrationKey, predicted uint64, probed bool) {
	gpus, opts = place(estimate)
	_, _, cost, _ := measure(gpus, opts, estimate, true)
	if cost != estimate {
		if moved, movedOpts := place(cost); !sameDevices(moved, gpus) {
			slog.Info("the measurement moved the placement; measuring it where it will run",
				"estimate", format.HumanBytes2(estimate), "measured", format.HumanBytes2(cost),
				"devices", len(gpus), "moved_to", len(moved))
			gpus, opts, estimate = moved, movedOpts, cost
		}
	}
	// probed describes the final measurement only. It labels where the prediction came
	// from, and a probe of a placement that was then abandoned -- or one that failed --
	// contributed nothing to it: after a move onto an already-calibrated placement, the
	// number is calibration, whatever was probed on the way.
	opts, key, predicted, probed = measure(gpus, opts, estimate, false)
	return gpus, opts, key, predicted, probed
}

func sameDevices(a, b []ml.DeviceInfo) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].DeviceID != b[i].DeviceID {
			return false
		}
	}
	return true
}

// predictionSource names where a prediction came from, which is the difference between a
// number that was measured and one that was inferred from metadata.
func predictionSource(calibrated, probed bool) string {
	switch {
	case probed:
		return "probe"
	case calibrated:
		return "calibration"
	default:
		return "metadata"
	}
}

// predictLlamaServerVRAM estimates VRAM for a llama-server load, preferring a measurement
// of an earlier load made from the same inputs over the metadata estimate.
func predictLlamaServerVRAM(cal *llm.VRAMCalibration, key llm.CalibrationKey, req *LlmRequest, f *ggml.GGML, numCtx int) uint64 {
	weights, bytesPerToken := llm.PredictServerVRAMParts(req.model.ModelPath, req.model.ProjectorPaths, f)
	predicted, _ := cal.Predict(key, numCtx, weights, bytesPerToken)
	return predicted
}

// schedulerModelKey returns the scheduler map key for a model.
// GGUF-backed models use ModelPath; safetensors/image models without a
// ModelPath use manifest digest so distinct models don't collide.
func schedulerModelKey(m *Model) string {
	if m == nil {
		return ""
	}
	if m.ModelPath != "" {
		return m.ModelPath
	}
	if m.Digest != "" {
		return "digest:" + m.Digest
	}
	if m.Name != "" {
		return "name:" + m.Name
	}
	if m.ShortName != "" {
		return "short:" + m.ShortName
	}
	return ""
}

func resolveContextShift(shift *bool, m *Model) bool {
	if shift != nil {
		return *shift
	}

	return supportsContextShift(m)
}

func supportsContextShift(m *Model) bool {
	if m == nil {
		return true
	}

	if m.Config.ModelFamily == "deepseek2" || slices.Contains(m.Config.ModelFamilies, "deepseek2") {
		return false
	}

	return true
}

func effectiveModelContext(numCtx int, f *ggml.GGML) int {
	return effectiveContext(numCtx, modelTrainContext(f))
}

func modelTrainContext(f *ggml.GGML) int {
	if f == nil {
		return 0
	}

	return int(f.KV().ContextLength())
}

func effectiveContext(numCtx, trainCtx int) int {
	if trainCtx > 0 && numCtx > trainCtx {
		return trainCtx
	}

	return numCtx
}

func (s *Scheduler) getRunner(c context.Context, m *Model, opts api.Options, sessionDuration *api.Duration, numCtxAuto bool, numBatchAuto bool, shift *bool) (chan *runnerRef, chan error) {
	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	if m.CheckCapabilities(model.CapabilityVision) == nil {
		// multimodal models require at least 2048 context
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	contextShift := false
	if m.ModelPath != "" {
		contextShift = resolveContextShift(shift, m)
	}

	req := &LlmRequest{
		ctx:             c,
		model:           m,
		opts:            opts,
		sessionDuration: sessionDuration,
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
		numCtxAuto:      numCtxAuto,
		numBatchAuto:    numBatchAuto,
		contextShift:    contextShift,
		shift:           shift,
	}

	key := schedulerModelKey(req.model)
	s.loadedMu.Lock()
	runner := s.loaded[key]
	s.loadedMu.Unlock()
	if runner != nil && !runner.needsReload(c, req) && !s.onLeasedDevice(runner) {
		if req.useLoadedRunner(runner, s.finishedReqCh) {
			s.publishEvent(api.ModelEvent{Type: EventBusyStart, Model: runner.name})
			s.publishEvent(api.ModelEvent{Type: EventGenStart, Model: runner.name})
		}
	} else {
		select {
		case s.pendingReqCh <- req:
		default:
			req.errCh <- ErrMaxQueue
		}
	}
	return req.successCh, req.errCh
}

// Returns immediately, spawns go routines for the scheduler which will shutdown when ctx is done
func (s *Scheduler) Run(ctx context.Context) {
	slog.Debug("starting llm scheduler")
	go func() {
		s.processPending(ctx)
	}()

	go func() {
		s.processCompleted(ctx)
	}()
}

func (s *Scheduler) processPending(ctx context.Context) {
	maxRunners := envconfig.MaxRunners()

	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler pending loop")
			return
		case pending := <-s.pendingReqCh:
			// A measurement of the machine yields to any request, and its memory is free
			// before this one is placed.
			s.profiler.preempt()

			// Block other requests until we get this pending request running
			pending.schedAttempts++

			if pending.ctx.Err() != nil {
				slog.Debug("pending request cancelled or timed out, skipping scheduling")
				continue
			}
			logutil.Trace("processing incoming request", "model", pending.model.ModelPath)

			for {
				var runnerToExpire *runnerRef
				pendingKey := schedulerModelKey(pending.model)
				s.loadedMu.Lock()
				runner := s.loaded[pendingKey]
				loadedCount := len(s.loaded)
				runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
				for _, r := range s.loaded {
					runnersSnapshot = append(runnersSnapshot, r)
				}
				s.loadedMu.Unlock()

				if runner != nil {
					if runner.needsReload(ctx, pending) || s.onLeasedDevice(runner) {
						slog.Debug("reloading", "runner", runner)
						runnerToExpire = runner
					} else {
						// Runner is usable, return it
						logutil.Trace("using existing loaded runner", "model", pendingKey)
						if pending.useLoadedRunner(runner, s.finishedReqCh) {
							s.publishEvent(api.ModelEvent{Type: EventBusyStart, Model: runner.name})
							s.publishEvent(api.ModelEvent{Type: EventGenStart, Model: runner.name})
						}
						break
					}
				} else if maxRunners > 0 && loadedCount >= int(maxRunners) {
					slog.Debug("max runners achieved, unloading one to make room", "runner_count", loadedCount)
					runnerToExpire = s.findRunnerToUnload()
				} else {
					// Either no models are loaded or below envconfig.MaxRunners
					// Get a refreshed GPU list
					var gpus []ml.DeviceInfo
					if pending.opts.NumGPU == 0 {
						gpus = []ml.DeviceInfo{}
					} else {
						logutil.Trace("refreshing GPU list", "model", pending.model.ModelPath)
						all := s.getGpuFn(ctx, runnersSnapshot)
						gpus = s.withoutLeased(all)
						// Every GPU is lent to a job. Falling back to the CPU would take the
						// cores and memory bandwidth the job needs, so the request fails with
						// the reason instead; a request that asked for the CPU still runs.
						if len(all) > 0 && len(gpus) == 0 {
							pending.errCh <- errAllGPUsLeased
							break
						}
					}
					logutil.Trace("refreshing system information", "model", pending.model.ModelPath)
					systemInfo := s.getSystemInfoFn()
					if maxRunners <= 0 {
						// No user specified MaxRunners, so figure out what automatic setting to use for the next load attempt
						if pending.opts.NumGPU == 0 {
							// Only the number of GPUs is needed, which the device cache knows.
							// Fresh discovery starts a llama-server on every GPU, and a CPU-pinned
							// load has no reason to touch them.
							g := s.cachedDevices(ctx)
							maxRunners = uint(defaultModelsPerGPU * max(len(g), 1))
						} else {
							maxRunners = uint(defaultModelsPerGPU * max(len(gpus), 1))
						}
						slog.Debug("updating default concurrency", "OLLAMA_MAX_LOADED_MODELS", maxRunners, "gpu_count", len(gpus))
					}

					// Update free memory from currently loaded models
					logutil.Trace("updating free space", "gpu_count", len(gpus), "model", pending.model.ModelPath)
					s.updateFreeSpace(gpus)

					if loadedCount == 0 {
						// No models loaded. Load the model but prefer the best fit.
						slog.Debug("loading first model", "model", pending.model.ModelPath)
						if s.loadFn(pending, systemInfo, gpus, false) {
							slog.Debug("first model load requested retry", "model", pending.model.ModelPath)
							continue
						}
						break
					}

					// More than one loaded model, so we have to see if the
					// new one fits
					logutil.Trace("loading additional model", "model", pending.model.ModelPath)
					needEvict := s.loadFn(pending, systemInfo, gpus, true)
					if !needEvict {
						slog.Debug("new model fits with existing models, loading")
						break
					}

					// OOM retry path: load() crashed post-spawn and we still
					// have other models resident. Evict all of them, wait for
					// every unload, then loop back to retry the load once.
					// load() has already set oomRetryAttempted so a second
					// crash falls through to the fail-fast path.
					if pending.oomRetryAttempted {
						if !s.evictAllAndWait(ctx, pendingKey) {
							return
						}
						continue
					}

					runnerToExpire = s.findRunnerToUnload()
				}

				if runnerToExpire == nil {
					// While we were performing load calculations, the loaded runner(s) unloaded in parallel
					// so findRunnerToUnload returned no runners.  We'll try again and the loadedCount should be zero
					slog.Debug("runner to expire was nil, retrying")
					continue
				}
				// Trigger an expiration to unload once it's done
				runnerToExpire.refMu.Lock()
				slog.Debug("resetting model to expire immediately to make room", "runner", runnerToExpire, "refCount", runnerToExpire.refCount)
				if runnerToExpire.expireTimer != nil {
					runnerToExpire.expireTimer.Stop()
					runnerToExpire.expireTimer = nil
				}
				runnerToExpire.sessionDuration = 0
				if runnerToExpire.refCount <= 0 {
					s.expiredCh <- runnerToExpire
				}
				runnerToExpire.refMu.Unlock()
				// Wait for the unload to happen
				slog.Debug("waiting for pending requests to complete and unload to occur", "runner", runnerToExpire)
				select {
				case <-ctx.Done():
					slog.Debug("shutting down scheduler pending loop")
					return
				case <-s.unloadedCh:
					slog.Debug("unload completed", "runner", runnerToExpire)
					continue
				}
			}
		case <-s.unloadedCh:
			// An unload request when there are no pending request can be ignored
			slog.Debug("ignoring unload event with no pending requests")
		}
	}
}

func (s *Scheduler) processCompleted(ctx context.Context) {
	// Process completed requests, expired timers, and unloading models
	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler completed loop")
			return
		case finished := <-s.finishedReqCh:
			finishedKey := schedulerModelKey(finished.model)
			s.loadedMu.Lock()
			runner := s.loaded[finishedKey]
			s.loadedMu.Unlock()
			if runner == nil {
				slog.Error("finished request signal received after model unloaded", "modelPath", finishedKey)
				continue
			}
			runner.refMu.Lock()
			runner.refCount--
			if runner.refCount <= 0 {
				s.publishEvent(api.ModelEvent{Type: EventBusyEnd, Model: runner.name})
				if runner.sessionDuration <= 0 {
					slog.Debug("runner with zero duration has gone idle, expiring to unload", "runner", runner)
					if runner.expireTimer != nil {
						runner.expireTimer.Stop()
						runner.expireTimer = nil
					}
					s.expiredCh <- runner
				} else if runner.expireTimer == nil {
					slog.Debug("runner with non-zero duration has gone idle, adding timer", "runner", runner, "duration", runner.sessionDuration)
					runner.expireTimer = time.AfterFunc(runner.sessionDuration, func() {
						slog.Debug("timer expired, expiring to unload", "runner", runner)
						runner.refMu.Lock()
						defer runner.refMu.Unlock()
						if runner.expireTimer != nil {
							runner.expireTimer.Stop()
							runner.expireTimer = nil
						}
						s.expiredCh <- runner
					})
					runner.expiresAt = time.Now().Add(runner.sessionDuration)
					s.publishExpiry(runner.name, runner.expiresAt)
				} else {
					slog.Debug("runner with non-zero duration has gone idle, resetting timer", "runner", runner, "duration", runner.sessionDuration)
					runner.expireTimer.Reset(runner.sessionDuration)
					runner.expiresAt = time.Now().Add(runner.sessionDuration)
					s.publishExpiry(runner.name, runner.expiresAt)
				}
			}
			slog.Debug("after processing request finished event", "runner", runner, "refCount", runner.refCount)
			runner.refMu.Unlock()
		case runner := <-s.expiredCh:
			slog.Debug("runner expired event received", "runner", runner)
			runner.refMu.Lock()
			if runner.refCount > 0 {
				slog.Debug("expired event with positive ref count, retrying", "runner", runner, "refCount", runner.refCount)
				go func(runner *runnerRef) {
					// We can't unload yet, but want to as soon as the current request completes
					// So queue up another expired event
					time.Sleep(10 * time.Millisecond)
					s.expiredCh <- runner
				}(runner)
				runner.refMu.Unlock()
				continue
			}

			s.loadedMu.Lock()
			slog.Debug("got lock to unload expired event", "runner", runner)
			runnerToUnload := s.loaded[runner.modelKey]
			if runnerToUnload == nil {
				// If runnerToUnload is nil, we already processed an event and
				// unloaded it. This double unload can happen if the initial
				// request is canceled and we're trying to load another model
				// that requires this one to be evicted, or the settings change
				// and require a reload
				s.loadedMu.Unlock()
				runner.refMu.Unlock()
				slog.Debug("duplicate expired event, ignoring", "runner", runner)
			} else if runner.pid != runnerToUnload.pid {
				// If the pids do not match, we likely had multiple load
				// failures for the same model in quick succession due to
				// request context canceled and are draining the queue of
				// events. Ensure the orphaned runner is properly shut down, but
				// do not delete the mismatched loaded runner, or wait for VRAM
				// convergence.
				slog.Debug("orphaned runner shutting down", "orphan", runner, "loaded", runnerToUnload)
				runner.unload()
				s.loadedMu.Unlock()
				runner.refMu.Unlock()
			} else {
				slog.Debug("starting background wait for VRAM recovery", "runner", runner)
				runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
				for _, r := range s.loaded {
					runnersSnapshot = append(runnersSnapshot, r)
				}
				finished := s.waitForVRAMRecovery(runner, runnersSnapshot)
				// unload() clears runner.model, so the name has to be taken before it.
				// Read afterwards it is always empty, and an unload frame naming no model
				// cannot be attributed to anything by a client watching memory.
				name := ""
				if runner.model != nil {
					name = runner.model.Name
				}
				runner.unload()
				delete(s.loaded, runner.modelKey)
				s.loadedMu.Unlock()
				slog.Debug("runner terminated and removed from list, blocking for VRAM recovery", "runner", runner)
				<-finished
				runner.refMu.Unlock()
				s.publishEvent(api.ModelEvent{Type: EventUnload, Model: name})
				slog.Debug("sending an unloaded event", "runner", runner)
				s.unloadedCh <- struct{}{}
			}
		}
	}
}

// useLoadedRunner completes the pending request, sends the runner back to the requester,
// wires up a finished event after the request context completes, and resets the expiration
// timer. It attaches a request to a runner that is already loaded. It reports whether
// this was the transition from idle to working, which the caller publishes: the scheduler is
// not in scope here, and the count must be read under the lock that guards it.
func (pending *LlmRequest) useLoadedRunner(runner *runnerRef, finished chan *LlmRequest) (becameBusy bool) {
	runner.refMu.Lock()
	defer runner.refMu.Unlock()
	runner.refCount++
	becameBusy = runner.refCount == 1
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	if pending.sessionDuration != nil {
		runner.sessionDuration = pending.sessionDuration.Duration
	}
	pending.successCh <- runner
	go func() {
		<-pending.ctx.Done()
		slog.Debug("context for request finished", "runner", runner)
		finished <- pending
	}()
	return becameBusy
}

// load creates a new model based on req and loads it. If requireFull is true then the model must be loaded fully onto GPUs
// (if any). Returns whether the scheduler needs to evict a model to make this one fit.
func (s *Scheduler) load(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool {
	numParallel := max(int(envconfig.NumParallel()), 1)
	completion := req.model.CheckCapabilities(model.CapabilityCompletion) == nil

	// Embedding models should always be loaded with parallel=1
	if !completion {
		numParallel = 1
	}

	// Some architectures are not safe with num_parallel > 1.
	// ref: https://github.com/ollama/ollama/issues/4165
	if slices.Contains([]string{"mllama", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "qwen3next", "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe", "nemotron_h_omni"}, req.model.Config.ModelFamily) && numParallel != 1 {
		numParallel = 1
		slog.Warn("model architecture does not currently support parallel requests", "architecture", req.model.Config.ModelFamily)
	}

	sessionDuration := envconfig.KeepAlive()
	if req.sessionDuration != nil {
		sessionDuration = req.sessionDuration.Duration
	}

	s.loadedMu.Lock()
	llama := s.activeLoading
	var f *ggml.GGML
	loadGpus := gpus
	var launchOpts api.Options

	// When the load began, so load.complete can report a duration that covers the whole
	// thing rather than the sliver after the runner object was created.
	var loadStartedAt time.Time

	// Every load.start must be followed by exactly one load.complete or load.failed, and the
	// loading row and the in-flight count it set up must be undone. That used to happen only
	// on the goroutine that finishes a successful launch, so every exit between load.start
	// and that handoff leaked all three -- six of them, three of which are retries that
	// announce again and so drove loadsInFlight above zero for good. Measured: a client that
	// disconnected mid-load left a "loading" row on /api/ps that never cleared, for a load
	// whose llama-server had already exited, and no load.failed on the stream.
	var loadAnnounced, loadSettled bool
	abandon := func(err error, retrying bool) {
		if !loadAnnounced || loadSettled {
			return
		}
		loadSettled = true
		s.abandonLoad(req, loadStartedAt, err, retrying)
	}
	defer func() {
		// The net under the explicit calls below. Reaching it means an exit was added
		// without routing through abandon, so it says so rather than passing as ordinary.
		if loadAnnounced && !loadSettled {
			slog.Warn("load exited without settling the state it announced; cleaning up", "model", req.model.ModelPath)
			abandon(errors.New("load ended before the runner was ready"), false)
		}
	}()
	var weightsLoadedAt time.Time

	// The calibration key as it stood when the prediction was made. It must be captured
	// here rather than rebuilt later: applyAutomaticGenerationBatch rewrites
	// req.opts.NumBatch further down, so a key built after it describes a different load
	// than the one that was predicted, and the sample would be filed where no prediction
	// ever looks.
	var calibrationKey llm.CalibrationKey

	// The estimate this load was placed from, kept for the usage record written when it completes.
	var loadEstimate *api.LoadEstimate

	// The context the prediction was made at, kept in scope so the measurement this load
	// produces is recorded against the same value. req.opts.NumCtx is rewritten further
	// down from what llama-server actually chose, so it cannot be recomputed later.
	var predictedCtx int

	if llama == nil {
		var err error
		if !req.model.IsMLX() {
			var loadErr error
			f, loadErr = llm.LoadModel(req.model.ModelPath, 1024)
			if loadErr != nil {
				slog.Info("failed to load model metadata", "model", req.model.ModelPath, "error", loadErr)
				req.errCh <- loadErr
				s.loadedMu.Unlock()
				return false
			}

			predictedCtx = effectiveLlamaServerContext(req.opts.NumCtx, f, numParallel)

			// Placement and batch size are chosen from a first estimate, and both change
			// how much the load costs. Measured on Qwen3.8-Flash-Next at 256k, where the
			// model and context terms are identical across all three and every difference
			// is compute buffers:
			//
			//	1 device, -b 512    86.35 GiB
			//	1 device, -b 2048   91.65 GiB   batch:        +5.30
			//	2 devices, -b 2048 108.32 GiB   device count: +16.67
			//
			// So the key this load is filed under cannot be settled until both are, and a
			// probe run before them would measure a different load than the one that runs.
			bootstrapKey := vramCalibrationKey(req, gpus, numParallel)
			predicted := predictLlamaServerVRAM(s.vramCalibration, bootstrapKey, req, f, predictedCtx)
			var probed bool
			// The one measurement settlePlacement takes to decide the placement, kept so the
			// full measurement of that same invocation does not take it again.
			var decided struct {
				key  llm.CalibrationKey
				ctx  int
				vram uint64 // 0 when the probe ran and could not measure
				ran  bool
			}
			loadGpus, launchOpts, calibrationKey, predicted, probed = settlePlacement(predicted,
				func(estimate uint64) ([]ml.DeviceInfo, api.Options) {
					return selectLlamaServerPlacement(systemInfo, gpus, estimate, req.opts)
				},
				func(placed []ml.DeviceInfo, opts api.Options, estimate uint64, decideOnly bool) (api.Options, llm.CalibrationKey, uint64, bool) {
					availableForBatch, _, _ := availableMemoryForPlacement(systemInfo, placed, opts)
					req.applyAutomaticGenerationBatch(completion, predictedCtx, estimate, availableForBatch, llm.LlamaServerFlashAttention(placed), placed)
					opts.NumBatch = req.opts.NumBatch

					// Now the invocation is fully determined, so this key describes the load
					// that will actually run, and the measurement it produces is filed where
					// the next prediction for the same invocation will look.
					key := vramCalibrationKey(req, placed, numParallel)

					// A load with no GPU has no placement to decide and nothing a device
					// probe can tell it. Probing anyway ran on the CPU, hit the 30 s probe
					// timeout twice, and recorded nothing, so every load of a CPU-pinned
					// model paid a minute (measured: 62.8 s to load gemma4:e4b with
					// num_gpu 0, 60 of it probing).
					if len(placed) == 0 {
						return opts, key, predictLlamaServerVRAM(s.vramCalibration, key, req, f, predictedCtx), false
					}

					if decideOnly {
						// Where the load goes depends only on what it costs at its own context,
						// so that is all this asks. An invocation already calibrated is not
						// probed at all.
						if s.vramCalibration.SampleCount(key) >= 2 {
							return opts, key, predictLlamaServerVRAM(s.vramCalibration, key, req, f, predictedCtx), false
						}
						vram, measuredCtx, ok := s.probeOnce(req.ctx, req, f, opts, placed, numParallel, predictedCtx)
						decided.key, decided.ctx, decided.vram, decided.ran = key, measuredCtx, vram, true
						if !ok {
							return opts, key, predictLlamaServerVRAM(s.vramCalibration, key, req, f, predictedCtx), false
						}
						return opts, key, vram, true
					}

					var premeasured map[int]uint64
					if decided.ran && decided.key == key {
						premeasured = map[int]uint64{decided.ctx: decided.vram}
					}
					probed := s.probeCalibration(req.ctx, key, req, f, opts, placed, numParallel, predictedCtx, premeasured)
					return opts, key, predictLlamaServerVRAM(s.vramCalibration, key, req, f, predictedCtx), probed
				})

			// Unconditionally, not only when the probe ran. The first prediction was made
			// from a key built before the batch was settled, which addresses a different
			// bucket -- so leaving it in place when the probe is skipped means the load is
			// placed from a key that has no samples while the samples sit under the key it
			// will record to. That is the same defect this reordering exists to fix, and
			// it hid behind a source label taken from the settled key: the log read
			// "calibration" about a number that never consulted it. Measured on
			// gemma4:26b at 32k, 32.06 GiB placed against 18.40 GiB used, with two good
			// samples in the store.
			predicted = predictLlamaServerVRAM(s.vramCalibration, calibrationKey, req, f, predictedCtx)

			// The prediction is logged whether or not anything goes wrong with it. It is
			// the input to every placement decision and, until it is written down next to
			// the load it described, the only way to tell a good estimate from a lucky one
			// is to have been watching. Recorded against the settled key, so it can be
			// compared with the sample this load will file under that same key.
			s.warnIfPredictionIsALowerBound(calibrationKey, f, req.model.ModelPath, predictedCtx, predicted)
			predictedForLoad := predicted + generationBatchSurchargeForCompletion(completion, launchOpts.NumBatch)

			// Both figures are logged because only one of them is the prediction. The
			// calibration models what a load holds; the surcharge on top is what the
			// generation batch will additionally need, which the fit decision must cover
			// but which no sample of a loaded model contains. Comparing the bare figure
			// against a completed load makes the predictor look systematically low by the
			// surcharge, and comparing the total against a sample makes it look high.
			_, calibrated := s.vramCalibration.Predict(calibrationKey, predictedCtx, 0, 0)
			estimate := &api.LoadEstimate{
				Predicted:        int64(predicted),
				PredictedForLoad: int64(predictedForLoad),
				Source:           predictionSource(calibrated, probed),
				NumCtx:           predictedCtx,
				NumGPU:           len(loadGpus),
				NumBatch:         launchOpts.NumBatch,
				MetadataComplete: f.KV().KVCacheModelIsComplete(),
			}
			// The metadata model's view of how the prediction divides, published whatever
			// Source says. That is deliberate: when the total came from a probe or a
			// fitted line the total is measured but the split is not, and it is exactly
			// then that comparing this against what load.complete measured is worth
			// doing. An architecture whose cache the metadata cannot describe shows up as
			// a Weights that matches beside a KVCache that does not -- which a single
			// total can never distinguish from the model simply being bigger than thought.
			if weights, bytesPerToken := llm.PredictServerVRAMParts(req.model.ModelPath, req.model.ProjectorPaths, f); weights > 0 {
				estimate.Breakdown = &api.MemoryBreakdown{
					Weights: int64(weights),
					KVCache: int64(bytesPerToken * uint64(max(predictedCtx, 0))),
				}
			}
			// Emitted before load.start, because this is the decision that chose the
			// devices the load is about to run on. A placement that later spills, or that
			// leaves a second card idle, is otherwise unattributable from the stream.
			loadEstimate = estimate
			s.publishEvent(api.ModelEvent{Type: EventEstimate, Model: req.model.Name, Estimate: estimate})
			slog.Info("predicted llama-server VRAM",
				"model", req.model.ModelPath,
				"architecture", f.KV().Architecture(),
				"num_ctx", predictedCtx,
				"num_batch", launchOpts.NumBatch,
				"num_gpu", len(loadGpus),
				"predicted", predicted,
				"predicted_for_load", predictedForLoad,
				"samples", s.vramCalibration.SampleCount(calibrationKey),
				"source", predictionSource(calibrated, probed),
				"metadata_complete", f.KV().KVCacheModelIsComplete())

			// Pre-flight check: estimate whether the model fits in remaining memory.
			// llama-server auto-detects layers based on available VRAM, so if
			// we predict it won't fit, evict before spawning.
			if requireFull && !explicitPartialGPUOffload(launchOpts, f) && len(s.loaded) > 0 && len(loadGpus) > 0 {
				freeMemory, gpuFreeMemory, systemLimited := availableMemoryForPlacement(systemInfo, loadGpus, launchOpts)
				// Use 80% of free memory as threshold to leave headroom.
				if predictedForLoad > freeMemory*80/100 {
					slog.Info("llama-server model predicted to exceed available memory, evicting",
						"predicted", format.HumanBytes2(predictedForLoad),
						"predicted_num_ctx", predictedCtx,
						"num_batch", launchOpts.NumBatch,
						"available", format.HumanBytes2(freeMemory),
						"gpu_free", format.HumanBytes2(gpuFreeMemory),
						"system_free", format.HumanBytes2(systemInfo.FreeMemory),
						"system_limited", systemLimited)
					s.loadedMu.Unlock()
					return true
				}
				slog.Info("llama-server model fits alongside existing models",
					"predicted", format.HumanBytes2(predictedForLoad),
					"predicted_num_ctx", predictedCtx,
					"num_batch", launchOpts.NumBatch,
					"available", format.HumanBytes2(freeMemory),
					"gpu_free", format.HumanBytes2(gpuFreeMemory),
					"system_free", format.HumanBytes2(systemInfo.FreeMemory),
					"system_limited", systemLimited)
			}

			launchOpts = s.applyLlamaServerMmapDefaults(req, launchOpts, systemInfo, loadGpus, f, numParallel)
			req.contextShift = resolveContextShift(req.shift, req.model)

			config := llamaServerConfigForModel(req.model)
			config.ContextShift = req.contextShift
			// The load begins here. The runner object does not exist until this returns,
			// which is why a client watching /api/ps sees nothing at all for the duration:
			// there is no runner to report yet. The edge has to be emitted from the point
			// work actually starts, or it lands a millisecond before completion.
			loadStartedAt = time.Now().UTC()
			s.loadsInFlight.Add(1)
			s.setLoadingModel(model.ParseName(req.model.Name).DisplayShortest())
			s.publishEvent(api.ModelEvent{Type: EventLoadStart, Model: req.model.Name, At: loadStartedAt})
			loadAnnounced = true

			// A load has two halves and they cost quite differently. Reading and
			// transferring the weights dominates a cold load; the context -- KV cache and
			// compute buffers -- is built afterwards in one step, and on a long-context
			// model that step is most of what the model ends up holding. Reporting the
			// boundary lets a client say which half it is waiting on.
			onWeights := func(at time.Time, weights uint64) {
				weightsLoadedAt = at
				ev := api.ModelEvent{
					Type:       EventLoadWeights,
					Model:      req.model.Name,
					At:         at,
					DurationMs: at.Sub(loadStartedAt).Milliseconds(),
					SizeVRAM:   int64(weights),
				}
				// The breakdown at this instant holds weights and nothing else, which is
				// the point: a client can show the cache arriving afterwards rather than
				// having to infer it from two totals.
				if llama != nil {
					vram, _ := llama.MemoryBreakdownTotals()
					if vram.Total() > 0 {
						ev.Memory = &vram
					}
					ev.WeightsOnDisk = llama.WeightsOnDisk()
				}
				s.publishEvent(ev)
			}

			llama, err = s.newServerFn(systemInfo, loadGpus, req.model.ModelPath, f, req.model.AdapterPaths, req.model.ProjectorPaths, launchOpts, numParallel, config)
			if llama != nil {
				llama.SetOnWeightsLoaded(onWeights)
				// One registration for the life of the runner. The name is captured here
				// rather than read at fire time because the runner outlives this request.
				modelName := req.model.Name
				// Placement is fixed for the life of the runner, so it is captured once here.
				devices, split, numBatch := usageDevices(loadGpus), usageSplit(loadGpus, launchOpts), launchOpts.NumBatch
				runnerLlama := llama
				gpuIDs := make([]ml.DeviceID, 0, len(loadGpus))
				for _, g := range loadGpus {
					gpuIDs = append(gpuIDs, g.DeviceID)
				}
				layers, activeWeights, sliding := 0, 1.0, false
				if f != nil {
					layers, activeWeights = int(f.KV().BlockCount()), activeWeightsFraction(f)
					sliding = f.KV().Uint("attention.sliding_window") > 0
				}
				llama.SetOnGenerationDone(func(t api.GenerationTimings, meta *api.GenerationMeta) {
					timings := t
					ev := api.ModelEvent{Type: EventGenEnd, Model: modelName, Timings: &timings}
					if meta != nil {
						ev.Hint, ev.Shape = meta.Hint, meta.Shape
					}
					s.publishEvent(ev)
					numCtx, numCtxTotal := runnerLlama.GrantedContext()
					row := usageGeneration{
						At: time.Now(), Model: modelName, Timings: timings, Meta: meta,
						Devices: devices, NumCtx: numCtx, NumBatch: numBatch, Split: split,
					}
					// The profile's prediction for this generation, recorded beside what it
					// measured, so the prediction's error is visible per model and placement.
					in := decodeInputs{
						gpus: gpuIDs, fits: s.profileFits(), layers: layers, activeWeights: activeWeights,
						slidingWindow: sliding, grantedCtxTotal: numCtxTotal,
						memByGPU: make(map[ml.DeviceID]api.MemoryBreakdown, len(gpuIDs)),
					}
					for _, d := range gpuIDs {
						in.memByGPU[d] = runnerLlama.MemoryBreakdownByGPU(d)
					}
					total, vram := runnerLlama.MemorySize()
					in.partlyOnCPU = total > vram
					if ms, basis, ok := predictedEvalMs(in, timings.PromptTokens, timings.Decoded); ok {
						row.PredictedEvalMs, row.PredictedBasis = &ms, basis
						slog.Debug("decode speed against the profile's prediction", "model", modelName,
							"predicted_ms_per_token", ms, "measured_ms_per_token", timings.EvalMs/float64(max(timings.Decoded, 1)),
							"basis", basis)
					}
					s.db.recordGeneration(row)
				})
			}
			if err != nil {
				// some older models are not compatible with newer versions of llama.cpp
				// show a generalized compatibility error until there is a better way to
				// check for model compatibility
				if errors.Is(err, ggml.ErrUnsupportedFormat) || strings.Contains(err.Error(), "failed to load model") {
					err = fmt.Errorf("%v: this model may be incompatible with your version of Ollama. If you previously pulled this model, try updating it by running `ollama pull %s`", err, req.model.ShortName)
				}
			}
		} else {
			modelName := req.model.ShortName
			llama, err = mlxrunner.NewClient(modelName, req.opts.NumCtx)
		}
		if err != nil {
			slog.Info("failed to create server", "model", req.model.ShortName, "error", err)
			req.errCh <- err
			s.loadedMu.Unlock()
			abandon(err, false)
			return false
		}

		s.activeLoading = llama
		s.loadingPID.Store(int64(llama.Pid()))
	} else {
		wantPath := req.model.ModelPath
		if wantPath == "" {
			wantPath = req.model.ShortName
		}
		if s.activeLoading.ModelPath() != wantPath {
			panic(fmt.Errorf("attempting to load different model after eviction (original %v new %v)", s.activeLoading.ModelPath(), wantPath))
		}
	}

	s.loadedMu.Unlock()

	systemTotalMemory := systemInfo.TotalMemory
	systemFreeMemory := systemInfo.FreeMemory
	systemSwapFreeMemory := systemInfo.FreeSwap
	slog.Info("system memory", "total", format.HumanBytes2(systemTotalMemory), "free", format.HumanBytes2(systemFreeMemory), "free_swap", format.HumanBytes2(systemSwapFreeMemory))

	for _, gpu := range loadGpus {
		available := gpu.FreeMemory - envconfig.GpuOverhead() - gpu.MinimumMemory()
		if gpu.FreeMemory < envconfig.GpuOverhead()+gpu.MinimumMemory() {
			available = 0
		}
		slog.Info("gpu memory", "id", gpu.ID, "library", gpu.Library,
			"available", format.HumanBytes2(available),
			"free", format.HumanBytes2(gpu.FreeMemory),
			"minimum", format.HumanBytes2(gpu.MinimumMemory()),
			"overhead", format.HumanBytes2(envconfig.GpuOverhead()))
	}

	gpuIDs, err := llama.Load(req.ctx, systemInfo, loadGpus, requireFull)
	if err != nil {
		if errors.Is(err, llm.ErrLoadRequiredFull) {
			if !requireFull {
				// No other models loaded, yet we still don't fit, so report an error
				slog.Info("model is too large for system memory", "requireFull", requireFull)
				s.activeLoading.Close()
				s.activeLoading = nil
				req.errCh <- err
				abandon(err, false)
				return false
			}
			// Not an error the client sees: the scheduler evicts and calls load again, which
			// announces a fresh attempt. This one still has to end, or it never does.
			abandon(err, true)
			return true
		}

		slog.Info("Load failed", "model", req.model.ModelPath, "error", err)
		s.activeLoading.Close()
		s.activeLoading = nil

		s.loadedMu.Lock()
		loadedCount := len(s.loaded)
		s.loadedMu.Unlock()
		otherLoaded := loadedCount > 0
		if !req.oomRetryAttempted && llm.IsOutOfMemory(err) {
			if oldNumCtx, effectiveNumCtx, newNumCtx, oldNumBatch, newNumBatch, ok := req.reduceAutoNumCtxForLoadOOM(s.vramCalibration, f, numParallel, completion, systemInfo, loadGpus, launchOpts); ok {
				req.oomRetryAttempted = true
				slog.Warn("llama-server load failed; reducing automatic context and retrying once",
					"model", req.model.ModelPath,
					"old_num_ctx", oldNumCtx,
					"effective_num_ctx", effectiveNumCtx,
					"new_num_ctx", newNumCtx,
					"old_num_batch", oldNumBatch,
					"new_num_batch", newNumBatch,
					"loaded_count", loadedCount,
					"evict_all", otherLoaded,
					"error", err)
				abandon(err, true)
				return true
			}
		}
		if otherLoaded && !req.oomRetryAttempted && llm.IsOutOfMemory(err) {
			req.oomRetryAttempted = true
			slog.Warn("llama-server load failed; evicting all other models and retrying once", "model", req.model.ModelPath, "error", err)
			abandon(err, true)
			return true
		}

		req.errCh <- err
		abandon(err, false)
		return false
	}
	logTemplateSelection(req.model)

	// Determine if we have discrete GPUs which we should monitor VRAM usage on during shutdown
	discreteGPUs := false
iGPUScan:
	for _, devid := range gpuIDs {
		for _, dev := range loadGpus {
			if dev.DeviceID == devid {
				if !dev.Integrated {
					discreteGPUs = true
					break iGPUScan
				}
			}
		}
	}

	totalSize, vramSize := llama.MemorySize()
	trainContext := modelTrainContext(f)
	if effectiveNumCtx := llama.ContextLength(); req.model.ModelPath != "" && effectiveNumCtx > 0 {
		req.opts.NumCtx = effectiveNumCtx
		req.contextShift = resolveContextShift(req.shift, req.model)
	}
	runner := &runnerRef{
		model:           req.model,
		modelPath:       req.model.ModelPath,
		modelKey:        schedulerModelKey(req.model),
		llama:           llama,
		Options:         &req.opts,
		sessionDuration: sessionDuration,
		gpus:            gpuIDs,
		discreteGPUs:    discreteGPUs,
		totalSize:       totalSize,
		vramSize:        vramSize,
		loading:         true,
		pid:             llama.Pid(),
		numCtxAuto:      req.numCtxAuto,
		numBatchAuto:    req.numBatchAuto,
		useMMapAuto:     req.useMMapAuto,
		contextShift:    req.contextShift,
		trainContext:    trainContext,
	}
	runner.name = req.model.Name
	runner.bandwidthByGPU = make(map[ml.DeviceID]uint64, len(loadGpus))
	runner.pciByGPU = make(map[ml.DeviceID]string, len(loadGpus))
	for _, dev := range loadGpus {
		runner.bandwidthByGPU[dev.DeviceID] = dev.MemoryBandwidth()
		runner.pciByGPU[dev.DeviceID] = dev.PCIID
	}
	if f != nil {
		runner.expertCount = int(f.KV().Uint("expert_count"))
		runner.slidingWindow = f.KV().Uint("attention.sliding_window") > 0
		runner.layers = int(f.KV().BlockCount())
		runner.activeWeights = activeWeightsFraction(f)
	}
	runner.stillLoading.Store(true)
	runner.numParallel = numParallel
	runner.calibrationKey = calibrationKey
	runner.calibrationCtx = predictedCtx
	runner.loadStarted = loadStartedAt
	runner.weightsLoaded = weightsLoadedAt
	if runner.loadStarted.IsZero() {
		runner.loadStarted = time.Now()
	}
	runner.refMu.Lock() // hold lock until running or aborted

	s.loadedMu.Lock()
	if oldRunner, ok := s.loaded[runner.modelKey]; ok {
		// Shouldn't happen, but safeguard against leaking a runner
		slog.Warn("model was still loaded", "old_runner", oldRunner, "new_runner", runner)
		oldRunner.refMu.Lock()
		oldRunner.unload()
		oldRunner.refMu.Unlock()
	}
	s.activeLoading = nil
	s.loaded[runner.modelKey] = runner
	slog.Info("loaded runners", "count", len(s.loaded))
	s.loadedMu.Unlock()

	// From here the goroutine owns the attempt's outcome: it publishes load.complete or
	// load.failed and undoes the loading state either way.
	loadSettled = true
	go func() {
		defer runner.refMu.Unlock()
		if err = llama.WaitUntilRunning(req.ctx); err != nil {
			slog.Error("error loading llama server", "error", err)
			s.loadsInFlight.Add(-1)
			s.clearLoadingModel()
			s.publishEvent(api.ModelEvent{
				Type:       EventLoadFailed,
				Model:      req.model.Name,
				DurationMs: time.Since(runner.loadStarted).Milliseconds(),
				Reason:     err.Error(),
			})
			req.errCh <- err
			slog.Debug("triggering expiration for failed load", "runner", runner)
			s.expiredCh <- runner
			return
		}
		slog.Debug("finished setting up", "runner", runner)
		if runner.pid < 0 {
			runner.pid = llama.Pid()
		}
		runner.refCount++
		if runner.refCount == 1 {
			s.publishEvent(api.ModelEvent{Type: EventBusyStart, Model: req.model.Name})
			s.publishEvent(api.ModelEvent{Type: EventGenStart, Model: req.model.Name})
		}
		runner.loading = false
		runner.stillLoading.Store(false)
		// Publish a first reading while the lock is still held, so a reader arriving
		// before any successful read has something to report rather than nothing.
		runner.reportLocked()
		// The load has finished, so llama-server has reported every buffer it allocated.
		// Remember what it came to: the next load of this model made from the same inputs
		// is predicted from this rather than from metadata.
		loadedTotal, loadedVRAM := llama.MemorySize()

		// The total, not the device figure. They are the same number for a load that fit
		// entirely on the GPU -- MemorySize collapses them -- and they differ exactly when
		// it did not, which is the case that matters. llama-server re-fits at load time
		// against the memory actually free, so a load this predicted too low for does not
		// fail: it quietly offloads fewer layers. Recording the device figure then files
		// the spilled fraction as though it were the model's cost, which is wrong in the
		// direction that causes the next load to spill too -- measured on mistral:7b at 8k,
		// 1.45 GiB recorded for a model that needs 5.23.
		//
		// The total is what the load would have taken with room for it: 5.37 GiB against
		// 5.23 actually measured on a full offload, 2.7% apart. So a spill is not a lost
		// sample but a good one, taken at the moment memory ran out.
		// Filed against the context the memory was actually allocated for. The calibration
		// line maps context to memory, so an x-value that is 199 tokens short of the cache
		// it describes bends the line -- by slope x 199, which on a model holding 1 MiB of
		// KV per token is 199 MiB, every time an unaligned num_ctx is requested. The total,
		// not per slot: calibration's axis is effectiveModelContext x numParallel.
		recordedCtx := runner.calibrationCtx
		if _, grantedTotal := llama.GrantedContext(); grantedTotal > 0 {
			recordedCtx = grantedTotal
		}
		// Only a figure the engine reported. With no buffer lines parsed, MemorySize falls
		// back to the model file's size, and filing that as a measurement would put a number
		// that never came from the load into the line every later load is placed from.
		measured := true
		if m, ok := llama.(interface{ MemoryMeasured() bool }); ok {
			measured = m.MemoryMeasured()
		}
		if loadedTotal > 0 && !measured {
			slog.Warn("not recording this load's memory: the engine reported no buffer sizes, so the figure is the file size",
				"model", req.model.ModelPath)
		}
		if loadedTotal > 0 && measured {
			s.recordCalibration(runner.calibrationKey, recordedCtx, loadedTotal, "load")
		}
		s.loadsInFlight.Add(-1)
		s.clearLoadingModel()
		// A spill is reported rather than merely recorded. It is the visible consequence of
		// a prediction that was too low, and it is otherwise silent: the load succeeds, the
		// model answers, and the only trace is that it is slow.
		if loadedTotal > loadedVRAM {
			slog.Warn("model did not fit and is running partly on the CPU",
				"model", req.model.ModelPath,
				"num_ctx", recordedCtx,
				"needed", format.HumanBytes2(loadedTotal),
				"on_device", format.HumanBytes2(loadedVRAM),
				"on_host", format.HumanBytes2(loadedTotal-loadedVRAM))
		}

		complete := api.ModelEvent{
			Type:       EventLoadComplete,
			Model:      req.model.Name,
			DurationMs: time.Since(runner.loadStarted).Milliseconds(),
			SizeVRAM:   int64(loadedVRAM),
			SizeTotal:  int64(loadedTotal),
		}
		// What the load holds, split by what the memory is for. The host side is only
		// non-empty on a spill, and it is what makes one actionable: SizeTotal exceeding
		// SizeVRAM says how much went to the host, and this says what did.
		if vram, host := llama.MemoryBreakdownTotals(); vram.Total() > 0 {
			complete.Memory = &vram
			if host.Total() > 0 {
				complete.MemoryHost = &host
			}
		}
		complete.WeightsOnDisk = llama.WeightsOnDisk()
		complete.Placement = llama.LayerPlacement()
		if !runner.weightsLoaded.IsZero() && !runner.loadStarted.IsZero() {
			complete.WeightsMs = runner.weightsLoaded.Sub(runner.loadStarted).Milliseconds()
			complete.ContextMs = complete.DurationMs - complete.WeightsMs
		}
		s.publishEvent(complete)
		s.db.recordLoad(usageLoad{
			At: time.Now(), Model: req.model.Name, Estimate: loadEstimate,
			Devices: usageDevices(loadGpus), Split: usageSplit(loadGpus, launchOpts),
			SizeVRAM: complete.SizeVRAM, SizeTotal: complete.SizeTotal,
			WeightsMs: complete.WeightsMs, ContextMs: complete.ContextMs, TotalMs: complete.DurationMs,
		})
		go func() {
			<-req.ctx.Done()
			slog.Debug("context for request finished")
			s.finishedReqCh <- req
		}()
		req.successCh <- runner
	}()

	return false
}

func (req *LlmRequest) reduceAutoNumCtxForLoadOOM(cal *llm.VRAMCalibration, f *ggml.GGML, numParallel int, completion bool, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, launchOpts api.Options) (oldNumCtx, effectiveNumCtx, newNumCtx, oldNumBatch, newNumBatch int, ok bool) {
	if !req.numCtxAuto {
		return 0, 0, 0, 0, 0, false
	}

	oldNumCtx = req.opts.NumCtx
	oldNumBatch = req.opts.NumBatch
	effectiveNumCtx = oldNumCtx
	if f != nil {
		if trainCtx := int(f.KV().ContextLength()); trainCtx > 0 && effectiveNumCtx > trainCtx {
			effectiveNumCtx = trainCtx
		}
	}

	newNumCtx, ok = nextLowerAutoNumCtx(effectiveNumCtx)
	if !ok || newNumCtx >= oldNumCtx {
		return 0, 0, 0, 0, 0, false
	}

	req.opts.NumCtx = newNumCtx
	predictedCtx := effectiveLlamaServerContext(req.opts.NumCtx, f, numParallel)
	predictedVRAM := predictLlamaServerVRAM(cal, vramCalibrationKey(req, gpus, numParallel), req, f, predictedCtx)
	available, _, _ := availableMemoryForPlacement(systemInfo, gpus, launchOpts)
	req.applyAutomaticGenerationBatch(completion, predictedCtx, predictedVRAM, available, llm.LlamaServerFlashAttention(gpus), gpus)
	newNumBatch = req.opts.NumBatch
	return oldNumCtx, effectiveNumCtx, newNumCtx, oldNumBatch, newNumBatch, true
}

func explicitPartialGPUOffload(opts api.Options, f *ggml.GGML) bool {
	if opts.NumGPU <= 0 || f == nil {
		return false
	}

	return uint64(opts.NumGPU) < f.KV().BlockCount()+1
}

func effectiveLlamaServerContext(numCtx int, f *ggml.GGML, numParallel int) int {
	return effectiveModelContext(numCtx, f) * max(numParallel, 1)
}

const (
	llamaServerGenerationBatchDefault     = 512
	llamaServerGenerationBatchConstrained = 256
	llamaServerGenerationBatchMedium      = 1024
	llamaServerGenerationBatchLarge       = 2048

	llamaServerGenerationBatchMediumHeadroomPercent = 75
	llamaServerGenerationBatchLargeHeadroomPercent  = 60
)

func (req *LlmRequest) applyAutomaticGenerationBatch(completion bool, effectiveCtx int, predictedVRAM, availableMemory uint64, flashAttention ml.FlashAttentionType, gpus []ml.DeviceInfo) {
	if !completion || !req.numBatchAuto {
		return
	}

	req.opts.NumBatch = automaticGenerationBatch(effectiveCtx, predictedVRAM, availableMemory, flashAttention, gpus)
}

func generationBatchSurchargeForCompletion(completion bool, batch int) uint64 {
	if !completion {
		return 0
	}
	return generationBatchSurcharge(batch)
}

func automaticGenerationBatch(effectiveCtx int, predictedVRAM, availableMemory uint64, flashAttention ml.FlashAttentionType, gpus []ml.DeviceInfo) int {
	if flashAttention == ml.FlashAttentionDisabled && hasCUDADevice(gpus) {
		if constrainedCUDAWithoutFlashAttention(effectiveCtx, gpus) {
			return llamaServerGenerationBatchConstrained
		}
		return llamaServerGenerationBatchDefault
	}

	batch := generationBatchForContext(effectiveCtx)
	for batch > llamaServerGenerationBatchDefault && !generationBatchFits(batch, predictedVRAM, availableMemory) {
		batch = nextLowerGenerationBatch(batch)
	}
	return batch
}

func hasCUDADevice(gpus []ml.DeviceInfo) bool {
	return slices.ContainsFunc(gpus, func(gpu ml.DeviceInfo) bool {
		return gpu.Library == "CUDA"
	})
}

func constrainedCUDAWithoutFlashAttention(effectiveCtx int, gpus []ml.DeviceInfo) bool {
	if effectiveCtx <= 4096 {
		return false
	}
	return slices.ContainsFunc(gpus, func(gpu ml.DeviceInfo) bool {
		if gpu.Library != "CUDA" {
			return false
		}
		memory := gpu.FreeMemory
		if memory == 0 || (gpu.TotalMemory > 0 && gpu.TotalMemory < memory) {
			memory = gpu.TotalMemory
		}
		return memory > 0 && memory <= 8*format.GibiByte
	})
}

func generationBatchForContext(effectiveCtx int) int {
	switch {
	case effectiveCtx > 32768:
		return llamaServerGenerationBatchLarge
	case effectiveCtx > 4096:
		return llamaServerGenerationBatchMedium
	default:
		return llamaServerGenerationBatchDefault
	}
}

func generationBatchFits(batch int, predictedVRAM, availableMemory uint64) bool {
	if predictedVRAM == 0 || availableMemory == 0 {
		return true
	}

	threshold := availableMemory * 80 / 100
	if predictedVRAM > threshold {
		return false
	}
	if !generationBatchHasHeadroom(batch, predictedVRAM, availableMemory) {
		return false
	}

	return generationBatchSurcharge(batch) <= threshold-predictedVRAM
}

func generationBatchHasHeadroom(batch int, predictedVRAM, availableMemory uint64) bool {
	switch {
	case batch >= llamaServerGenerationBatchLarge:
		return predictedVRAM <= availableMemory*llamaServerGenerationBatchLargeHeadroomPercent/100
	case batch >= llamaServerGenerationBatchMedium:
		return predictedVRAM <= availableMemory*llamaServerGenerationBatchMediumHeadroomPercent/100
	default:
		return true
	}
}

func nextLowerGenerationBatch(batch int) int {
	switch {
	case batch > llamaServerGenerationBatchMedium:
		return llamaServerGenerationBatchMedium
	default:
		return llamaServerGenerationBatchDefault
	}
}

func generationBatchSurcharge(batch int) uint64 {
	switch {
	case batch >= llamaServerGenerationBatchLarge:
		return 2 * format.GibiByte
	case batch >= llamaServerGenerationBatchMedium:
		return 768 * format.MebiByte
	default:
		return 0
	}
}

func nextLowerAutoNumCtx(numCtx int) (int, bool) {
	switch {
	case numCtx > 32768:
		return 32768, true
	case numCtx > 4096:
		return 4096, true
	default:
		return 0, false
	}
}

func availableMemoryForLoad(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo) (available, gpuFree uint64, systemLimited bool) {
	var sharedGPUFree uint64
	var discreteGPUFree uint64
	for _, gpu := range gpus {
		gpuFree += gpu.FreeMemory
		if gpu.Integrated {
			sharedGPUFree += gpu.FreeMemory
		} else {
			discreteGPUFree += gpu.FreeMemory
		}
	}

	// On iGPUs, GPU free memory can be a static or slowly refreshed device
	// baseline. updateFreeSpace has already subtracted known Ollama runner
	// allocations from that baseline. Current system free memory is a separate
	// live measurement that already includes those loaded runners, so use the
	// smaller value for shared-memory GPUs without discounting discrete VRAM.
	if systemInfo.FreeMemory > 0 && sharedGPUFree > 0 && systemInfo.FreeMemory < sharedGPUFree {
		return discreteGPUFree + systemInfo.FreeMemory, gpuFree, true
	}

	return gpuFree, gpuFree, false
}

func availableMemoryForPlacement(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, opts api.Options) (available, gpuFree uint64, systemLimited bool) {
	placementGpus := gpusForPlacement(gpus, opts)
	if len(placementGpus) == 1 && opts.MainGPU != nil {
		gpuFree = placementGpus[0].FreeMemory
		available = availableMemoryForGPU(systemInfo, placementGpus[0])
		systemLimited = available < gpuFree
		return available, gpuFree, systemLimited
	}

	return availableMemoryForLoad(systemInfo, placementGpus)
}

func gpusForPlacement(gpus []ml.DeviceInfo, opts api.Options) []ml.DeviceInfo {
	if opts.MainGPU != nil && *opts.MainGPU >= 0 && *opts.MainGPU < len(gpus) {
		return []ml.DeviceInfo{gpus[*opts.MainGPU]}
	}

	return gpus
}

func selectLlamaServerPlacement(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, predictedVRAM uint64, opts api.Options) ([]ml.DeviceInfo, api.Options) {
	launchOpts := opts
	if len(gpus) <= 1 || opts.NumGPU == 0 {
		return gpus, launchOpts
	}

	groups := ml.ByLibrary(gpus)
	if len(groups) == 0 {
		return gpus, launchOpts
	}

	if opts.MainGPU != nil {
		gpu, available, ok := bestExplicitMainGPU(systemInfo, groups, *opts.MainGPU)
		if !ok {
			selected := bestGPUGroupByAvailableMemory(systemInfo, groups)
			slog.Warn("requested main_gpu is outside the selected GPU group; passing value through to llama-server",
				"main_gpu", *opts.MainGPU,
				"gpu_count", len(selected))
			logSelectedGPUGroup(gpus, selected)
			return selected, launchOpts
		}

		selected, launchOpts := singleLlamaServerGPUPlacement(gpu, launchOpts)
		slog.Info("selecting requested single GPU for llama-server model",
			"requested_main_gpu", *opts.MainGPU,
			"main_gpu", *launchOpts.MainGPU,
			"id", gpu.ID,
			"filter_id", gpu.FilterID,
			"library", gpu.Library,
			"name", gpu.Name,
			"description", gpu.Description,
			"integrated", gpu.Integrated,
			"available", format.HumanBytes2(available))
		logSelectedGPUGroup(gpus, selected)
		return selected, launchOpts
	}

	if !envconfig.SchedSpread() && predictedVRAM > 0 {
		gpu, available, ok := bestSingleGPUFit(systemInfo, groups, predictedVRAM)
		if ok {
			selected, launchOpts := singleLlamaServerGPUPlacement(gpu, launchOpts)
			slog.Info("selecting single GPU for llama-server model",
				"main_gpu", *launchOpts.MainGPU,
				"id", gpu.ID,
				"filter_id", gpu.FilterID,
				"library", gpu.Library,
				"name", gpu.Name,
				"description", gpu.Description,
				"integrated", gpu.Integrated,
				"predicted", format.HumanBytes2(predictedVRAM),
				"available", format.HumanBytes2(available))
			logSelectedGPUGroup(gpus, selected)
			return selected, launchOpts
		}
	}

	selected := bestGPUGroupByAvailableMemory(systemInfo, groups)
	logSelectedGPUGroup(gpus, selected)
	return selected, launchOpts
}

func singleLlamaServerGPUPlacement(gpu ml.DeviceInfo, opts api.Options) ([]ml.DeviceInfo, api.Options) {
	mainGPU := 0
	opts.MainGPU = &mainGPU
	return []ml.DeviceInfo{gpu}, opts
}

func bestExplicitMainGPU(systemInfo ml.SystemInfo, groups [][]ml.DeviceInfo, mainGPU int) (gpu ml.DeviceInfo, available uint64, ok bool) {
	if mainGPU < 0 {
		return ml.DeviceInfo{}, 0, false
	}

	for _, group := range groups {
		if mainGPU >= len(group) {
			continue
		}
		candidate := group[mainGPU]
		candidateAvailable := availableMemoryForGPU(systemInfo, candidate)
		if !ok || betterPlacementGPU(candidate, candidateAvailable, gpu, available) {
			gpu = candidate
			available = candidateAvailable
			ok = true
		}
	}

	return gpu, available, ok
}

func bestSingleGPUFit(systemInfo ml.SystemInfo, groups [][]ml.DeviceInfo, predictedVRAM uint64) (gpu ml.DeviceInfo, available uint64, ok bool) {
	for _, group := range groups {
		for _, candidate := range group {
			candidateAvailable := availableMemoryForGPU(systemInfo, candidate)
			if predictedVRAM > candidateAvailable*uint64(envconfig.SingleGPUFitPercent())/100 {
				continue
			}
			if !ok || betterPlacementGPU(candidate, candidateAvailable, gpu, available) {
				gpu = candidate
				available = candidateAvailable
				ok = true
			}
		}
	}

	return gpu, available, ok
}

func betterPlacementGPU(candidate ml.DeviceInfo, candidateAvailable uint64, current ml.DeviceInfo, currentAvailable uint64) bool {
	if candidate.Integrated != current.Integrated {
		return !candidate.Integrated
	}

	return candidateAvailable > currentAvailable
}

func bestGPUGroupByAvailableMemory(systemInfo ml.SystemInfo, groups [][]ml.DeviceInfo) []ml.DeviceInfo {
	var best []ml.DeviceInfo
	var bestAvailable uint64
	for _, group := range groups {
		available, _, _ := availableMemoryForLoad(systemInfo, group)
		if best == nil || betterPlacementGroup(group, available, best, bestAvailable) {
			best = group
			bestAvailable = available
		}
	}

	return best
}

func betterPlacementGroup(candidate []ml.DeviceInfo, candidateAvailable uint64, current []ml.DeviceInfo, currentAvailable uint64) bool {
	candidateDiscrete := hasDiscreteGPU(candidate)
	currentDiscrete := hasDiscreteGPU(current)
	if candidateDiscrete != currentDiscrete {
		return candidateDiscrete
	}

	return candidateAvailable > currentAvailable
}

func hasDiscreteGPU(gpus []ml.DeviceInfo) bool {
	for _, gpu := range gpus {
		if !gpu.Integrated {
			return true
		}
	}
	return false
}

func availableMemoryForGPU(systemInfo ml.SystemInfo, gpu ml.DeviceInfo) uint64 {
	if gpu.Integrated && systemInfo.FreeMemory > 0 && systemInfo.FreeMemory < gpu.FreeMemory {
		return systemInfo.FreeMemory
	}

	return gpu.FreeMemory
}

func logSelectedGPUGroup(all, selected []ml.DeviceInfo) {
	if len(selected) == 0 || len(selected) == len(all) {
		return
	}

	slog.Info("selecting GPU backend for llama-server model",
		"library", selected[0].Library,
		"gpu_count", len(selected),
		"available_gpu_count", len(all))
}

func (s *Scheduler) applyLlamaServerMmapDefaults(req *LlmRequest, launchOpts api.Options, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, f *ggml.GGML, numParallel int) api.Options {
	predictedCtx := effectiveLlamaServerContext(req.opts.NumCtx, f, numParallel)
	predictedVRAM := predictLlamaServerVRAM(s.vramCalibration, vramCalibrationKey(req, gpus, numParallel), req, f, predictedCtx)
	availableVRAM, _, _ := availableMemoryForPlacement(systemInfo, gpus, launchOpts)

	if reason := disableMmapDefaultReason(runtime.GOOS, req.opts, gpus, f.KV().BlockCount(), predictedVRAM, availableVRAM); reason != "" {
		useMmap := false
		req.opts.UseMMap = &useMmap
		req.useMMapAuto = true
		slog.Info("disabling mmap for llama-server load by default",
			"model", req.model.ModelPath,
			"reason", reason)
	} else {
		s.maybeDisableMmapForHostPressure(req, launchOpts, systemInfo, gpus, f, numParallel)
	}

	launchOpts.UseMMap = req.opts.UseMMap
	return launchOpts
}

func disableMmapDefaultReason(goos string, opts api.Options, gpus []ml.DeviceInfo, blockCount, predictedVRAM, availableVRAM uint64) string {
	if opts.UseMMap != nil {
		return ""
	}
	if opts.NumGPU == 0 || len(gpus) == 0 || allDevicesLibrary(gpus, "cpu") {
		return "cpu"
	}
	if goos == "windows" && hasDeviceLibrary(gpus, "cuda") {
		return "windows_cuda"
	}
	if hasDeviceLibrary(gpus, "metal") {
		if opts.NumGPU > 0 && blockCount > 0 && uint64(opts.NumGPU) < blockCount+1 {
			return "metal_partial_offload"
		}
		if opts.NumGPU < 0 && predictedVRAM > 0 && availableVRAM > 0 && predictedVRAM > availableVRAM {
			return "metal_partial_offload"
		}
	}
	return ""
}

func hasDeviceLibrary(gpus []ml.DeviceInfo, library string) bool {
	for _, gpu := range gpus {
		if strings.EqualFold(gpu.Library, library) {
			return true
		}
	}
	return false
}

func allDevicesLibrary(gpus []ml.DeviceInfo, library string) bool {
	if len(gpus) == 0 {
		return false
	}
	for _, gpu := range gpus {
		if !strings.EqualFold(gpu.Library, library) {
			return false
		}
	}
	return true
}

func (s *Scheduler) maybeDisableMmapForHostPressure(req *LlmRequest, launchOpts api.Options, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, f *ggml.GGML, numParallel int) {
	modelSize := modelFileSize(req.model.ModelPath)
	loadedMmapSize := s.loadedMmapModelSizeLocked()
	predictedCtx := effectiveLlamaServerContext(req.opts.NumCtx, f, numParallel)
	predictedVRAM := predictLlamaServerVRAM(s.vramCalibration, vramCalibrationKey(req, gpus, numParallel), req, f, predictedCtx)
	availableVRAM, _, _ := availableMemoryForPlacement(systemInfo, gpus, launchOpts)
	placementGpus := gpusForPlacement(gpus, launchOpts)

	if !disableMmapForHostPressure(runtime.GOOS, req.opts, systemInfo, placementGpus, modelSize, loadedMmapSize, predictedVRAM, availableVRAM) {
		return
	}

	useMmap := false
	req.opts.UseMMap = &useMmap
	req.useMMapAuto = true
	slog.Info("disabling mmap for llama-server load due to host memory pressure",
		"model", req.model.ModelPath,
		"model_size", format.HumanBytes2(modelSize),
		"loaded_mmap_size", format.HumanBytes2(loadedMmapSize),
		"headroom", format.HumanBytes2(mmapHostPressureHeadroom(systemInfo.TotalMemory)),
		"system_free", format.HumanBytes2(systemInfo.FreeMemory),
		"system_total", format.HumanBytes2(systemInfo.TotalMemory),
		"predicted_vram", format.HumanBytes2(predictedVRAM),
		"available_vram", format.HumanBytes2(availableVRAM),
	)
}

func disableMmapForHostPressure(goos string, opts api.Options, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, modelSize, loadedMmapSize, predictedVRAM, availableVRAM uint64) bool {
	if opts.UseMMap != nil || goos != "linux" || modelSize == 0 || systemInfo.FreeMemory == 0 || !allDiscreteGPUs(gpus) {
		return false
	}

	// Only back off mmap when we still expect the model to fit on discrete GPU.
	// If VRAM is already tight, disabling mmap can make partial CPU offload
	// worse by turning file-backed mappings into anonymous memory.
	if predictedVRAM == 0 || availableVRAM == 0 || predictedVRAM > availableVRAM*80/100 {
		return false
	}

	pressure := modelSize + loadedMmapSize + mmapHostPressureHeadroom(systemInfo.TotalMemory)
	return systemInfo.FreeMemory < pressure
}

func allDiscreteGPUs(gpus []ml.DeviceInfo) bool {
	if len(gpus) == 0 {
		return false
	}
	for _, gpu := range gpus {
		if gpu.Integrated {
			return false
		}
	}
	return true
}

func mmapHostPressureHeadroom(totalMemory uint64) uint64 {
	if totalMemory == 0 {
		return 8 * format.GigaByte
	}
	return max(8*format.GigaByte, totalMemory/10)
}

func modelFileSize(path string) uint64 {
	if path == "" {
		return 0
	}
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return uint64(info.Size())
}

func (s *Scheduler) loadedMmapModelSizeLocked() uint64 {
	var total uint64
	for _, r := range s.loaded {
		if !runnerUsesMmap(r) {
			continue
		}
		if size := modelFileSize(r.modelPath); size > 0 {
			total += size
		} else {
			total += r.totalSize
		}
	}
	return total
}

func runnerUsesMmap(r *runnerRef) bool {
	if r == nil || r.Options == nil || r.Options.UseMMap == nil {
		return true
	}
	return *r.Options.UseMMap
}

func (s *Scheduler) updateFreeSpace(allGpus []ml.DeviceInfo) {
	if len(allGpus) == 0 {
		return
	}
	predMap := map[ml.DeviceID]uint64{} // Sum up the total predicted usage per GPU for all runners
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()
	for _, r := range runners {
		r.refMu.Lock()
		if r.llama != nil {
			for _, gpu := range allGpus {
				predMap[gpu.DeviceID] += r.llama.VRAMByGPU(gpu.DeviceID)
			}
		} else {
			slog.Warn("unexpected nil runner reference, memory prediction may be incorrect")
		}
		r.refMu.Unlock()
	}

	// Now that we've summed up all the GPU usage predictions across all the loaded runners, update the gpu list
	for i := range allGpus {
		if p, ok := predMap[allGpus[i].DeviceID]; ok {
			slog.Debug("gpu reported", "gpu", allGpus[i].ID, "library", allGpus[i].Library, "available", format.HumanBytes2(allGpus[i].FreeMemory))
			if p > allGpus[i].TotalMemory {
				// Shouldn't happen
				slog.Warn("predicted usage exceeds VRAM", "gpu", allGpus[i].ID, "totalMemory", allGpus[i].TotalMemory, "predicted", p)
				allGpus[i].FreeMemory = 0
			} else if (allGpus[i].TotalMemory - p) < allGpus[i].FreeMemory { // predicted free is smaller than reported free, use it
				// TODO maybe we should just always trust our numbers, since cuda's free memory reporting is laggy
				// and we might unload models we didn't actually need to.  The risk is if some other GPU intensive app is loaded
				// after we start our first runner, then we'll never account for that, so picking the smallest free value seems prudent.
				allGpus[i].FreeMemory = allGpus[i].TotalMemory - p
			}
			slog.Info("updated VRAM based on existing loaded models", "gpu", allGpus[i].ID, "library", allGpus[i].Library, "total", format.HumanBytes2(allGpus[i].TotalMemory), "available", format.HumanBytes2(allGpus[i].FreeMemory))
		}
	}
}

// TODO consolidate sched_types.go
type runnerRef struct {
	// Fixed at load, for the decode roofline: per-device bandwidth and bus address of the
	// devices this runner occupies, and the two model properties that decide whether a dense
	// bound applies. Captured here so /api/ps never has to run device discovery to answer.
	bandwidthByGPU map[ml.DeviceID]uint64
	pciByGPU       map[ml.DeviceID]string
	expertCount    int
	slidingWindow  bool
	// For the expected decode speed (decode_speed.go): the model's layer count and the share of
	// its weights one token reads.
	layers        int
	activeWeights float64

	refMu    sync.Mutex
	refCount uint // prevent unloading if > 0

	llama llm.LlamaServer
	pid   int

	// calibrationKey and calibrationCtx are the inputs this load's prediction was made
	// from, kept so its measurement is recorded under the key a later prediction from the
	// same inputs will look up.
	calibrationKey llm.CalibrationKey
	calibrationCtx int

	// name is fixed when the runner is created and never reassigned, so it can be read
	// while another goroutine holds refMu -- which is the whole point: it lets a loading
	// runner be named without waiting for the load to finish.
	name string

	// weightsLoaded is when this runner's weights reached device memory, which splits the
	// load into its two halves. Zero if the runner never reported the boundary.
	weightsLoaded time.Time

	// loadStarted is when this runner began loading, so load.complete can report how long
	// it took rather than only that it finished.
	loadStarted time.Time

	loading bool // True only during initial load, then false forever

	// stillLoading mirrors loading for readers that cannot take refMu. A load holds that
	// lock for its whole duration, so the only way to ask "is this one still arriving?"
	// without blocking is a field that does not need it.
	stillLoading atomic.Bool

	// lastReported is the most recent complete reading of this runner, kept so a reader
	// that finds refMu busy can report what was true a moment ago instead of reporting
	// zeros. A resident model's figures do not change while its lock is held.
	lastReported atomic.Pointer[loadedModel]
	gpus         []ml.DeviceID // Recorded at time of provisioning
	discreteGPUs bool          // True if all devices are discrete GPUs - used to skip VRAM recovery check for iGPUs
	vramSize     uint64
	totalSize    uint64

	sessionDuration time.Duration
	expireTimer     *time.Timer
	expiresAt       time.Time
	// leaseExpiring is set once a lease has asked for this runner to be unloaded, so the
	// lease's polling does not ask again.
	leaseExpiring bool

	model        *Model
	modelPath    string
	modelKey     string
	numParallel  int
	numCtxAuto   bool
	numBatchAuto bool
	useMMapAuto  bool
	contextShift bool
	trainContext int
	*api.Options
}

// The refMu must already be held when calling unload
func (runner *runnerRef) unload() {
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	if runner.llama != nil {
		runner.llama.Close()
	}
	runner.model = nil
	runner.Options = nil
	runner.gpus = nil
	runner.contextShift = false
}

func (runner *runnerRef) needsReload(ctx context.Context, req *LlmRequest) bool {
	slog.Debug("evaluating already loaded", "model", schedulerModelKey(req.model))
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	timeout := 10 * time.Second
	if runner.loading {
		timeout = 2 * time.Minute // Initial load can take a long time for big models on slow systems...
	}

	if runner.Options == nil {
		return true
	}

	// Don't reload runner if num_gpu=-1 was provided
	optsExisting := runner.Options.Runner
	optsNew := req.opts.Runner
	optsNew.NumCtx = effectiveContext(optsNew.NumCtx, runner.trainContext)
	if runner.numCtxAuto && req.numCtxAuto {
		optsNew.NumCtx = optsExisting.NumCtx
	}
	if runner.numBatchAuto && req.numBatchAuto {
		optsNew.NumBatch = optsExisting.NumBatch
	}
	if runner.useMMapAuto && optsNew.UseMMap == nil {
		optsNew.UseMMap = optsExisting.UseMMap
	}
	if optsNew.NumGPU < 0 {
		optsExisting.NumGPU = -1
		optsNew.NumGPU = -1
	}

	contextShift := req.contextShift
	if req.model.ModelPath != "" {
		contextShift = resolveContextShift(req.shift, req.model)
	}
	if runner.contextShift != contextShift {
		return true
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if !reflect.DeepEqual(runner.model.AdapterPaths, req.model.AdapterPaths) || // have the adapters changed?
		!reflect.DeepEqual(runner.model.ProjectorPaths, req.model.ProjectorPaths) || // have the projectors changed?
		(!runner.model.IsMLX() && !reflect.DeepEqual(optsExisting, optsNew)) || // have the runner options changed?
		runner.llama.Ping(ctx) != nil {
		return true
	}

	return false
}

// Free memory reporting on GPUs can lag for a while even after the runner
// exits, so we have to keep checking until we see the available memory recover,
// otherwise subsequent model loads will get far less layers loaded or worse
// case, may completely fall back to CPU mode.
// This routine must be called before the runner unloads so it can establish
// a before and after GPU memory allocation.  The returned channel
// will be notified when we're done waiting, or have timed out and should
// proceed anyway
func (s *Scheduler) waitForVRAMRecovery(runner *runnerRef, runners []ml.FilteredRunnerDiscovery) chan any {
	finished := make(chan any, 1)

	// CPU, Metal and iGPUs don't need checking, so no waiting required
	if len(runner.gpus) == 0 || !runner.discreteGPUs ||
		(len(runner.gpus) == 1 && runner.gpus[0].Library == "Metal") {
		finished <- struct{}{}
		slog.Debug("no need to wait for VRAM recovery", "runner", runner)
		return finished
	}
	start := time.Now()

	// Establish a baseline before we unload
	gpusBefore := s.getGpuFn(context.Background(), runners)
	var totalMemoryBefore, freeMemoryBefore uint64
	for _, gpu := range gpusBefore {
		totalMemoryBefore += gpu.TotalMemory
		freeMemoryBefore += gpu.FreeMemory
	}
	totalMemoryNow := totalMemoryBefore
	freeMemoryNow := freeMemoryBefore

	go func() {
		// typical convergence is 0.5-1.5s - If it takes too long to discover and converge, let the scheduler estimate VRAM usage
		ctx, cancel := context.WithTimeout(context.Background(), s.waitForRecovery)
		defer cancel()
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				// Query GPUs, look for free to go back up
				gpusNow := s.getGpuFn(ctx, runners)
				totalMemoryNow = 0
				freeMemoryNow = 0
				for _, gpu := range gpusNow {
					totalMemoryNow += gpu.TotalMemory
					freeMemoryNow += gpu.FreeMemory
				}
				if freeMemoryNow > freeMemoryBefore {
					logutil.Trace("gpu VRAM convergence", "percent", int(float32(freeMemoryNow-freeMemoryBefore)/float32(runner.vramSize)*100))
				} else {
					logutil.Trace("gpu VRAM convergence", "percent", 0)
				}
				// If we're within ~75% of the estimated memory usage recovered, bail out
				if float32(freeMemoryNow-freeMemoryBefore) > float32(runner.vramSize)*0.75 {
					slog.Debug(fmt.Sprintf("gpu VRAM free memory converged after %0.2f seconds", time.Since(start).Seconds()), "free_before", format.HumanBytes2(freeMemoryBefore), "free_now", format.HumanBytes2(freeMemoryNow), "runner", runner)
					finished <- struct{}{}
					return
				}
			case <-ctx.Done():
				slog.Debug("gpu VRAM usage didn't recover within timeout", "seconds", time.Since(start).Seconds(), "free_before", format.HumanBytes2(freeMemoryBefore), "free_now", format.HumanBytes2(freeMemoryNow), "runner", runner)
				finished <- struct{}{}
				return
			}
		}
	}()
	return finished
}

// LogValue may run from goroutines that already hold refMu (see the scheduler
// debug logs), so the unload-mutable fields are read only under TryLock and
// omitted when the lock is contended.
func (runner *runnerRef) LogValue() slog.Value {
	if runner == nil {
		return slog.StringValue("nil")
	}
	modelID := runner.modelPath
	if modelID == "" {
		modelID = runner.modelKey
	}
	attrs := []slog.Attr{}
	if runner.refMu.TryLock() {
		if runner.model != nil {
			attrs = append(attrs, slog.String("name", runner.model.Name))
		}
		if len(runner.gpus) > 0 {
			attrs = append(attrs,
				slog.Any("inference", slices.Clone(runner.gpus)),
			)
		}
		attrs = append(attrs, slog.Int("pid", runner.pid))
		if runner.Options != nil {
			attrs = append(attrs, slog.Int("num_ctx", runner.Options.NumCtx))
		}
		runner.refMu.Unlock()
	}
	attrs = append(attrs,
		slog.String("size", format.HumanBytes2(runner.totalSize)),
		slog.String("vram", format.HumanBytes2(runner.vramSize)),
		slog.Int("parallel", runner.numParallel),
		slog.String("model", modelID),
	)
	return slog.GroupValue(attrs...)
}

// Implements discover.RunnerDiscovery
func (runner *runnerRef) GetPort() int {
	if runner.llama != nil {
		return runner.llama.GetPort()
	}
	return -1
}

func (runner *runnerRef) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	if runner.llama != nil {
		return runner.llama.GetDeviceInfos(ctx)
	}
	return nil
}

func (runner *runnerRef) GetActiveDeviceIDs() []ml.DeviceID {
	return runner.gpus
}

func (runner *runnerRef) HasExited() bool {
	if runner.llama != nil {
		return runner.llama.HasExited()
	}
	return true
}

type ByDurationAndName []*runnerRef

func (a ByDurationAndName) Len() int      { return len(a) }
func (a ByDurationAndName) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByDurationAndName) Less(i, j int) bool {
	// Primary sort by session duration (uint64 to handle negatives)
	d1 := uint64(a[i].sessionDuration)
	d2 := uint64(a[j].sessionDuration)
	if d1 != d2 {
		return d1 < d2
	}
	// Secondary sort by model key/path lex order
	n1 := a[i].modelPath
	if n1 == "" {
		n1 = a[i].modelKey
	}
	n2 := a[j].modelPath
	if n2 == "" {
		n2 = a[j].modelKey
	}
	return n1 < n2
}

// TODO - future consideration to pick runners based on size
// type BySize []*runnerRef
// func (a BySize) Len() int           { return len(a) }
// func (a BySize) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
// func (a BySize) Less(i, j int) bool { return a[i].vramSize < a[j].vramSize }

// evictAllAndWait synchronously expires every currently loaded runner except
// the one being loaded (matched by modelKey) and waits for all unload events
// to drain. Returns false if the context was cancelled mid-wait so the caller
// can exit the scheduling loop. Used by the OOM retry path in processPending.
func (s *Scheduler) evictAllAndWait(ctx context.Context, keepKey string) bool {
	s.loadedMu.Lock()
	runnersToExpire := make([]*runnerRef, 0, len(s.loaded))
	for key, r := range s.loaded {
		if key == keepKey {
			continue
		}
		runnersToExpire = append(runnersToExpire, r)
	}
	s.loadedMu.Unlock()

	if len(runnersToExpire) == 0 {
		return true
	}

	slog.Info("evicting all other loaded models for OOM retry", "count", len(runnersToExpire))
	for _, r := range runnersToExpire {
		if r.model != nil {
			s.publishEvent(api.ModelEvent{Type: EventEvict, Model: r.model.Name, Reason: "oom-retry"})
		}
	}
	for _, runner := range runnersToExpire {
		runner.refMu.Lock()
		if runner.expireTimer != nil {
			runner.expireTimer.Stop()
			runner.expireTimer = nil
		}
		runner.sessionDuration = 0
		if runner.refCount <= 0 {
			s.expiredCh <- runner
		}
		runner.refMu.Unlock()
	}

	// Wait for every unload event. Each runner produces exactly one
	// unloadedCh signal when its cleanup finishes.
	for range runnersToExpire {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler during evict-all wait")
			return false
		case <-s.unloadedCh:
		}
	}
	return true
}

func (s *Scheduler) expireRunnersForRuntimeOOM(model *Model, err error) {
	if !llm.IsOutOfMemory(err) {
		return
	}

	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, runner := range s.loaded {
		runners = append(runners, runner)
	}
	s.loadedMu.Unlock()

	if len(runners) == 0 {
		return
	}

	slog.Warn("runtime OOM detected; expiring loaded models to clear memory before next request", "model", schedulerModelKey(model), "error", err)
	for _, runner := range runners {
		runner.refMu.Lock()
		if runner.expireTimer != nil {
			runner.expireTimer.Stop()
			runner.expireTimer = nil
		}
		runner.sessionDuration = 0
		if runner.refCount <= 0 {
			s.expiredCh <- runner
		}
		runner.refMu.Unlock()
	}
}

// findRunnerToUnload finds a runner to unload to make room for a new model
func (s *Scheduler) findRunnerToUnload() *runnerRef {
	s.loadedMu.Lock()
	runnerList := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runnerList = append(runnerList, r)
	}
	s.loadedMu.Unlock()
	if len(runnerList) == 0 {
		slog.Debug("no loaded runner to unload")
		return nil
	}

	// In the future we can enhance the algorithm to be smarter about picking the optimal runner to unload
	// e.g., if we have multiple options, will one make room for the request?
	sort.Sort(ByDurationAndName(runnerList))

	// First try to find a runner that's already idle
	for _, runner := range runnerList {
		runner.refMu.Lock()
		rc := runner.refCount
		runner.refMu.Unlock()
		if rc == 0 {
			slog.Debug("found an idle runner to unload", "runner", runner)
			return runner
		}
	}
	// None appear idle, just wait for the one with the shortest duration
	slog.Debug("no idle runners, picking the shortest duration", "runner_count", len(runnerList), "runner", runnerList[0])
	return runnerList[0]
}

func (s *Scheduler) unloadAllRunners() {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()

	if s.activeLoading != nil {
		slog.Debug("shutting down currently loading runner")
		s.activeLoading.Close()
		s.activeLoading = nil
	}

	for model, runner := range s.loaded {
		if runner.llama != nil {
			slog.Debug("shutting down runner", "model", model)
			runner.llama.Close()
		}
	}
}

func (s *Scheduler) expireRunner(model *Model) {
	modelKey := schedulerModelKey(model)
	s.loadedMu.Lock()
	runner, ok := s.loaded[modelKey]
	s.loadedMu.Unlock()
	if ok {
		runner.refMu.Lock()
		runner.expiresAt = time.Now()
		if runner.expireTimer != nil {
			runner.expireTimer.Stop()
			runner.expireTimer = nil
		}
		runner.sessionDuration = 0
		if runner.refCount <= 0 {
			s.expiredCh <- runner
		}
		runner.refMu.Unlock()
	}
}

// loadedModel is a point-in-time snapshot of a loaded runner's state, safe to
// use without holding any scheduler locks.
type loadedModel struct {
	model *Model

	// name and loading describe a runner that could not be inspected because it is still
	// loading. model is nil in that case; everything else is unknown until the load ends.
	name    string
	loading bool

	// busy reports that a request is in flight against this runner. It matters because
	// expiresAt does not move while one is: the deadline is only pushed out when a request
	// finishes, so a countdown drawn from expiresAt during a long generation runs down and
	// past zero for a model that is resident and working.
	busy bool

	size          int64
	sizeVRAM      int64
	contextLength int
	expiresAt     time.Time
	gpus          []ml.DeviceID
	vramByGPU     map[ml.DeviceID]uint64

	// memVRAM splits sizeVRAM by what the memory holds, and memByGPU does the same per
	// device. weightsOnDisk is the size of the files loaded from, which is not memory and
	// so is kept beside the breakdown rather than inside it.
	memVRAM       api.MemoryBreakdown
	memByGPU      map[ml.DeviceID]api.MemoryBreakdown
	weightsOnDisk int64
	placement     *api.ModelPlacement
	activity      *api.RunnerActivity

	// Inputs to the decode roofline; see modelRoofline.
	bandwidthByGPU  map[ml.DeviceID]uint64
	pciByGPU        map[ml.DeviceID]string
	expertCount     int
	slidingWindow   bool
	grantedCtxTotal int
	layers          int
	activeWeights   float64
}

// loadedModels returns a snapshot of the currently loaded models for status
// reporting without exposing the scheduler's internal runner bookkeeping.
// abandonLoad ends a load attempt that will not reach a runner: it undoes the loading row
// and the in-flight count, and publishes the load.failed that pairs with its load.start.
//
// A retry is still reported as a failure, because it is one -- the attempt ended, and the
// next one announces itself with its own load.start. Leaving the first unterminated would
// give a client an edge it can never close.
func (s *Scheduler) abandonLoad(req *LlmRequest, started time.Time, err error, retrying bool) {
	s.loadsInFlight.Add(-1)
	s.clearLoadingModel()

	reason := "load did not complete"
	if err != nil {
		reason = err.Error()
	}
	if retrying {
		reason += "; retrying"
	}
	s.publishEvent(api.ModelEvent{
		Type:       EventLoadFailed,
		Model:      req.model.Name,
		DurationMs: time.Since(started).Milliseconds(),
		Reason:     reason,
	})
}

func (s *Scheduler) setLoadingModel(name string) {
	s.loadingModel.Store(&name)
}

func (s *Scheduler) clearLoadingModel() {
	s.loadingModel.Store(nil)
	s.loadingPID.Store(0)
}

// LoadingModel is the model currently being loaded, or "" if none.
func (s *Scheduler) LoadingModel() string {
	if p := s.loadingModel.Load(); p != nil {
		return *p
	}
	return ""
}

// runnerMark is what /api/info says about a runner process: the model it serves, named as
// /api/ps names it, and whether it is still loading.
type runnerMark struct {
	model   string
	loading bool
}

func displayModelName(name string) string {
	if short := model.ParseName(name).DisplayShortest(); short != "" {
		return short
	}
	return name
}

// runnerPIDs maps each runner's process id to the model it serves: resident runners, and
// the one being loaded. Reads only fields fixed when a runner was created -- llama is never
// reassigned and its process starts in its constructor -- and atomics, so no runner lock is
// taken and a runner busy loading does not stall /api/info.
//
// The loading runner matters most: measured on qwen3.5:0.8b, its process held 14 MB, then
// 1.4 GB, then 5.9 GB on the card before the load returned, and without this it was named
// an anonymous helper for all of it.
func (s *Scheduler) runnerPIDs() map[int]runnerMark {
	s.loadedMu.Lock()
	out := make(map[int]runnerMark, len(s.loaded)+1)
	for _, r := range s.loaded {
		if r.llama == nil {
			continue
		}
		if pid := r.llama.Pid(); pid > 0 {
			out[pid] = runnerMark{model: displayModelName(r.name), loading: r.stillLoading.Load()}
		}
	}
	s.loadedMu.Unlock()

	if pid := int(s.loadingPID.Load()); pid > 0 {
		if _, resident := out[pid]; !resident {
			if name := s.LoadingModel(); name != "" {
				out[pid] = runnerMark{model: displayModelName(name), loading: true}
			}
		}
	}
	return out
}

func (s *Scheduler) loadedModels() []loadedModel {
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()

	// refMu must not be acquired while holding loadedMu: the expiration path
	// locks them in the opposite order.
	models := make([]loadedModel, 0, len(runners))
	for _, r := range runners {
		// A loading runner holds refMu for the whole load, so waiting for it here makes
		// this call block for as long as the load takes -- tens of seconds for a large
		// model, during which a client polling for state gets nothing back at all. Report
		// that the model is loading instead. The name is read from a field fixed at
		// construction rather than from r.model, which is only safe under the lock.
		if !r.refMu.TryLock() {
			// The lock being busy is not the same as the model still loading, and
			// conflating them reported a resident, serving model as "loading" with zeroed
			// size and no devices -- while the same response's vram_used still said 94 GB
			// was held. refMu is taken by ordinary traffic: a request finishing, an
			// expiry being reset, another model being admitted.
			switch snapshot := r.lastReported.Load(); {
			case snapshot != nil && !r.stillLoading.Load():
				// Resident, merely contended. What was last read is still true: a loaded
				// model's figures do not change while its lock is held. Only busy can
				// have moved, and it moves under this same lock.
				models = append(models, *snapshot)
			default:
				// Either genuinely still loading, or loaded so recently that nothing has
				// read it yet -- and in both cases its identity is all that is known.
				models = append(models, loadedModel{name: r.name, loading: true})
			}
			continue
		}
		if r.model == nil {
			// Unloaded after the snapshot above was taken
			r.refMu.Unlock()
			continue
		}
		lm := r.reportLocked()
		r.refMu.Unlock()

		models = append(models, lm)
	}
	return models
}

// reportLocked builds this runner's reportable state and remembers it, so a later reader
// that finds refMu busy has something true to report instead of zeros. refMu must be held.
//
// Storing it here rather than only in the reader matters: a runner that has just finished
// loading has never been read, so without this the first reader to find its lock busy has
// no snapshot and drops the row entirely -- the model disappears from /api/ps rather than
// merely reading wrong, which is not an improvement.
func (r *runnerRef) reportLocked() loadedModel {
	lm := loadedModel{
		model:     r.model,
		busy:      r.refCount > 0,
		size:      int64(r.totalSize),
		sizeVRAM:  int64(r.vramSize),
		expiresAt: r.expiresAt,
		gpus:      slices.Clone(r.gpus),
	}
	if r.llama != nil {
		// The context the engine built, not the one asked for. They differ whenever num_ctx
		// is not a multiple of 256 -- asked 12345, the engine allocated 12544 -- and this
		// figure is what a client can actually fill. The asked value stays in Options,
		// where runner reuse compares it; see GrantedContext for why the two must not merge.
		lm.contextLength = r.llama.ContextLength()
		if granted, _ := r.llama.GrantedContext(); granted > 0 {
			lm.contextLength = granted
		}
		total, vram := r.llama.MemorySize()
		lm.size = int64(total)
		lm.sizeVRAM = int64(vram)

		// Per-device residency. A backend with one device reports the whole
		// figure for it; one whose device names don't map back reports zero
		// rather than guessing.
		lm.vramByGPU = make(map[ml.DeviceID]uint64, len(lm.gpus))
		lm.memByGPU = make(map[ml.DeviceID]api.MemoryBreakdown, len(lm.gpus))
		for _, dev := range lm.gpus {
			lm.vramByGPU[dev] = r.llama.VRAMByGPU(dev)
			lm.memByGPU[dev] = r.llama.MemoryBreakdownByGPU(dev)
		}
		lm.memVRAM, _ = r.llama.MemoryBreakdownTotals()
		lm.weightsOnDisk = r.llama.WeightsOnDisk()
		lm.placement = r.llama.LayerPlacement()
		lm.bandwidthByGPU = r.bandwidthByGPU
		lm.pciByGPU = r.pciByGPU
		lm.expertCount = r.expertCount
		lm.slidingWindow = r.slidingWindow
		lm.layers, lm.activeWeights = r.layers, r.activeWeights
		_, lm.grantedCtxTotal = r.llama.GrantedContext()
		// Briefly cached inside the runner, so a polled /api/ps does not make one HTTP
		// round trip per resident model per request.
		lm.activity = r.llama.Activity(context.Background(), lm.busy)
	}
	// The scheduler waits to set expiresAt, so a model that is still loading may have the
	// zero value. Estimate expiration from the session duration instead.
	if lm.expiresAt.IsZero() {
		lm.expiresAt = time.Now().Add(r.sessionDuration)
	}

	r.lastReported.Store(&lm)
	return lm
}
