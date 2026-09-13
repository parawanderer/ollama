package server

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// The box profile measures this machine's decode speed, once, from a model ollama writes
// itself. See api.BoxProfile for what it reports, and slop-zone notebooks/box-profile.ipynb for
// the measurements it rests on: on the machine it was built on, the numbers from a zero-weight
// llama predicted real models' one-GPU decode within 2-4% and their tensor-split decode within
// 2% (dense models and gpt-oss:120b).
//
// It runs only when nothing else would notice: no model loaded or loading, no request waiting,
// no GPU leased, no memory held and no work running on the GPUs by anything else, and the
// server quiet for a minute. Any request or lease stops it at once, and it tries again at the
// next quiet minute. It is never run on a timer once measured: a GPU initialisation on an idle
// card is exactly what preceded both GSP faults on the machine this was built on.

const (
	profileCheckEvery     = 30 * time.Second
	profileIdleFor        = time.Minute
	profileMaxStored      = 8
	profileMinFreeShare   = 0.85
	profileMaxBusyPercent = 5
)

// profileShape is one synthetic model. Device shapes vary width and depth independently,
// which is what separates the per-layer overhead from the bandwidth; fitted to four of them,
// the three numbers reproduced a nine-shape grid within 1.5%.
type profileShape struct{ width, layers int }

func deviceShapes(wide int) []profileShape {
	return []profileShape{{2048, 8}, {2048, 64}, {wide, 8}, {wide, 24}}
}

// linkShapes are measured under tensor split. The reduction cost grows with width (11 µs at
// 4096 against 15 at 8192 on PCIe Gen5 x8), so it is measured at two.
func linkShapes(wide int) []profileShape {
	return []profileShape{{wide / 2, 64}, {wide, 24}}
}

// profileWidth is the widest shape a device can hold with room to spare. 8192 matches the
// widest models people run; a smaller device measures at 4096.
func profileWidth(free uint64) (int, error) {
	for _, w := range []int{8192, 4096} {
		if free >= modelBytes(profileShape{w, 24})+2<<30 {
			return w, nil
		}
	}
	return 0, fmt.Errorf("%s free is too little to measure", formatGiB(free))
}

func modelBytes(s profileShape) uint64 {
	return llm.SyntheticModel{Embedding: s.width, Layers: s.layers}.ReadPerToken()
}

func formatGiB(b uint64) string { return fmt.Sprintf("%.1f GiB", float64(b)/(1<<30)) }

// storedProfile is one machine's measurement, keyed by PCI address so it survives a
// rediscovery that renumbers the devices.
type storedProfile struct {
	MeasuredAt time.Time            `json:"measured_at"`
	Devices    []api.ProfileDevice  `json:"devices,omitempty"`
	Links      []api.ProfileLink    `json:"links,omitempty"`
	Failures   []api.ProfileFailure `json:"failures,omitempty"`
}

type boxProfiler struct {
	db *serverDB

	// Seams for tests.
	timeFn   func(ctx context.Context, gpus []ml.DeviceInfo, modelPath string, tensorSplit bool) (float64, error)
	engineFn func() string
	now      func() time.Time

	// tempDir is where synthetic models are written; empty means os.TempDir().
	tempDir string

	mu        sync.Mutex
	profiles  map[string]*storedProfile
	lastBusy  time.Time
	cancel    context.CancelFunc
	done      chan struct{}
	engine    string
	engineSet bool
}

// newBoxProfiler loads the latest measurement of each machine from db. A nil db keeps
// measurements in the process only.
func newBoxProfiler(db *serverDB) *boxProfiler {
	p := &boxProfiler{
		db:       db,
		timeFn:   llm.TimeDecode,
		engineFn: engineFingerprint,
		now:      time.Now,
		profiles: map[string]*storedProfile{},
	}
	p.lastBusy = p.now() // a server that just started is not yet known to be quiet
	if db != nil {
		if profiles, err := db.loadProfiles(); err != nil {
			slog.Warn("could not read the box profile; it will be measured again", "error", err)
		} else {
			p.profiles = profiles
		}
	}
	return p
}

// preempt stops a measurement in progress and waits until its llama-server has exited, so
// the memory it held is free before anything is placed. Called for every request that reaches
// the scheduler's queue and for every lease; when nothing is running it only notes the time.
func (p *boxProfiler) preempt() {
	if p == nil {
		return
	}
	p.mu.Lock()
	p.lastBusy = p.now()
	cancel, done := p.cancel, p.done
	p.mu.Unlock()
	if cancel != nil {
		cancel()
		<-done
	}
}

// run checks every profileCheckEvery whether the machine needs measuring and is quiet enough
// to measure. candidates returns the GPUs, or a reason they may not be touched now; it must
// not start device discovery, which would itself initialise the GPUs.
func (p *boxProfiler) run(ctx context.Context, candidates func() ([]ml.DeviceInfo, string)) {
	removeStaleProfileDirs(p.tempRoot())
	t := time.NewTicker(profileCheckEvery)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		p.tick(ctx, candidates)
	}
}

// tick measures the machine if it has no profile yet and is quiet. It reports whether it
// measured.
func (p *boxProfiler) tick(ctx context.Context, candidates func() ([]ml.DeviceInfo, string)) bool {
	gpus, busy := candidates()
	if len(gpus) == 0 {
		return false
	}
	id := p.identity(gpus)
	mctx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})

	// The quiet check and publishing cancel happen under one lock, so a request that arrives
	// after the check always finds something to preempt, and one that arrived before it has
	// already made the server not quiet.
	p.mu.Lock()
	_, have := p.profiles[id]
	quiet := p.now().Sub(p.lastBusy) >= profileIdleFor
	if have || busy != "" || !quiet {
		p.mu.Unlock()
		cancel()
		return false
	}
	p.cancel, p.done = cancel, done
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		p.cancel, p.done = nil, nil
		p.mu.Unlock()
		cancel()
		close(done)
	}()

	slog.Info("measuring this machine's decode speed; this happens once for these GPUs, driver and engine", "gpus", len(gpus))
	started := p.now()
	prof, err := p.measure(mctx, gpus)
	if err != nil {
		slog.Info("stopped measuring this machine's decode speed; it will resume when the server is quiet", "reason", err)
		return false
	}
	p.mu.Lock()
	p.profiles[id] = prof
	p.evictLocked()
	p.mu.Unlock()
	p.db.enqueue(profileRow{Identity: id, Profile: *prof})
	slog.Info("measured this machine's decode speed", "took", p.now().Sub(started).Round(time.Second),
		"devices", len(prof.Devices), "links", len(prof.Links), "failures", len(prof.Failures))
	return true
}

// measure times the synthetic shapes on each GPU alone, fits each GPU's three numbers, then
// times tensor split across each set and backs out the link cost. It returns an error only if
// it was interrupted; a measurement that fails is recorded as a failure in the profile.
//
// Every step is logged at info level, because this runs unprompted and touches the GPUs and the
// disk: someone reading the log should be able to see what it did and that it cleaned up.
func (p *boxProfiler) measure(ctx context.Context, gpus []ml.DeviceInfo) (*storedProfile, error) {
	dir, err := os.MkdirTemp(p.tempRoot(), profileDirPrefix())
	if err != nil {
		return nil, err
	}
	slog.Info("box profile: writing synthetic models", "dir", dir)
	defer func() {
		files, _ := os.ReadDir(dir)
		if err := os.RemoveAll(dir); err != nil {
			slog.Warn("box profile: could not remove the synthetic models; they are removed at the next start", "dir", dir, "error", err)
			return
		}
		slog.Info("box profile: removed the synthetic models", "dir", dir, "files", len(files))
	}()

	models := map[profileShape]string{}
	model := func(s profileShape) (string, error) {
		if path, ok := models[s]; ok {
			return path, nil
		}
		path := filepath.Join(dir, fmt.Sprintf("synthetic-%dx%d.gguf", s.width, s.layers))
		if err := (llm.SyntheticModel{Embedding: s.width, Layers: s.layers}).Write(path); err != nil {
			return "", err
		}
		attrs := []any{"file", filepath.Base(path), "width", s.width, "layers", s.layers}
		if info, err := os.Stat(path); err == nil {
			attrs = append(attrs, "size", formatGiB(uint64(info.Size())))
		}
		if onDisk, ok := allocatedBytes(path); ok {
			attrs = append(attrs, "on_disk_bytes", onDisk)
		}
		slog.Info("box profile: wrote a synthetic model (zero weights, sparse)", attrs...)
		models[s] = path
		return path, nil
	}
	timeShape := func(s profileShape, set []ml.DeviceInfo, tensor bool) (float64, error) {
		path, err := model(s)
		if err != nil {
			return 0, err
		}
		started := p.now()
		ms, err := p.timeFn(ctx, set, path, tensor)
		// The load read the file through the page cache: gigabytes of zero pages that would
		// otherwise stay cached until the file is deleted.
		releasePageCache(path)
		split := "none"
		if tensor {
			split = "tensor"
		}
		attrs := []any{"devices", pciList(set), "split", split, "width", s.width, "layers", s.layers,
			"took", p.now().Sub(started).Round(time.Millisecond)}
		if err != nil {
			if ctx.Err() == nil {
				slog.Warn("box profile: timing failed", append(attrs, "error", err)...)
			}
			return 0, err
		}
		slog.Info("box profile: timed decode", append(attrs, "ms_per_token", round3(ms))...)
		return ms, nil
	}

	gpus = slices.Clone(gpus)
	slices.SortFunc(gpus, func(a, b ml.DeviceInfo) int { return cmp.Compare(a.PCIID, b.PCIID) })

	prof := &storedProfile{MeasuredAt: p.now()}
	fits := map[string]deviceFit{}
	widths := map[string]int{}
	var ok []ml.DeviceInfo
	for _, g := range gpus {
		wide, err := profileWidth(g.FreeMemory)
		if err == nil {
			var f deviceFit
			if f, err = fitDevice(func(s profileShape) (float64, error) {
				return timeShape(s, []ml.DeviceInfo{g}, false)
			}, deviceShapes(wide)); err == nil {
				fits[g.PCIID], widths[g.PCIID] = f, wide
				ok = append(ok, g)
				d := f.wire(g.PCIID)
				prof.Devices = append(prof.Devices, d)
				slog.Info("box profile: measured a device", "pci_id", g.PCIID,
					"bandwidth", fmt.Sprintf("%.3f TB/s", float64(d.BandwidthBytesPerSec)/1e12),
					"token_overhead_ms", d.TokenOverheadMs, "layer_overhead_us", d.LayerOverheadUs, "fit_error_pct", d.FitErrorPct)
				continue
			}
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		slog.Warn("box profile: could not measure a device; recorded as a failure", "pci_id", g.PCIID, "error", err)
		prof.Failures = append(prof.Failures, api.ProfileFailure{What: "device", PCIIDs: []string{g.PCIID}, Error: err.Error()})
	}

	for _, set := range tensorSets(ok) {
		wide := 8192
		pcis := make([]string, len(set))
		for i, g := range set {
			wide, pcis[i] = min(wide, widths[g.PCIID]), g.PCIID
		}
		link := api.ProfileLink{PCIIDs: pcis}
		var failed error
		for _, s := range linkShapes(wide) {
			t, err := timeShape(s, set, true)
			if err != nil {
				failed = err
				break
			}
			link.Reductions = append(link.Reductions, api.LinkReduction{
				Width: s.width, Us: reductionUs(t, s, set, fits),
			})
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if failed != nil {
			slog.Warn("box profile: tensor split could not be measured; recorded as a failure", "devices", pciList(set), "error", failed)
			prof.Failures = append(prof.Failures, api.ProfileFailure{What: "tensor_split", PCIIDs: pcis, Error: failed.Error()})
			continue
		}
		slog.Info("box profile: measured a link", "devices", pciList(set), "reductions_us", link.Reductions)
		prof.Links = append(prof.Links, link)
	}
	return prof, nil
}

func pciList(set []ml.DeviceInfo) string {
	out := make([]string, len(set))
	for i, g := range set {
		out[i] = g.PCIID
	}
	return strings.Join(out, ",")
}

// profileDirPrefix names a measurement's directory after the process that owns it, which is
// how removeStaleProfileDirs tells a live server's directory from a dead one's.
func profileDirPrefix() string { return fmt.Sprintf("ollama-profile-%d-", os.Getpid()) }

func (p *boxProfiler) tempRoot() string {
	if p.tempDir != "" {
		return p.tempDir
	}
	return os.TempDir()
}

// profileStaleAfter is how old a measurement directory must be to be removed when its owner
// cannot be determined. A whole measurement takes about half a minute.
const profileStaleAfter = time.Hour

// removeStaleProfileDirs removes the measurement directories a server that died mid-measurement
// left behind: files of many GiB on paper, a few KiB on disk (more where the filesystem does
// not keep holes). Called once at startup, before this process has made any of its own, so a
// directory named after this process's id is stale too -- in a container, the server is
// always the same pid.
func removeStaleProfileDirs(root string) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return
	}
	for _, e := range entries {
		name := e.Name()
		if !e.IsDir() || !strings.HasPrefix(name, "ollama-profile-") {
			continue
		}
		path := filepath.Join(root, name)
		var pid int
		owned := false
		if _, err := fmt.Sscanf(strings.TrimPrefix(name, "ollama-profile-"), "%d-", &pid); err == nil && pid > 0 {
			owned = true
		}
		switch {
		case owned && pid != os.Getpid() && processAlive(pid):
			continue // another server, measuring now
		case !owned:
			info, err := e.Info()
			if err != nil || time.Since(info.ModTime()) < profileStaleAfter {
				continue
			}
		}
		if err := os.RemoveAll(path); err != nil {
			slog.Warn("box profile: could not remove a measurement directory an earlier server left", "dir", path, "error", err)
			continue
		}
		slog.Info("box profile: removed a measurement directory an earlier server left", "dir", path)
	}
}

// processAlive reports whether pid is a running process. An answer it cannot get counts as
// alive, which keeps the directory: leaving files behind is the cheaper mistake.
func processAlive(pid int) bool {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	err = proc.Signal(syscall.Signal(0))
	return err == nil || !errors.Is(err, os.ErrProcessDone)
}

// tensorSets are the device sets tensor split is measured across: every pair, and every
// device together when there are more than two. Only devices of one backend can split a
// tensor between them.
func tensorSets(gpus []ml.DeviceInfo) [][]ml.DeviceInfo {
	byLib := map[string][]ml.DeviceInfo{}
	for _, g := range gpus {
		byLib[g.Library] = append(byLib[g.Library], g)
	}
	var sets [][]ml.DeviceInfo
	for _, lib := range slices.Sorted(maps.Keys(byLib)) {
		g := byLib[lib]
		for i := range g {
			for j := i + 1; j < len(g); j++ {
				sets = append(sets, []ml.DeviceInfo{g[i], g[j]})
			}
		}
		if len(g) > 2 {
			sets = append(sets, g)
		}
	}
	return sets
}

// deviceFit is one GPU's decode time per token: a + c·layers + bytes ÷ bw.
type deviceFit struct {
	a, c, bw float64 // ms, ms per layer, bytes per ms
	worstPct float64
}

func (f deviceFit) ms(s profileShape) float64 {
	return f.a + f.c*float64(s.layers) + float64(modelBytes(s))/f.bw
}

func (f deviceFit) wire(pci string) api.ProfileDevice {
	return api.ProfileDevice{
		PCIID:                pci,
		BandwidthBytesPerSec: uint64(f.bw * 1000),
		TokenOverheadMs:      round3(f.a),
		LayerOverheadUs:      round3(f.c * 1000),
		FitErrorPct:          round3(f.worstPct),
	}
}

func round3(v float64) float64 { return math.Round(v*1000) / 1000 }

// fitDevice times each shape and fits a, c and bw by least squares.
func fitDevice(timeOf func(profileShape) (float64, error), shapes []profileShape) (deviceFit, error) {
	var rows [][3]float64
	var ys []float64
	for _, s := range shapes {
		t, err := timeOf(s)
		if err != nil {
			return deviceFit{}, err
		}
		rows = append(rows, [3]float64{1, float64(s.layers), float64(modelBytes(s))})
		ys = append(ys, t)
	}
	x, err := leastSquares3(rows, ys)
	if err != nil {
		return deviceFit{}, err
	}
	if x[2] <= 0 {
		return deviceFit{}, errors.New("decode time did not grow with the bytes read")
	}
	f := deviceFit{a: x[0], c: x[1], bw: 1 / x[2]}
	for i, s := range shapes {
		f.worstPct = max(f.worstPct, 100*math.Abs(f.ms(s)-ys[i])/ys[i])
	}
	return f, nil
}

// leastSquares3 solves the normal equations for three unknowns.
func leastSquares3(rows [][3]float64, ys []float64) ([3]float64, error) {
	var m [3][4]float64
	for k, r := range rows {
		for i := range 3 {
			for j := range 3 {
				m[i][j] += r[i] * r[j]
			}
			m[i][3] += r[i] * ys[k]
		}
	}
	for col := range 3 {
		pivot := col
		for r := col + 1; r < 3; r++ {
			if math.Abs(m[r][col]) > math.Abs(m[pivot][col]) {
				pivot = r
			}
		}
		if math.Abs(m[pivot][col]) < 1e-300 {
			return [3]float64{}, errors.New("shapes do not determine the fit")
		}
		m[col], m[pivot] = m[pivot], m[col]
		for r := range 3 {
			if r == col {
				continue
			}
			k := m[r][col] / m[col][col]
			for c := col; c < 4; c++ {
				m[r][c] -= k * m[col][c]
			}
		}
	}
	return [3]float64{m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]}, nil
}

// reductionUs backs one reduction's cost out of a tensor-split time. Only the bandwidth part of
// a device's time divides among the devices; the per-token and per-layer overheads do not, and
// the §9 form that divided all of it made MoE models' link cost look twice what it was.
//
// With devices of different speeds, the split is only as fast as the slowest, so that one's
// fit stands for the set.
func reductionUs(tensorMs float64, s profileShape, set []ml.DeviceInfo, fits map[string]deviceFit) float64 {
	var one, minBW float64
	for _, g := range set {
		f := fits[g.PCIID]
		one = max(one, f.ms(s))
		if minBW == 0 || f.bw < minBW {
			minBW = f.bw
		}
	}
	n := float64(len(set))
	bwPart := float64(modelBytes(s)) / minBW
	us := (tensorMs - (one - bwPart*(1-1/n))) * 1000 / (2 * float64(s.layers))
	return round3(max(0, us))
}

// identity names the GPUs, driver and engine a profile was measured on. It changes on a card
// swap, a driver update or a new engine build; it does not change on a restart.
func (p *boxProfiler) identity(gpus []ml.DeviceInfo) string {
	p.mu.Lock()
	if !p.engineSet {
		p.engine, p.engineSet = p.engineFn(), true
	}
	engine := p.engine
	p.mu.Unlock()

	parts := make([]string, 0, len(gpus)+1)
	for _, g := range gpus {
		parts = append(parts, fmt.Sprintf("%s|%s|%s|%d|%d.%d|%d", g.PCIID, g.Library, g.Description,
			g.TotalMemory, g.DriverMajor, g.DriverMinor, g.NVIDIADriverMajor))
	}
	slices.Sort(parts)
	sum := sha256.Sum256([]byte(strings.Join(append(parts, engine), "\n")))
	return hex.EncodeToString(sum[:8])
}

// engineFingerprint identifies the engine build by the files it runs from: the llama-server
// binary and the backend libraries beside it. Reading their size and modification time costs
// nothing and does not start the engine, which would initialise the GPUs.
func engineFingerprint() string {
	exe, err := llm.FindLlamaServer()
	if err != nil {
		return ""
	}
	var parts []string
	_ = filepath.WalkDir(filepath.Dir(exe), func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		name := d.Name()
		if path != exe && !strings.HasPrefix(name, "libggml") && !strings.HasPrefix(name, "ggml") {
			return nil
		}
		if info, err := d.Info(); err == nil {
			parts = append(parts, fmt.Sprintf("%s:%d:%d", name, info.Size(), info.ModTime().Unix()))
		}
		return nil
	})
	slices.Sort(parts)
	return strings.Join(parts, ",")
}

// evictLocked keeps at most profileMaxStored machines in memory, dropping the oldest measurement.
// The database keeps every one.
func (p *boxProfiler) evictLocked() {
	for len(p.profiles) > profileMaxStored {
		var oldest string
		for id, pr := range p.profiles {
			if oldest == "" || pr.MeasuredAt.Before(p.profiles[oldest].MeasuredAt) {
				oldest = id
			}
		}
		delete(p.profiles, oldest)
	}
}

// report is the profile for these GPUs as /api/info presents it, with each device's current id.
func (p *boxProfiler) report(gpus []ml.DeviceInfo) *api.BoxProfile {
	if p == nil || len(gpus) == 0 {
		return nil
	}
	id := p.identity(gpus)
	p.mu.Lock()
	defer p.mu.Unlock()
	stored, ok := p.profiles[id]
	if !ok {
		state := "pending"
		if p.cancel != nil {
			state = "measuring"
		}
		return &api.BoxProfile{State: state}
	}
	ids := map[string]string{}
	for _, g := range gpus {
		ids[g.PCIID] = g.ID
	}
	out := &api.BoxProfile{
		State:      "measured",
		MeasuredAt: &stored.MeasuredAt,
		Links:      stored.Links,
		Failures:   stored.Failures,
	}
	for _, d := range stored.Devices {
		d.ID = ids[d.PCIID]
		out.Devices = append(out.Devices, d)
	}
	return out
}

// profileCandidates returns the GPUs if the box profile may measure them now, or a reason it
// may not. It reads the cached device list and the driver's live figures only: starting
// discovery here would initialise every GPU on each check.
func (s *Scheduler) profileCandidates() ([]ml.DeviceInfo, string) {
	s.refreshFreeMemory()
	s.deviceCacheMu.Lock()
	devices := slices.Clone(s.deviceCache)
	s.deviceCacheMu.Unlock()
	if len(devices) == 0 {
		return nil, "no GPUs discovered"
	}

	s.loadedMu.Lock()
	loaded := len(s.loaded)
	s.loadedMu.Unlock()
	switch {
	case loaded > 0:
		return devices, "a model is loaded"
	case s.loadsInFlight.Load() > 0:
		return devices, "a model is loading"
	case len(s.pendingReqCh) > 0:
		return devices, "requests are waiting"
	case len(s.leases.holders()) > 0:
		return devices, "GPUs are leased"
	}
	for _, d := range devices {
		if d.TotalMemory > 0 && float64(d.FreeMemory) < profileMinFreeShare*float64(d.TotalMemory) {
			return devices, "something else holds memory on " + d.PCIID
		}
		if u := d.Utilization; u != nil && u.GPUPercent != nil && *u.GPUPercent > profileMaxBusyPercent {
			return devices, "something else is running on " + d.PCIID
		}
	}
	return devices, ""
}
