package server

import (
	"log/slog"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

// The per-model speed correction: what the box profile's prediction gets wrong for a given model
// on a given placement, learned from the generations it has served. The profile times a plain
// synthetic llama, so it knows the machine but not the architecture: gpt-oss:120b does more per
// layer than a llama, qwen3.5's recurrent layers more still. Rather than write architecture
// knowledge by hand, each model's error is measured: the correction is the median of measured ÷
// predicted over its recent clean generations, and it multiplies the profile's prediction.

const (
	// correctionWindow is how many recent samples a key keeps, so a model's correction follows a
	// driver or engine change rather than averaging it away.
	correctionWindow = 20
	// correctionMinSamples is how many a key needs before it is applied: one sample is a guess.
	correctionMinSamples = 3
	// correctionMinDecoded excludes very short decodes, whose per-token time is mostly noise.
	correctionMinDecoded = 16
)

// speedCorrections holds each key's recent measured ÷ predicted ratios.
type speedCorrections struct {
	mu     sync.Mutex
	ratios map[string][]float64
}

func newSpeedCorrections() *speedCorrections {
	return &speedCorrections{ratios: map[string][]float64{}}
}

// speedKey identifies what a correction applies to: a model on a kind of device set, split one
// way. deviceKinds is usageDeviceKinds, not the device names, so the same model on the other of
// two identical cards shares its correction rather than starting over. It is built from strings
// the generations table stores, so a correction can be rebuilt from the table.
func speedKey(model, deviceKinds, split string) string {
	return model + "|" + deviceKinds + "|" + split
}

// factor returns the median ratio for key and how many samples it rests on.
func (c *speedCorrections) factor(key string) (float64, int) {
	if c == nil {
		return 1, 0
	}
	c.mu.Lock()
	r := slices.Clone(c.ratios[key])
	c.mu.Unlock()
	if len(r) == 0 {
		return 1, 0
	}
	slices.Sort(r)
	mid := len(r) / 2
	if len(r)%2 == 1 {
		return r[mid], len(r)
	}
	return (r[mid-1] + r[mid]) / 2, len(r)
}

func (c *speedCorrections) add(key string, ratio float64) {
	if c == nil || !(ratio > 0) {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	r := append(c.ratios[key], ratio)
	if len(r) > correctionWindow {
		r = r[len(r)-correctionWindow:]
	}
	c.ratios[key] = r
}

// predictAndLearn fills a generation's row with the profile's prediction and, once the model has
// earned one, the corrected prediction, then lets the generation teach the correction if its
// timing is clean. The corrected prediction is made from what was learned before this
// generation, so its error is an honest test of the correction rather than a fit to itself. It
// returns the same prediction for gen.end, or nil when there is none.
func (s *Scheduler) predictAndLearn(key string, in decodeInputs, t api.GenerationTimings, row *usageGeneration) *api.DecodePrediction {
	ms, basis, ok := predictedEvalMs(in, t.PromptTokens, t.Decoded)
	if !ok {
		return nil
	}
	row.PredictedEvalMs, row.PredictedBasis = &ms, basis
	out := &api.DecodePrediction{
		MsPerToken: round3(ms), Basis: "profile", ExcludesCacheRead: basis == "profile_no_kv",
		OccupancyTokens: float64(t.PromptTokens) + float64(t.Decoded)/2,
	}
	if factor, n := s.speed.factor(key); n >= correctionMinSamples {
		corrected := ms * factor
		row.CorrectedEvalMs, row.CorrectionSamples = &corrected, n
		out.MsPerToken, out.Basis = round3(corrected), "profile_corrected"
		out.ProfileMsPerToken, out.CorrectionFactor, out.CorrectionSamples = round3(ms), round3(factor), n
	}
	measured := t.EvalMs / float64(max(t.Decoded, 1))
	if usableForCorrection(t.Decoded, row.Contended, ms) {
		s.speed.add(key, measured/ms)
	}
	slog.Debug("decode speed against the profile's prediction", "key", key,
		"predicted_ms_per_token", ms, "corrected_ms_per_token", row.CorrectedEvalMs,
		"measured_ms_per_token", measured, "basis", basis, "contended", row.Contended)
	return out
}

// usable reports whether a generation's timing may teach the correction.
func usableForCorrection(decoded int, contended bool, predictedMs float64) bool {
	return decoded >= correctionMinDecoded && !contended && predictedMs > 0
}

// loadSpeedCorrections rebuilds the corrections from the generations already recorded. Rows from
// before device_kinds was recorded carry only device names; legacyKinds turns those into kinds
// where it can, and rows it cannot are left out rather than guessed.
func (d *serverDB) loadSpeedCorrections(c *speedCorrections, legacyKinds func(devices string) string) error {
	if d == nil {
		return nil
	}
	rows, err := d.db.Query(`SELECT model, coalesce(devices, ''), coalesce(device_kinds, ''), coalesce(split, ''),
			eval_ms, decoded, predicted_eval_ms_per_token
		FROM generations WHERE predicted_eval_ms_per_token > 0 AND decoded >= ? AND coalesce(contended, 0) = 0 ORDER BY id`,
		correctionMinDecoded)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var model, devices, kinds, split string
		var evalMs, predicted float64
		var decoded int
		if err := rows.Scan(&model, &devices, &kinds, &split, &evalMs, &decoded, &predicted); err != nil {
			return err
		}
		if kinds == "" && legacyKinds != nil {
			kinds = legacyKinds(devices)
		}
		if kinds == "" {
			continue
		}
		c.add(speedKey(model, kinds, split), evalMs/float64(decoded)/predicted)
	}
	return rows.Err()
}

// legacyDeviceKinds maps a row's device names ("CUDA0,CUDA1") to kinds using the devices present
// now, but only for a library whose devices are all one kind. A name is an enumeration index, and
// indexes shift when a card faults or is swapped, so "CUDA1 then" need not be "CUDA1 now"; when
// every card of the library is the same kind, which one it was does not matter. On a mixed box
// the row maps to nothing.
func legacyDeviceKinds(gpus []ml.DeviceInfo) func(devices string) string {
	kindOf := map[string]string{} // library -> its one kind, or "" when it has several
	for _, g := range gpus {
		k, seen := kindOf[g.Library]
		switch {
		case !seen:
			kindOf[g.Library] = deviceKind(g)
		case k != deviceKind(g):
			kindOf[g.Library] = ""
		}
	}
	return func(devices string) string {
		if devices == "" {
			return ""
		}
		var kinds []string
		for _, name := range strings.Split(devices, ",") {
			kind := ""
			for lib, k := range kindOf {
				if strings.HasPrefix(name, lib) && isDigits(name[len(lib):]) {
					kind = k
				}
			}
			if kind == "" {
				return ""
			}
			kinds = append(kinds, kind)
		}
		slices.Sort(kinds)
		return strings.Join(kinds, ",")
	}
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// runnerUsage is what a runner's generation callback and the rest of the scheduler share about
// it. The callback is created before the runner exists, so the runner holds a pointer to this
// rather than the callback holding the runner.
type runnerUsage struct {
	key  string
	gpus []ml.DeviceID

	// inFlight mirrors refCount and idleSinceMs records when it last reached zero, so another
	// runner's generation can ask, without taking any lock, whether this one was working at the
	// same time.
	inFlight    atomic.Int32
	idleSinceMs atomic.Int64
	unloaded    atomic.Bool
}

// usageRegistry is every runner's usage, with its own lock. contended runs in a generation's
// completion path, and loadedMu is held across whole loads (fit probes included, seconds at a
// time), so taking loadedMu there would hold up the reply to a client of another model.
//
// An unloaded runner stays for usageForgetAfter past its last request: a generation that
// overlapped its final one is still contended.
type usageRegistry struct {
	mu  sync.Mutex
	set map[*runnerUsage]struct{}
}

const usageForgetAfter = 10 * time.Minute

func (r *usageRegistry) add(u *runnerUsage) {
	if u == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.set == nil {
		r.set = map[*runnerUsage]struct{}{}
	}
	r.set[u] = struct{}{}
}

// others returns every usage but u, dropping unloaded ones idle for longer than usageForgetAfter.
func (r *usageRegistry) others(u *runnerUsage) []*runnerUsage {
	r.mu.Lock()
	defer r.mu.Unlock()
	forget := time.Now().Add(-usageForgetAfter).UnixMilli()
	out := make([]*runnerUsage, 0, len(r.set))
	for o := range r.set {
		if o.unloaded.Load() && o.inFlight.Load() == 0 && o.idleSinceMs.Load() < forget {
			delete(r.set, o)
			continue
		}
		if o != u {
			out = append(out, o)
		}
	}
	return out
}

// noteRequests records the runner's request count; called wherever refCount changes, under refMu.
func (u *runnerUsage) noteRequests(n uint) {
	if u == nil {
		return
	}
	u.inFlight.Store(int32(n))
	if n == 0 {
		u.idleSinceMs.Store(time.Now().UnixMilli())
	}
}

// contended reports whether another runner sharing a device with u worked at any point since
// sinceMs. Two runners on one card share its bandwidth, so a generation timed while another ran
// measures the contention, not the model.
func (s *Scheduler) contended(u *runnerUsage, sinceMs int64) bool {
	for _, o := range s.usages.others(u) {
		if !sharesDevice(o.gpus, u.gpus) {
			continue
		}
		if o.inFlight.Load() > 0 || o.idleSinceMs.Load() > sinceMs {
			return true
		}
	}
	return false
}

func sharesDevice(a, b []ml.DeviceID) bool {
	for _, x := range a {
		if slices.Contains(b, x) {
			return true
		}
	}
	return false
}
