// Package power reads power from sources outside the GPUs -- the CPU's own energy counter,
// a metering smart plug, anything else -- and keeps the latest reading of each for /api/info
// and so for the /api/events stream.
//
// It is pluggable on purpose. Every source is a Provider registered under a kind name, and
// OLLAMA_POWER_SOURCES chooses which run and with what argument. Adding a source is one file
// with a Provider and an init() calling Register; see README.md in this directory.
//
// Every reading names its provider and its Scope, so that a consumer can never take a
// reading of part of the machine (the CPU package, one GPU) for the whole of it (the wall).
// GPU board power is not here: it is read per device from NVML and reported on each GPU.
package power

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"
)

// Scope says how much of the machine a reading covers.
type Scope string

const (
	// ScopeWall is everything behind the meter, measured at the socket: PSU losses included.
	ScopeWall Scope = "wall"
	// ScopeSystem is the whole machine measured on the DC side (a PSU or BMC reporting it).
	ScopeSystem Scope = "system"
	// ScopeCPU is the CPU package alone.
	ScopeCPU Scope = "cpu"
	// ScopeGPU is one or more GPUs alone.
	ScopeGPU Scope = "gpu"
)

// Reading is one sample from one source. Every figure is optional: a nil means the source
// does not report it, never that it reported zero.
type Reading struct {
	// PowerMW is the instantaneous draw in milliwatts, as the source reports it or, for a
	// cumulative counter, averaged over the interval since the previous read.
	PowerMW *int64
	// EnergyMJ is the source's cumulative energy counter in millijoules. It only grows,
	// except where it wraps (EnergyWrapMJ) or the source resets it (a plug's own reset).
	// Integrating this is exact; integrating PowerMW samples is an approximation.
	EnergyMJ *int64
	// EnergyWrapMJ is the value at which EnergyMJ wraps to zero, for a counter that does.
	EnergyWrapMJ *int64
	// SourceTime is the source's own timestamp for the reading, verbatim, when it gives one.
	// It is not parsed: a plug's clock and timezone are whatever it was configured with.
	SourceTime string
}

// Provider is one power source.
type Provider interface {
	// Scope is how much of the machine this provider measures.
	Scope() Scope
	// Interval is how often to read it. Reads are made on a goroutine of their own, so a
	// slow source never delays a sample; the sample carries the latest reading and its time.
	Interval() time.Duration
	// Read takes one reading. It must not change anything on the source.
	Read(ctx context.Context) (Reading, error)
}

// Factory builds a provider from the argument given after the kind in OLLAMA_POWER_SOURCES
// ("" when none was given). Returning an error leaves the source out, logged once.
type Factory func(arg string) (Provider, error)

var (
	factoriesMu sync.Mutex
	factories   = map[string]Factory{}
)

// Register makes a provider kind available by name. Call it from init().
func Register(kind string, f Factory) {
	factoriesMu.Lock()
	defer factoriesMu.Unlock()
	factories[kind] = f
}

// Sample is the latest state of one configured source, as /api/info reports it.
type Sample struct {
	// Provider is the configured kind, e.g. "rapl" or "tasmota".
	Provider string
	// Name is the label given in the configuration (label=kind@arg), to tell apart two
	// sources of one kind -- two plugs, say. Empty when none was given.
	Name  string
	Scope Scope
	Reading
	// AtMS is when ollama took the reading, in Unix milliseconds. Zero before the first one.
	AtMS int64
	// Error is why the latest read failed. The figures above are then absent: a reading
	// that could not be taken is never repeated as if it were current.
	Error string
}

type source struct {
	kind string
	name string
	p    Provider

	mu     sync.Mutex
	latest Sample
}

// Sources is the running set of configured sources.
type Sources struct {
	list []*source
}

// Parse reads a source list: comma-separated entries of [label=]kind[@argument], for
// example "rapl,box=tasmota@http://192.168.1.50". "none" or an empty list means no sources.
func Parse(spec string) (*Sources, error) {
	s := &Sources{}
	spec = strings.TrimSpace(spec)
	if spec == "" || spec == "none" {
		return s, nil
	}
	factoriesMu.Lock()
	defer factoriesMu.Unlock()
	var errs []string
	for _, entry := range strings.Split(spec, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		var name string
		if label, rest, ok := strings.Cut(entry, "="); ok && !strings.Contains(label, "@") {
			name, entry = label, rest
		}
		kind, arg, _ := strings.Cut(entry, "@")
		f, ok := factories[kind]
		if !ok {
			errs = append(errs, fmt.Sprintf("%s: unknown power source (known: %s)", kind, strings.Join(known(), ", ")))
			continue
		}
		p, err := f(arg)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", kind, err))
			continue
		}
		s.list = append(s.list, &source{kind: kind, name: name, p: p})
	}
	if len(errs) > 0 {
		return s, fmt.Errorf("power sources: %s", strings.Join(errs, "; "))
	}
	return s, nil
}

func known() []string {
	var k []string
	for name := range factories {
		k = append(k, name)
	}
	sort.Strings(k)
	return k
}

// Run reads every source on its own interval until ctx is done.
func (s *Sources) Run(ctx context.Context) {
	for _, src := range s.list {
		go src.run(ctx)
	}
}

func (src *source) run(ctx context.Context) {
	t := time.NewTicker(src.p.Interval())
	defer t.Stop()
	for {
		src.readOnce(ctx)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func (src *source) readOnce(ctx context.Context) {
	rctx, cancel := context.WithTimeout(ctx, max(src.p.Interval(), 2*time.Second))
	defer cancel()
	r, err := src.p.Read(rctx)
	now := time.Now().UnixMilli()

	src.mu.Lock()
	defer src.mu.Unlock()
	if err != nil {
		if src.latest.Error == "" {
			slog.Debug("power source read failed", "provider", src.kind, "error", err)
		}
		src.latest = Sample{Provider: src.kind, Name: src.name, Scope: src.p.Scope(), AtMS: now, Error: err.Error()}
		return
	}
	src.latest = Sample{Provider: src.kind, Name: src.name, Scope: src.p.Scope(), Reading: r, AtMS: now}
}

// Latest returns the most recent sample of every configured source, in configured order.
// A source that has not been read yet is left out.
func (s *Sources) Latest() []Sample {
	if s == nil {
		return nil
	}
	out := make([]Sample, 0, len(s.list))
	for _, src := range s.list {
		src.mu.Lock()
		l := src.latest
		src.mu.Unlock()
		if l.AtMS != 0 {
			out = append(out, l)
		}
	}
	return out
}

func ptr[T any](v T) *T { return &v }
