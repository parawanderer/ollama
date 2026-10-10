package power

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// rapl reads the CPU package's energy counter from the powercap sysfs, which Linux exposes
// for both Intel RAPL and AMD's equivalent under the same intel-rapl names.
//
// Two things stand between ollama and it. energy_uj is mode 0400, root only, because a
// fine-grained energy counter is a side channel (PLATYPUS, CVE-2020-8694). And Docker masks
// /sys/devices/virtual/powercap -- where the class links point -- in every container, root
// or not. The working arrangement is to bind-mount the host's directory read-only somewhere
// else in a container that runs as root:
//
//	-v /sys/devices/virtual/powercap:/host/powercap:ro    and    rapl@/host/powercap
//
// Only the package domain is reported. AMD has no DRAM domain, and "core" is a part of the
// package, so adding it would count the cores twice.
type rapl struct {
	dir  string // the package-0 domain directory
	wrap int64  // max_energy_range_uj

	mu        sync.Mutex
	lastUJ    int64
	lastAt    time.Time
	haveFirst bool
}

// raplRoots are tried in order when no argument is given: the host's own sysfs, then the
// conventional bind-mount point.
var raplRoots = []string{"/sys/class/powercap", "/host/powercap", "/host/powercap/intel-rapl"}

func newRAPL(arg string) (Provider, error) {
	roots := raplRoots
	if arg != "" {
		roots = []string{arg, filepath.Join(arg, "intel-rapl")}
	}
	var errs []string
	for _, root := range roots {
		dir, err := findPackage(root)
		if err != nil {
			errs = append(errs, err.Error())
			continue
		}
		wrap, err := readInt(filepath.Join(dir, "max_energy_range_uj"))
		if err != nil {
			errs = append(errs, err.Error())
			continue
		}
		if _, err := readInt(filepath.Join(dir, "energy_uj")); err != nil {
			errs = append(errs, err.Error())
			continue
		}
		return &rapl{dir: dir, wrap: wrap}, nil
	}
	return nil, errors.New(strings.Join(errs, "; "))
}

// findPackage returns the first intel-rapl:N directory under root whose name is package-N.
func findPackage(root string) (string, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return "", err
	}
	for _, e := range entries {
		n := e.Name()
		if !strings.HasPrefix(n, "intel-rapl:") || strings.Count(n, ":") != 1 {
			continue
		}
		dir := filepath.Join(root, n)
		name, err := os.ReadFile(filepath.Join(dir, "name"))
		if err == nil && strings.HasPrefix(strings.TrimSpace(string(name)), "package-") {
			return dir, nil
		}
	}
	return "", fmt.Errorf("%s: no package domain", root)
}

func readInt(path string) (int64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
}

func (r *rapl) Scope() Scope            { return ScopeCPU }
func (r *rapl) Interval() time.Duration { return 250 * time.Millisecond }

func (r *rapl) Read(context.Context) (Reading, error) {
	uj, err := readInt(filepath.Join(r.dir, "energy_uj"))
	if err != nil {
		return Reading{}, err
	}
	now := time.Now()

	r.mu.Lock()
	defer r.mu.Unlock()
	reading := Reading{EnergyMJ: ptr(uj / 1000), EnergyWrapMJ: ptr(r.wrap / 1000)}
	if r.haveFirst {
		d := uj - r.lastUJ
		if d < 0 { // wrapped
			d += r.wrap
		}
		if dt := now.Sub(r.lastAt); dt > 0 {
			// microjoules per microsecond is watts; times 1000 for milliwatts
			reading.PowerMW = ptr(d * 1000 / dt.Microseconds())
		}
	}
	r.lastUJ, r.lastAt, r.haveFirst = uj, now, true
	return reading, nil
}

func init() { Register("rapl", newRAPL) }
