//go:build linux

package discover

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ollama/ollama/ml"
)

// sysfsPCIRoot is a variable so tests can point it at a fixture directory. The real
// readings here were captured from a GPU whose firmware had hung, and reproducing that
// state on demand is not possible, so the fixture is the only way to test against it.
var sysfsPCIRoot = "/sys/bus/pci/devices"

// busStateFor reads what the PCIe layer knows about a device.
//
// Every file it reads is world-readable, which is the point: this box's agents have no
// sudo, and the equivalent lspci fields need root. It also does not go through the GPU
// driver, so it still answers for a device the driver has given up on.
func busStateFor(pciID string) *ml.DeviceBusState {
	if pciID == "" {
		return nil
	}

	dir := filepath.Join(sysfsPCIRoot, pciID)
	if _, err := os.Stat(dir); err != nil {
		// Not enumerated. Reported rather than omitted: "the kernel cannot see this
		// device either" is the single most useful fact about a missing GPU, and an
		// absent Bus block would be read as "we did not look".
		return &ml.DeviceBusState{Present: false}
	}

	state := &ml.DeviceBusState{Present: true}
	state.LinkSpeed = readTrimmed(filepath.Join(dir, "current_link_speed"))
	state.PowerState = readTrimmed(filepath.Join(dir, "power_state"))
	if w, err := strconv.Atoi(readTrimmed(filepath.Join(dir, "current_link_width"))); err == nil {
		state.LinkWidth = w
	}
	state.MaxLinkSpeed = readTrimmed(filepath.Join(dir, "max_link_speed"))
	if w, err := strconv.Atoi(readTrimmed(filepath.Join(dir, "max_link_width"))); err == nil {
		state.MaxLinkWidth = w
	}
	state.FatalErrors = sumAERCounters(filepath.Join(dir, "aer_dev_fatal"))
	state.NonFatalErrors = sumAERCounters(filepath.Join(dir, "aer_dev_nonfatal"))
	return state
}

func readTrimmed(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// sumAERCounters totals a sysfs AER file, whose format is one "NAME COUNT" pair per line.
//
// Returns 0 when the file is absent, which is the common case: AER is only exposed for
// devices whose root port supports it. Zero and "not reported" are deliberately the same
// here, because both mean "no evidence of a link problem" and neither is actionable.
func sumAERCounters(path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	total := 0
	for _, line := range strings.Split(string(b), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		if n, err := strconv.Atoi(fields[1]); err == nil {
			total += n
		}
	}
	return total
}
