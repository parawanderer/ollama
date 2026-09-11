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

	return busStateAt(filepath.Join(sysfsPCIRoot, pciID))
}

// busStateAt is busStateFor for a device directory the caller has already resolved.
func busStateAt(dir string) *ml.DeviceBusState {
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

// nvidiaGPUsInSysfs lists every NVIDIA display-class device the kernel enumerates.
//
// This is the OUTERMOST ring of the three that know about a GPU, and it is the only one
// that survives everything:
//
//	sysfs  - every PCI device, whatever the driver is doing. Never lost a device here.
//	NVML   - devices the driver is managing. Keeps a faulted device, with name and UUID...
//	CUDA   - devices usable for compute. Drops a faulted device entirely.
//
// The middle ring was originally treated as the superset, and that was WRONG in a way only
// a second incident showed. NVML did keep the faulted card at first, but after the driver
// was torn down and reloaded without it, NVML reported one device too -- so a detector
// built on "NVML minus CUDA" went quiet while the broken card was still bolted to the
// machine, which is the exact silence this whole mechanism exists to break.
//
// sysfs never wavered: the device stayed at 0000:03:00.0, class 0x030000, vendor 0x10de,
// still bound to the nvidia driver, through the fault, the hung unload and the reload.
func nvidiaGPUsInSysfs() []string {
	entries, err := os.ReadDir(sysfsPCIRoot)
	if err != nil {
		return nil
	}

	var out []string
	for _, entry := range entries {
		dir := filepath.Join(sysfsPCIRoot, entry.Name())

		// 0x10de is NVIDIA. Other vendors' GPUs are deliberately not reported: this
		// answers "is a card the backend should have offered missing", and only the
		// devices whose absence we can explain belong in that answer.
		if !strings.EqualFold(readTrimmed(filepath.Join(dir, "vendor")), "0x10de") {
			continue
		}

		// Display controller (0x0300xx). The class is read rather than assumed because a
		// vendor ships more than GPUs on a PCI bus, and an audio function or a bridge
		// reported as a missing GPU would be a permanent false alarm.
		if !strings.HasPrefix(readTrimmed(filepath.Join(dir, "class")), "0x0300") {
			continue
		}

		out = append(out, entry.Name())
	}
	return out
}

// PCIeMaxLink reports the fastest PCIe link a device can have as installed -- the lower of
// the card's capability and its upstream port's -- from sysfs, for any vendor, or zeros.
//
// The card alone is the wrong answer, and on this box it is wrong in the direction a user
// would believe: each RTX PRO 6000 reports x16, while the root port above it (00:01.1,
// 00:01.3) reports x8, because the board splits its 16 CPU lanes between two slots. The link
// can never train wider than the narrower end, so "max x16" would describe a link that
// cannot exist. The same holds for speed: a Gen5 card in a Gen4 slot runs at Gen4.
//
// Capability only. The current link is deliberately not offered: an idle card drops to Gen1
// under ASPM, so an instantaneous reading beside a capability reads as a fault.
func PCIeMaxLink(pciID string) (generation, width int) {
	if pciID == "" {
		return 0, 0
	}
	dir := filepath.Join(sysfsPCIRoot, pciID)
	generation, width = linkCapability(dir)

	resolved, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return generation, width
	}
	portGen, portWidth := linkCapability(filepath.Dir(resolved))
	if portGen > 0 && (generation == 0 || portGen < generation) {
		generation = portGen
	}
	if portWidth > 0 && (width == 0 || portWidth < width) {
		width = portWidth
	}
	return generation, width
}

func linkCapability(dir string) (generation, width int) {
	generation = pcieGeneration(readTrimmed(filepath.Join(dir, "max_link_speed")))
	width, _ = strconv.Atoi(readTrimmed(filepath.Join(dir, "max_link_width")))
	return generation, width
}

// pcieGeneration maps the kernel's link-speed string to a PCIe generation.
//
// The strings are a closed set -- pci_speed_string() in drivers/pci/probe.c indexes a static
// table -- so this is an exact match on the kernel's own vocabulary, not a parse of a number.
// Anything else, including "Unknown" and the legacy PCI/AGP entries, is 0.
func pcieGeneration(speed string) int {
	switch speed {
	case "2.5 GT/s PCIe":
		return 1
	case "5.0 GT/s PCIe":
		return 2
	case "8.0 GT/s PCIe":
		return 3
	case "16.0 GT/s PCIe":
		return 4
	case "32.0 GT/s PCIe":
		return 5
	case "64.0 GT/s PCIe":
		return 6
	}
	return 0
}
