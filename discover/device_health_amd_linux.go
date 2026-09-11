//go:build linux

package discover

import (
	"errors"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/ollama/ollama/ml"
)

// sysfsRoot is the root the AMD path reads KFD topology and devices from. A variable so the
// tests can point it at a fixture tree: a faulted AMD GPU is not something a test can
// arrange, so the fault states below are exercised through injected read errors.
var sysfsRoot = "/sys"

// amdProbe is what was learned about one AMD compute device the backend did not offer.
type amdProbe struct {
	PCIID      string
	Integrated bool

	// Driver is the basename of the device's driver link: "amdgpu", "vfio-pci", or "" when
	// unbound. Anything but amdgpu means the device was deliberately given to something
	// else -- passthrough to a VM being the common case -- and is not ours to report.
	Driver string

	// RuntimeStatus is power/runtime_status: "active", "suspended", "suspending", ...
	RuntimeStatus string

	// LiveReadErr is the error from reading gpu_busy_percent, or nil if it answered.
	//
	// That attribute is the AMD equivalent of the NVML temperature call on the NVIDIA path:
	// it goes through amdgpu_pm_get_sensor_generic, which asks the device's power-management
	// firmware for live state. mem_info_vram_total does NOT -- it returns a number the driver
	// stored at init -- so, exactly as with nvmlDeviceGetMemoryInfo, a health check built on
	// the memory counters would succeed on a dead device and never fire.
	LiveReadErr error

	Bus *ml.DeviceBusState
}

// classifyAMD decides whether an AMD compute device the backend did not offer is broken.
//
// It reports only on error codes that the driver source shows a healthy device cannot
// produce. That is deliberately stricter than the NVIDIA path, and the reason is that on
// AMD "not offered" is ordinary: ollama's ROCm build supports a fixed list of gfx targets,
// and most Ryzen desktops carry an integrated GPU that KFD lists as a compute node and
// ollama correctly declines. This box is one -- gfx1036, simd_count 4, active and healthy.
// Absence from the backend is therefore not evidence of anything on AMD, and a detector
// that treated it as evidence would raise a false alarm on nearly every AMD APU.
//
// The codes, from drivers/gpu/drm/amd/pm/amdgpu_pm.c and amdgpu_dpm.c:
//
//	EPERM   amdgpu_pm_get_access_if_active: the device is runtime-suspended. An idle AMD
//	        dGPU does this by design. Healthy.
//	EINVAL  amdgpu_dpm_read_sensor: the power-management backend has no sensor reader at
//	        all -- SR-IOV virtual functions, some older parts. Healthy.
//	ENOENT  the attribute was never created for this ASIC. Nothing learned.
//	EBUSY   amdgpu_pm_dev_state_check: amdgpu_in_reset(), or the device's initialisation
//	        did not reach its default level. Not a state a working device sits in.
//	other   the firmware did not answer. Reported with the errno verbatim, because the
//	        source does not pin down which code a halted or lost device returns and
//	        guessing one would put a confident word on an unmeasured state.
func classifyAMD(p amdProbe) (ml.UnavailableDevice, bool) {
	if p.Driver != "amdgpu" || p.RuntimeStatus != "active" || p.LiveReadErr == nil {
		return ml.UnavailableDevice{}, false
	}

	switch {
	case errors.Is(p.LiveReadErr, syscall.EPERM),
		errors.Is(p.LiveReadErr, syscall.EINVAL),
		errors.Is(p.LiveReadErr, fs.ErrNotExist):
		return ml.UnavailableDevice{}, false
	}

	d := ml.UnavailableDevice{PCIID: p.PCIID, Bus: p.Bus}
	if errors.Is(p.LiveReadErr, syscall.EBUSY) {
		d.Reason = "reset_in_progress"
		d.Detail = "the driver reports the device busy: a GPU reset is under way, or its initialisation did not complete"
		d.Recovery = "a reset normally completes in seconds and this clears on its own. If it is still " +
			"reported on later polls the reset failed; reboot"
	} else {
		d.Reason = "unresponsive"
		d.Detail = "the device's power-management firmware did not answer: " + p.LiveReadErr.Error()
		d.Recovery = "the driver cannot read live state from the device; reboot, and check the kernel " +
			"log for 'GPU reset' lines to see whether a hang preceded this"
	}
	return d, true
}

// readAMDLiveState reads the one attribute that has to ask the device. A variable only so
// tests can make it fail the ways the driver source says it can.
var readAMDLiveState = func(deviceDir string) error {
	_, err := os.ReadFile(filepath.Join(deviceDir, "gpu_busy_percent"))
	return err
}

// amdUnavailableDevices checks every AMD device KFD lists as a compute node and the backend
// did not offer. It reuses readROCmLinuxSysfsDevices -- the same KFD walk discovery uses --
// so the question "which AMD devices are compute devices" has one answer in this package.
func amdUnavailableDevices(usable map[string]bool) []ml.UnavailableDevice {
	devices, err := readROCmLinuxSysfsDevices(sysfsRoot)
	if err != nil {
		// No KFD: no amdgpu compute stack loaded, which on an NVIDIA-only box is the
		// normal case and not worth a log line.
		return nil
	}

	var out []ml.UnavailableDevice
	for _, dev := range devices {
		if dev.pciID == "" || usable[strings.ToLower(dev.pciID)] {
			continue
		}

		dir := filepath.Join(sysfsRoot, "bus", "pci", "devices", dev.pciID)
		probe := amdProbe{
			PCIID:         dev.pciID,
			Integrated:    dev.integrated,
			Driver:        amdDriverName(dir),
			RuntimeStatus: readTrimmed(filepath.Join(dir, "power", "runtime_status")),
			Bus:           busStateAt(dir),
		}
		// Only ask a device that is awake. Reading a suspended one returns EPERM without
		// waking it, so this is not about cost -- it is that there is nothing to learn.
		if probe.Driver == "amdgpu" && probe.RuntimeStatus == "active" {
			probe.LiveReadErr = readAMDLiveState(dir)
		}

		if d, ok := classifyAMD(probe); ok {
			slog.Warn("an AMD GPU is present but cannot be used",
				"pci_id", d.PCIID, "reason", d.Reason, "detail", d.Detail)
			out = append(out, d)
			continue
		}
		// Seen and not a fault. Logged because "was my AMD GPU even noticed?" is the question
		// people ask when ollama does not use one, and the usual answer -- its gfx target is
		// not in this ROCm build -- is not a hardware problem.
		slog.Debug("AMD compute device not offered by the backend, and healthy",
			"pci_id", probe.PCIID, "integrated", probe.Integrated,
			"runtime_status", probe.RuntimeStatus, "live_read_err", probe.LiveReadErr)
	}
	return out
}

// amdDriverName goes through readSysfsDriverName, the reader discovery already uses, rather
// than a second implementation. The first version of this had its own os.Readlink, and the
// two disagreed on a device the test fixture built -- which is the point: one question, one
// answer in this package.
func amdDriverName(deviceDir string) string {
	driver, _ := readSysfsDriverName(filepath.Join(deviceDir, "driver"))
	return driver
}
