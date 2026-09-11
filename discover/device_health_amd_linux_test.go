//go:build linux

package discover

import (
	"io/fs"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// The healthy readings here are this box's own AMD iGPU, read live on 2026-09-11:
//
//	0000:7a:00.0  driver amdgpu  runtime_status active  gpu_busy_percent 0
//	KFD node 1    simd_count 4   gfx_target_version 100306 (gfx1036)  location_id 31232 = 7a:00.0
//
// The fault readings are the error codes drivers/gpu/drm/amd/pm/amdgpu_pm.c and amdgpu_dpm.c
// return, read from source because no faulted AMD card was available. That is weaker than the
// NVIDIA fixture, which was captured from a real fault, and the tests say so by name: they
// check the classification of codes the source proves possible, not behaviour anyone watched.

func healthyAMDProbe() amdProbe {
	return amdProbe{
		PCIID: "0000:7a:00.0", Integrated: true,
		Driver: "amdgpu", RuntimeStatus: "active",
		Bus: healthyBus(),
	}
}

// The case that decides the whole design. This iGPU is a KFD compute node that ollama
// correctly does not use -- gfx1036 is not in its ROCm build -- and it answers every query.
// Reporting it would raise a false alarm on nearly every Ryzen desktop in existence.
func TestAMDDoesNotReportAHealthyDeviceTheBackendDeclined(t *testing.T) {
	if _, report := classifyAMD(healthyAMDProbe()); report {
		t.Error("a healthy AMD iGPU the backend declined was reported as unavailable; absence from " +
			"the backend is ordinary on AMD and is not evidence of a fault")
	}
}

// Each of these is a code a HEALTHY device produces, per the driver source. Every one of them
// is also what a naive "the live read failed, so it is broken" check would have reported.
func TestAMDDoesNotMistakeHealthyErrorCodesForFaults(t *testing.T) {
	for name, err := range map[string]error{
		"EPERM: runtime-suspended, which an idle dGPU does by design":  syscall.EPERM,
		"EINVAL: no sensor reader at all, e.g. an SR-IOV function":     syscall.EINVAL,
		"ENOENT: the attribute was never created for this ASIC":        fs.ErrNotExist,
		"ENOENT via the raw errno, as os.ReadFile actually returns it": syscall.ENOENT,
	} {
		t.Run(name, func(t *testing.T) {
			p := healthyAMDProbe()
			p.LiveReadErr = &fs.PathError{Op: "read", Path: "gpu_busy_percent", Err: err}
			if d, report := classifyAMD(p); report {
				t.Errorf("reported %q for a code a working device returns", d.Reason)
			}
		})
	}
}

func TestAMDReportsAResetInProgress(t *testing.T) {
	p := healthyAMDProbe()
	p.LiveReadErr = &fs.PathError{Op: "read", Path: "gpu_busy_percent", Err: syscall.EBUSY}

	d, report := classifyAMD(p)
	if !report {
		t.Fatal("EBUSY on an active amdgpu device is amdgpu_in_reset() or an incomplete init; it must be reported")
	}
	if d.Reason != "reset_in_progress" {
		t.Errorf("reason = %q, want reset_in_progress", d.Reason)
	}
	if d.PCIID != "0000:7a:00.0" || d.Bus == nil {
		t.Errorf("identity or bus state lost: pci=%q bus=%v", d.PCIID, d.Bus)
	}
}

// For codes the source does not pin down, the errno is reported verbatim rather than mapped
// to a guessed cause.
func TestAMDReportsAnUnexplainedFailureWithoutInventingACause(t *testing.T) {
	p := healthyAMDProbe()
	p.LiveReadErr = &fs.PathError{Op: "read", Path: "gpu_busy_percent", Err: syscall.EIO}

	d, report := classifyAMD(p)
	if !report {
		t.Fatal("EIO from an active amdgpu device is not something a working device returns")
	}
	if d.Reason != "unresponsive" {
		t.Errorf("reason = %q, want unresponsive", d.Reason)
	}
	if !contains(d.Detail, syscall.EIO.Error()) {
		t.Errorf("detail %q does not carry the errno; an unmapped code must be shown, not replaced", d.Detail)
	}
}

// A device handed to vfio-pci for passthrough, or left unbound, is somebody else's. Reporting
// it would alarm every Proxmox user who passes a GPU to a VM.
func TestAMDIgnoresDevicesBoundToAnotherDriver(t *testing.T) {
	for _, driver := range []string{"vfio-pci", ""} {
		p := healthyAMDProbe()
		p.Driver = driver
		p.LiveReadErr = syscall.EBUSY
		if _, report := classifyAMD(p); report {
			t.Errorf("driver %q: reported a device that amdgpu does not own", driver)
		}
	}
}

// A suspended device is not probed at all, so even a stale error must not be read as a fault.
func TestAMDDoesNotJudgeASuspendedDevice(t *testing.T) {
	p := healthyAMDProbe()
	p.RuntimeStatus = "suspended"
	p.LiveReadErr = syscall.EBUSY
	if _, report := classifyAMD(p); report {
		t.Error("a runtime-suspended device was classified; it cannot be asked without waking it")
	}
}

// TestAMDWalksKFDAndSkipsWhatTheBackendOffered exercises the real wiring over a fixture
// tree built with the same helper discovery's own KFD tests use, so the two cannot drift
// apart on what a KFD compute node looks like.
func TestAMDWalksKFDAndSkipsWhatTheBackendOffered(t *testing.T) {
	root := t.TempDir()
	old, oldRead := sysfsRoot, readAMDLiveState
	sysfsRoot = root
	t.Cleanup(func() { sysfsRoot, readAMDLiveState = old, oldRead })

	writeFakeROCmNode(t, root, fakeROCmNode{node: 1, renderMinor: 129, pciID: "0000:7a:00.0", gfxVersion: "100306"})
	writeFakeROCmNode(t, root, fakeROCmNode{node: 2, renderMinor: 130, pciID: "0000:0b:00.0", gfxVersion: "110000"})
	for _, pci := range []string{"0000:7a:00.0", "0000:0b:00.0"} {
		linkAMDDevice(t, root, pci)
	}

	// The dGPU is in a reset; the iGPU is healthy.
	readAMDLiveState = func(dir string) error {
		if filepath.Base(dir) == "0000:0b:00.0" {
			return &fs.PathError{Op: "read", Path: "gpu_busy_percent", Err: syscall.EBUSY}
		}
		return nil
	}

	t.Run("neither offered", func(t *testing.T) {
		got := amdUnavailableDevices(usableSet(nil))
		if len(got) != 1 || got[0].PCIID != "0000:0b:00.0" || got[0].Reason != "reset_in_progress" {
			t.Fatalf("got %+v, want exactly the dGPU in reset; the healthy iGPU must not appear", got)
		}
	})

	t.Run("the faulted one was offered", func(t *testing.T) {
		if got := amdUnavailableDevices(usableSet([]string{"0000:0b:00.0"})); len(got) != 0 {
			t.Errorf("got %+v; a device the backend offered is never unavailable", got)
		}
	})
}

// linkAMDDevice gives a fake device the two parts real sysfs has and the shared helper does
// not: the /sys/bus/pci/devices/<id> symlink and a runtime power state.
func linkAMDDevice(t *testing.T, root, pci string) {
	t.Helper()
	target := filepath.Join(root, "devices", pci)
	busDir := filepath.Join(root, "bus", "pci", "devices")
	if err := os.MkdirAll(busDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(busDir, pci)); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(target, "power"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFakeSysfsFile(t, filepath.Join(target, "power"), "runtime_status", "active\n")
}
