//go:build linux

package discover

import (
	"os"
	"path/filepath"
	"testing"
)

// TestSysfsCatchesAGPUNVMLHasAlsoLost is the regression for a hole found the hard way.
//
// The detector was built on "NVML minus CUDA", because a faulted GPU stayed visible to NVML
// with its name and UUID while vanishing from CUDA. That held for the fault as first
// captured. It then STOPPED holding on the same machine on the same day: after the driver
// was torn down and reloaded without the dead card, NVML reported one device too, and the
// detector went quiet with a broken GPU still bolted into the machine.
//
// sysfs never lost it -- 0000:03:00.0, vendor 0x10de, class 0x030000, still bound to the
// nvidia driver -- through the fault, a hung module unload and the reload. So sysfs is the
// superset and NVML is a middle tier that can lose devices.
//
// The lesson generalises past this bug: a detector whose whole job is to notice something
// missing must be anchored to the layer that cannot itself forget.
func TestSysfsCatchesAGPUNVMLHasAlsoLost(t *testing.T) {
	old := sysfsPCIRoot
	sysfsPCIRoot = "testdata/pci"
	t.Cleanup(func() { sysfsPCIRoot = old })

	// The healthy card is the only one the backend offered.
	got := gpusOnlySysfsCanSee(usableSet([]string{"0000:01:00.0"}), nil)

	if len(got) != 1 {
		t.Fatalf("got %d unavailable devices, want 1 (the card sysfs can see and nothing else can)", len(got))
	}
	if got[0].PCIID != "0000:03:00.0" {
		t.Errorf("pci = %q, want %q", got[0].PCIID, "0000:03:00.0")
	}
	if got[0].Reason != "not_reported_by_driver" {
		t.Errorf("reason = %q, want %q", got[0].Reason, "not_reported_by_driver")
	}
	// No status code exists for these -- the driver is not talking about them -- so the
	// report must not imply one was read.
	if got[0].Detail == "GPU requires reset" {
		t.Error("a reason was invented for a device the driver never described")
	}
	if got[0].Bus == nil || !got[0].Bus.Present {
		t.Error("the bus state is the entire evidence that this device exists; it must be reported")
	}
}

// A device the backend already offered must never appear as unavailable, or a healthy
// two-GPU machine reports one of its working cards as broken on every poll.
func TestSysfsDoesNotReportDevicesTheBackendOffered(t *testing.T) {
	old := sysfsPCIRoot
	sysfsPCIRoot = "testdata/pci"
	t.Cleanup(func() { sysfsPCIRoot = old })

	got := gpusOnlySysfsCanSee(usableSet([]string{"0000:01:00.0", "0000:03:00.0"}), nil)
	if len(got) != 0 {
		t.Errorf("got %d unavailable devices, want 0; every enumerated GPU was offered by the backend", len(got))
	}
}

// A device NVML already described must not be reported twice, once with its real reason and
// again as an anonymous sysfs entry.
func TestSysfsDoesNotDuplicateWhatNVMLAlreadyExplained(t *testing.T) {
	old := sysfsPCIRoot
	sysfsPCIRoot = "testdata/pci"
	t.Cleanup(func() { sysfsPCIRoot = old })

	seen := map[string]bool{"0000:03:00.0": true}
	got := gpusOnlySysfsCanSee(usableSet([]string{"0000:01:00.0"}), seen)
	if len(got) != 0 {
		t.Errorf("got %d, want 0: NVML already reported this device with a real reason", len(got))
	}
}

// The kernel's link-speed strings are a closed set (pci_speed_string, drivers/pci/probe.c), so
// the mapping is an exact match on its own vocabulary.
func TestPCIeGenerationFromTheKernelsStrings(t *testing.T) {
	for speed, want := range map[string]int{
		"2.5 GT/s PCIe": 1, "5.0 GT/s PCIe": 2, "8.0 GT/s PCIe": 3,
		"16.0 GT/s PCIe": 4, "32.0 GT/s PCIe": 5, "64.0 GT/s PCIe": 6,
		"Unknown": 0, "": 0, "8x AGP": 0, "133 MHz PCI-X 533": 0,
	} {
		if got := pcieGeneration(speed); got != want {
			t.Errorf("pcieGeneration(%q) = %d, want %d", speed, got, want)
		}
	}
}

// Read from this box: each card reports x16 and the root port above it reports x8, because the
// board splits 16 CPU lanes between two slots. The installed link cannot exceed x8, so that is
// the maximum -- reporting the card's x16 would describe a link that cannot exist.
func TestPCIeMaxLinkIsTheNarrowerOfCardAndPort(t *testing.T) {
	root := t.TempDir()
	port := filepath.Join(root, "devices", "pci0000:00", "0000:00:01.1")
	card := filepath.Join(port, "0000:01:00.0")
	for dir, vals := range map[string][2]string{
		port: {"32.0 GT/s PCIe", "8"},
		card: {"32.0 GT/s PCIe", "16"},
	} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		writeFakeSysfsFile(t, dir, "max_link_speed", vals[0]+"\n")
		writeFakeSysfsFile(t, dir, "max_link_width", vals[1]+"\n")
	}
	busDir := filepath.Join(root, "bus", "pci", "devices")
	if err := os.MkdirAll(busDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(card, filepath.Join(busDir, "0000:01:00.0")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	old := sysfsPCIRoot
	sysfsPCIRoot = busDir
	t.Cleanup(func() { sysfsPCIRoot = old })

	gen, width := PCIeMaxLink("0000:01:00.0")
	if gen != 5 || width != 8 {
		t.Errorf("PCIeMaxLink = Gen%d x%d, want Gen5 x8: the port, not the card, limits the width", gen, width)
	}
}
