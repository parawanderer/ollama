package discover

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

// The readings in these tests are transcribed from a real fault, captured live before the
// GPU was reset: reports/captures/gpu1-gsp-fault-2026-09-11.json in the slop-zone repo. The
// state cannot be reproduced on demand -- it took twelve days of uptime to happen once --
// so a fixture recalled from memory would be the only alternative, and that is exactly how
// a matcher that matches nothing gets written.
//
// What was captured, on a GPU whose GSP firmware had hung:
//
//	nvmlDeviceGetCount_v2 = 2          cuDeviceGetCount = 1
//	device 1: name and UUID readable, temperature -> 16 "GPU requires reset"
//	          memory -> SUCCESS, free identical to the healthy card
//	          sysfs: present, D0, 5.0 GT/s x8, AER counters all zero

func healthyBus() *ml.DeviceBusState {
	return &ml.DeviceBusState{Present: true, LinkSpeed: "5.0 GT/s PCIe", LinkWidth: 8, PowerState: "D0"}
}

func TestClassifyReportsAResetRequiredGPUAsRecoverable(t *testing.T) {
	got := classifyUnavailable(deviceProbe{
		PCIID:            "0000:03:00.0",
		Name:             "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
		UUID:             "GPU-ea77999f-c55b-d5ed-bcae-71115e033a47",
		Status:           nvmlErrorResetRequired,
		DetailFromDriver: "GPU requires reset",
		Bus:              healthyBus(),
	})

	if got.Reason != "reset_required" {
		t.Errorf("reason = %q, want %q", got.Reason, "reset_required")
	}
	if got.Detail != "GPU requires reset" {
		t.Errorf("detail = %q; the driver's own wording must pass through unparaphrased so it "+
			"can be searched for verbatim", got.Detail)
	}
	if got.Recovery == "" {
		t.Error("a reset-required GPU has a known fix and must say so; an empty Recovery means " +
			"'no known action', which would send an operator looking for a hardware fault")
	}
	// The identity has to survive, or a two-GPU machine cannot say WHICH card is broken.
	if got.UUID == "" || got.PCIID == "" {
		t.Errorf("identity lost: pci=%q uuid=%q", got.PCIID, got.UUID)
	}
}

// TestClassifyDistinguishesAMissingCardFromABrokenOne is the reason the bus state is read
// at all. The driver reports both as a device it cannot reach; only sysfs knows whether the
// device is still electrically present, and the two have completely different fixes --
// reseat and check power, versus reset the GPU.
func TestClassifyDistinguishesAMissingCardFromABrokenOne(t *testing.T) {
	onBus := classifyUnavailable(deviceProbe{
		Status: nvmlErrorGPUIsLost, DetailFromDriver: "GPU is lost", Bus: healthyBus(),
	})
	offBus := classifyUnavailable(deviceProbe{
		Status: nvmlErrorGPUIsLost, DetailFromDriver: "GPU is lost",
		Bus: &ml.DeviceBusState{Present: false},
	})

	if onBus.Reason != "lost" {
		t.Errorf("still enumerated: reason = %q, want %q", onBus.Reason, "lost")
	}
	if offBus.Reason != "not_enumerated" {
		t.Errorf("gone from the bus: reason = %q, want %q", offBus.Reason, "not_enumerated")
	}
	if onBus.Reason == offBus.Reason {
		t.Error("a card that fell off the bus and a card whose driver gave up are reported " +
			"identically; the bus state is being ignored and the operator is sent to the wrong fix")
	}
}

// A device that answers everything and is still not offered is NOT a hardware fault, and
// saying so would send someone to reseat a working card. The usual cause is a visibility
// filter.
func TestClassifyDoesNotCallAHealthyDeviceBroken(t *testing.T) {
	got := classifyUnavailable(deviceProbe{Status: nvmlSuccess, Bus: healthyBus()})

	if got.Reason != "not_offered_by_backend" {
		t.Errorf("reason = %q, want %q", got.Reason, "not_offered_by_backend")
	}
	for _, bad := range []string{"reset", "reseat", "power-cycle", "RMA"} {
		if contains(got.Recovery, bad) {
			t.Errorf("Recovery %q sends an operator at the hardware for a device that answered "+
				"every query", got.Recovery)
		}
	}
}

// PCIe errors point somewhere else entirely -- the slot, the riser, the cabling -- so they
// have to be surfaced even when the driver has already named a GPU-level reason.
func TestClassifyFlagsPCIeErrorsAlongsideTheDriverReason(t *testing.T) {
	bus := healthyBus()
	bus.FatalErrors = 3

	got := classifyUnavailable(deviceProbe{Status: nvmlErrorResetRequired, Bus: bus})

	if got.Reason != "reset_required" {
		t.Errorf("reason = %q, want the driver's verdict preserved", got.Reason)
	}
	if !contains(got.Recovery, "PCIe errors") {
		t.Errorf("Recovery = %q, want it to mention the link errors; a card blamed for a bad "+
			"slot gets replaced and the fault follows the slot", got.Recovery)
	}
}

func contains(haystack, needle string) bool {
	return len(haystack) >= len(needle) && (haystack == needle || indexOf(haystack, needle) >= 0)
}

func indexOf(h, n string) int {
	for i := 0; i+len(n) <= len(h); i++ {
		if h[i:i+len(n)] == n {
			return i
		}
	}
	return -1
}

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
