package discover

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

type tl = ml.TopologyLink

// These drive the NVLink branches, which the machine this was written on cannot reach: its
// GPUs have no NVLink, so NVML answers NOT_SUPPORTED on link 0. The probe values mirror the
// shapes NVML documents; the one real capture is TestClassifyAPCIeOnlyPairAsThisBoxReads.

// Real: on this box both cards answer NOT_SUPPORTED on link 0 and NVML reports HOSTBRIDGE (30).
func TestClassifyAPCIeOnlyPairAsThisBoxReads(t *testing.T) {
	l := classifyNVIDIAPair("0000:01:00.0", "0000:03:00.0", nvidiaPairProbe{PCIeLevel: 30, NVLinkAbsent: true})
	if l.Type != "pcie" || l.Path != "PHB" || l.PCIePath != "PHB" {
		t.Errorf("got %+v, want pcie via PHB", l)
	}
}

// A 3090 pair bridged by NVLink: direct links counted, and the PCIe route kept underneath.
func TestClassifyADirectNVLinkPair(t *testing.T) {
	l := classifyNVIDIAPair("a", "b", nvidiaPairProbe{PCIeLevel: 30, DirectLinks: 4, Version: 3})
	if l.Type != "nvlink" || l.Path != "NV4" || l.NVLinkCount != 4 || l.NVLinkVersion != 3 {
		t.Errorf("got %+v, want nvlink NV4 v3", l)
	}
	if l.PCIePath != "PHB" {
		t.Errorf("pcie_path = %q; the route beneath the bridge must still be reported", l.PCIePath)
	}
	if l.Bandwidth != 0 {
		t.Error("an NVLink bandwidth was produced; it may only be read, never derived")
	}
}

// Through an NVSwitch no link of one card ends at the other, yet peer-to-peer works over
// NVLink. That must not fall through to "pcie".
func TestClassifyAnNVSwitchPairAsNVLinkWithoutACount(t *testing.T) {
	l := classifyNVIDIAPair("a", "b", nvidiaPairProbe{PCIeLevel: 50, P2PNVLink: true})
	if l.Type != "nvlink" || l.NVLinkCount != 0 || l.Reason == "" {
		t.Errorf("got %+v, want nvlink with no count and a reason", l)
	}
}

// The loud failure: an NVLink call failing for any reason other than "no NVLink on this card"
// makes the pair unknown, never pcie. On a 4x3090 a silent pcie would be a confident lie.
func TestClassifyAFailedNVLinkQueryAsUnknownNotPCIe(t *testing.T) {
	l := classifyNVIDIAPair("a", "b", nvidiaPairProbe{PCIeLevel: 30, FailStatus: 999, FailDetail: "Unknown Error (999)"})
	if l.Type != "unknown" {
		t.Fatalf("type = %q, want unknown", l.Type)
	}
	if l.Reason == "" || !contains(l.Reason, "Unknown Error (999)") {
		t.Errorf("reason %q does not carry the driver's own words", l.Reason)
	}
}

func TestTopologyCoverageAndStatus(t *testing.T) {
	ids := []string{"0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"}
	u := unavailableTopology(ids, "NVML is not available")
	if len(u.Links) != 6 {
		t.Errorf("%d links for 4 GPUs, want 6: every unordered pair exactly once", len(u.Links))
	}
	if u.Status != "unavailable" {
		t.Errorf("status = %q", u.Status)
	}

	ok := classifyNVIDIAPair("a", "b", nvidiaPairProbe{PCIeLevel: 30, NVLinkAbsent: true})
	bad := classifyNVIDIAPair("a", "c", nvidiaPairProbe{PCIeLevel: 30, FailStatus: 15, FailDetail: "GPU is lost"})
	if status, _ := topologyStatus([]tl{ok, bad}); status != "partial" {
		t.Errorf("one unknown pair: status = %q, want partial", status)
	}
}

// Gen5 x8 is this box's slots: 32 GT/s x 128/130 / 8 x 8 lanes = 31.5 GB/s one way. Measured
// peer-to-peer on this box was 27.7 GB/s, which is why this is labelled derived, not measured.
func TestPCIeLinkBandwidthFromTheSpec(t *testing.T) {
	if got := pcieLinkBandwidth(5, 8) / 100_000_000; got != 315 {
		t.Errorf("Gen5 x8 = %.1f GB/s, want 31.5", float64(got)/10)
	}
	if pcieLinkBandwidth(6, 16) != 0 {
		t.Error("Gen6 has FLIT encoding; no single efficiency applies, so it must be omitted")
	}
}
