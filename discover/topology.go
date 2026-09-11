package discover

import (
	"fmt"
	"sort"
	"strings"

	"github.com/ollama/ollama/ml"
)

// nvmlTopologyPath maps nvmlGpuTopologyLevel_t to nvidia-smi's vocabulary. Values read from
// nvml.h (CUDA 13.0): INTERNAL 0, SINGLE 10, MULTIPLE 20, HOSTBRIDGE 30, NODE 40, SYSTEM 50.
func nvmlTopologyPath(level int) string {
	switch level {
	case 0:
		return "INTERNAL"
	case 10:
		return "PIX"
	case 20:
		return "PXB"
	case 30:
		return "PHB"
	case 40:
		return "NODE"
	case 50:
		return "SYS"
	}
	return ""
}

// nvidiaPairProbe is what NVML said about one pair, separated from how it was asked so the
// classification can be tested -- including the NVLink cases, which cannot be reached on a
// machine without NVLink silicon and would otherwise be written entirely blind.
type nvidiaPairProbe struct {
	// PCIeLevel is nvmlDeviceGetTopologyCommonAncestor's answer; PCIeStatus its return code.
	PCIeLevel  int
	PCIeStatus int

	// NVLinkAbsent is true when link 0 of the first device answers NOT_SUPPORTED: the card
	// has no NVLink at all, which is the ordinary state on most GPUs and not a failure.
	NVLinkAbsent bool

	// DirectLinks counts active NVLinks from the first device whose remote end is the
	// second; Version is the NVLink version of those links.
	DirectLinks int
	Version     int

	// P2PNVLink is nvmlDeviceGetP2PStatus with the NVLink capability. It can be true with no
	// direct link, which is how a pair connected through an NVSwitch looks.
	P2PNVLink bool

	// FailStatus and FailDetail record the first NVLink call that failed in a way that is
	// not "this card has no NVLink". Such a pair is unknown, never assumed to be PCIe.
	FailStatus int
	FailDetail string
}

// classifyNVIDIAPair turns what NVML said into a link. Its one rule is that an unanswered
// question produces "unknown" with the driver's own words, never a guess: an empty or wrong
// entry on a 4x3090 would read as "no NVLink here", which is a confident lie.
func classifyNVIDIAPair(a, b string, p nvidiaPairProbe) ml.TopologyLink {
	l := ml.TopologyLink{A: a, B: b}
	if p.PCIeStatus == nvmlSuccess {
		l.PCIePath = nvmlTopologyPath(p.PCIeLevel)
	}

	switch {
	case p.FailStatus != 0:
		l.Type = "unknown"
		l.Reason = fmt.Sprintf("NVML could not report NVLink for this pair: %s", p.FailDetail)
	case p.DirectLinks > 0:
		l.Type = "nvlink"
		l.Path = fmt.Sprintf("NV%d", p.DirectLinks)
		l.NVLinkCount = p.DirectLinks
		l.NVLinkVersion = p.Version
	case p.P2PNVLink:
		// Reachable over NVLink, but no link of the first card ends at the second: the
		// fabric is in between. The link count to the switch is not a count to this peer,
		// so none is reported.
		l.Type = "nvlink"
		l.Reason = "NVLink peer-to-peer is available but not over a direct link, which is how an NVSwitch looks; no link count is reported"
	case p.PCIeStatus != nvmlSuccess:
		l.Type = "unknown"
		l.Reason = "NVML reported neither an NVLink nor a PCIe path for this pair"
	default:
		l.Type = "pcie"
		l.Path = l.PCIePath
	}
	return l
}

// pcieLinkBandwidth derives one direction's bandwidth for a link from the PCIe specification --
// transfer rate times encoding efficiency, per lane -- not from any table of cards. Gen6 is
// omitted: its FLIT encoding does not reduce to one efficiency figure.
func pcieLinkBandwidth(generation, width int) uint64 {
	var gtps, num, den float64
	switch generation {
	case 1:
		gtps, num, den = 2.5, 8, 10
	case 2:
		gtps, num, den = 5, 8, 10
	case 3:
		gtps, num, den = 8, 128, 130
	case 4:
		gtps, num, den = 16, 128, 130
	case 5:
		gtps, num, den = 32, 128, 130
	default:
		return 0
	}
	if width <= 0 {
		return 0
	}
	return uint64(gtps * 1e9 * num / den / 8 * float64(width))
}

// topologyStatus summarises the links: "partial" as soon as any pair is unknown.
func topologyStatus(links []ml.TopologyLink) (string, string) {
	var unknown []string
	for _, l := range links {
		if l.Type == "unknown" {
			unknown = append(unknown, l.A+"<->"+l.B)
		}
	}
	if len(unknown) == 0 {
		return "measured", ""
	}
	return "partial", "could not classify " + strings.Join(unknown, ", ")
}

// unavailableTopology lists every pair as unknown with the same reason, so the coverage rule
// holds even when nothing could be read.
func unavailableTopology(pciIDs []string, detail string) *ml.Topology {
	t := &ml.Topology{Status: "unavailable", Detail: detail, GPUs: pciIDs}
	forEachPair(pciIDs, func(a, b string) {
		t.Links = append(t.Links, ml.TopologyLink{A: a, B: b, Type: "unknown", Reason: detail})
	})
	return t
}

func forEachPair(ids []string, fn func(a, b string)) {
	for i := range ids {
		for j := i + 1; j < len(ids); j++ {
			fn(ids[i], ids[j])
		}
	}
}

func sortedPCIIDs(ids []string) []string {
	out := append([]string(nil), ids...)
	sort.Strings(out)
	return out
}
