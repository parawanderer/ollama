//go:build linux

package discover

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
)

// Topology reports how the offered GPUs connect, pair by pair: NVML for NVIDIA pairs, KFD for
// AMD pairs. A pair spanning two vendors has no shared report and is listed as unknown rather
// than assumed to be PCIe. Returns nil for fewer than one GPU.
func Topology(devices []ml.DeviceInfo) *ml.Topology {
	var nvidia, amd, all []string
	vendor := make(map[string]string)
	for _, d := range devices {
		if d.PCIID == "" {
			continue
		}
		all = append(all, d.PCIID)
		switch d.Library {
		case "CUDA":
			nvidia = append(nvidia, d.PCIID)
			vendor[d.PCIID] = "nvidia"
		case "ROCm":
			amd = append(amd, d.PCIID)
			vendor[d.PCIID] = "amd"
		default:
			vendor[d.PCIID] = strings.ToLower(d.Library)
		}
	}
	if len(all) == 0 {
		return nil
	}
	all = sortedPCIIDs(all)

	known := make(map[[2]string]ml.TopologyLink)
	var details []string
	for _, part := range []*ml.Topology{partTopology(nvidia, nvidiaTopology), partTopology(amd, amdTopology)} {
		if part == nil {
			continue
		}
		if part.Detail != "" {
			details = append(details, part.Detail)
		}
		for _, l := range part.Links {
			known[[2]string{l.A, l.B}] = l
		}
	}

	t := &ml.Topology{GPUs: all}
	forEachPair(all, func(a, b string) {
		if l, ok := known[[2]string{a, b}]; ok {
			t.Links = append(t.Links, l)
			return
		}
		t.Links = append(t.Links, ml.TopologyLink{A: a, B: b, Type: "unknown",
			Reason: "these GPUs are from different vendors (" + vendor[a] + ", " + vendor[b] + "), and no tool reports a link between them"})
	})
	t.Status, t.Detail = topologyStatus(t.Links)
	if t.Status != "measured" && len(details) > 0 {
		t.Detail = strings.Join(details, "; ")
	}
	// A vendor part that could not be read at all makes the whole report unavailable only
	// if nothing else was measured.
	if len(t.Links) > 0 {
		allUnknown := true
		for _, l := range t.Links {
			if l.Type != "unknown" {
				allUnknown = false
				break
			}
		}
		if allUnknown && len(details) > 0 {
			t.Status = "unavailable"
		}
	}
	return t
}

func partTopology(ids []string, fn func([]string) *ml.Topology) *ml.Topology {
	if len(ids) < 2 {
		return nil
	}
	return fn(ids)
}

// topologyTTL bounds how often NVML is opened for this. Links between cards do not change
// while the machine runs; the TTL only exists so a hot-plugged or failed card is noticed.
const topologyTTL = 30 * time.Second

var topologyCache struct {
	sync.Mutex
	key    string
	at     time.Time
	result *ml.Topology
}

// CachedTopology is Topology behind a short cache keyed on the device set, for the same
// reason as CachedUnavailableDevices: opening NVML costs ~17 ms against ~1 ms for /api/info.
func CachedTopology(devices []ml.DeviceInfo) *ml.Topology {
	ids := make([]string, 0, len(devices))
	for _, d := range devices {
		ids = append(ids, d.Library+"/"+d.PCIID)
	}
	key := strings.Join(sortedPCIIDs(ids), ",")

	topologyCache.Lock()
	defer topologyCache.Unlock()
	if topologyCache.key == key && time.Since(topologyCache.at) < topologyTTL {
		return topologyCache.result
	}
	topologyCache.key, topologyCache.at = key, time.Now()
	topologyCache.result = Topology(devices)
	return topologyCache.result
}

// KFD io_link types, from drivers/gpu/drm/amd/amdkfd/kfd_crat.h.
const (
	kfdLinkPCIe = 2
	kfdLinkXGMI = 11
)

// amdTopology reads KFD's io_links for each pair of these AMD GPUs.
//
// Read on a real AMD device before writing this: KFD links are DIRECTED -- each pair appears
// once from each end, with different flags -- and they include GPU-to-CPU links. So links are
// normalised to unordered GPU pairs and CPU nodes are dropped, or the one-entry-per-pair rule
// would count double. Bandwidth is the driver's maximum_bandwidth_mbs (megabytes per second).
func amdTopology(pciIDs []string) *ml.Topology {
	pciIDs = sortedPCIIDs(pciIDs)
	nodeRoot := filepath.Join(sysfsRoot, "class", "kfd", "kfd", "topology", "nodes")
	entries, err := os.ReadDir(nodeRoot)
	if err != nil {
		return unavailableTopology(pciIDs, "KFD topology is not readable: "+err.Error())
	}

	nodePCI := make(map[int]string)
	for _, e := range entries {
		id, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		props, err := readKFDNodeProperties(filepath.Join(nodeRoot, e.Name(), "properties"))
		if err != nil || !props.isGPU() {
			continue
		}
		if dev, err := readROCmDRMDevice(sysfsRoot, props.drmRenderMinor); err == nil && dev.pciID != "" {
			nodePCI[id] = strings.ToLower(dev.pciID)
		}
	}

	type kfdLink struct {
		linkType     int
		bandwidthMBs uint64
	}
	links := make(map[[2]string]kfdLink)
	for id, pci := range nodePCI {
		linkDirs, _ := filepath.Glob(filepath.Join(nodeRoot, strconv.Itoa(id), "io_links", "*"))
		for _, dir := range linkDirs {
			kv := readKFDProperties(filepath.Join(dir, "properties"))
			peer, ok := nodePCI[int(kv["node_to"])]
			if !ok {
				continue // a CPU node, or a GPU not offered
			}
			key := [2]string{pci, peer}
			if key[1] < key[0] {
				key[0], key[1] = key[1], key[0]
			}
			if _, seen := links[key]; !seen {
				links[key] = kfdLink{linkType: int(kv["type"]), bandwidthMBs: kv["max_bandwidth"]}
			}
		}
	}

	visible := make(map[string]bool)
	for _, pci := range nodePCI {
		visible[pci] = true
	}
	t := &ml.Topology{GPUs: pciIDs}
	forEachPair(pciIDs, func(a, b string) {
		l := ml.TopologyLink{A: a, B: b}
		key := [2]string{strings.ToLower(a), strings.ToLower(b)}
		if key[1] < key[0] {
			key[0], key[1] = key[1], key[0]
		}
		kl, ok := links[key]
		switch {
		case !visible[key[0]] || !visible[key[1]]:
			l.Type = "unknown"
			l.Reason = "KFD does not describe one of these GPUs to this process; it may not be passed into the container"
		case !ok:
			l.Type = "unknown"
			l.Reason = "KFD lists no direct link between these GPUs"
		case kl.linkType == kfdLinkXGMI:
			l.Type, l.Path = "xgmi", "XGMI"
		case kl.linkType == kfdLinkPCIe:
			l.Type, l.Path = "pcie", "PCIE"
		default:
			l.Type = "unknown"
			l.Reason = "KFD reports link type " + strconv.Itoa(kl.linkType) + ", which this code does not map"
		}
		if ok && kl.bandwidthMBs > 0 && l.Type != "unknown" {
			l.Bandwidth, l.BandwidthSource = kl.bandwidthMBs*1_000_000, "kfd_io_link"
		}
		t.Links = append(t.Links, l)
	})
	t.Status, t.Detail = topologyStatus(t.Links)
	return t
}

// readKFDProperties parses a KFD "name value" properties file into numbers.
func readKFDProperties(path string) map[string]uint64 {
	out := make(map[string]uint64)
	f, err := os.Open(path)
	if err != nil {
		return out
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) != 2 {
			continue
		}
		if v, err := strconv.ParseUint(fields[1], 10, 64); err == nil {
			out[fields[0]] = v
		}
	}
	return out
}
