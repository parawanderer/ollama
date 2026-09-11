//go:build linux

package discover

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// The fixture follows the io_links read from this box's AMD iGPU:
//
//	nodes/0/io_links/0: type 2 node_from 0 node_to 1 weight 20 max_bandwidth 32000 flags 3
//	nodes/1/io_links/0: type 2 node_from 1 node_to 0 weight 20 max_bandwidth 32000 flags 1
//
// -- directed, both ends listed, and a CPU node among them -- extended to two GPUs joined by
// xGMI (type 11), which the box does not have. The xGMI values are therefore constructed.
func TestAMDTopologyNormalisesDirectedLinksAndDropsTheCPU(t *testing.T) {
	root := t.TempDir()
	old := sysfsRoot
	sysfsRoot = root
	t.Cleanup(func() { sysfsRoot = old })

	// node 0 is the CPU (no drm node, simd 0); nodes 1 and 2 are GPUs.
	cpu := filepath.Join(root, "class", "kfd", "kfd", "topology", "nodes", "0")
	if err := os.MkdirAll(cpu, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFakeSysfsFile(t, cpu, "properties", "cpu_cores_count 32\nsimd_count 0\nvendor_id 0\ndevice_id 0\ndrm_render_minor 0\n")
	writeFakeROCmNode(t, root, fakeROCmNode{node: 1, renderMinor: 129, pciID: "0000:0a:00.0", gfxVersion: "90402"})
	writeFakeROCmNode(t, root, fakeROCmNode{node: 2, renderMinor: 130, pciID: "0000:0b:00.0", gfxVersion: "90402"})

	link := func(from, idx, to, typ, bw int) {
		dir := filepath.Join(root, "class", "kfd", "kfd", "topology", "nodes", strconv.Itoa(from), "io_links", strconv.Itoa(idx))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		writeFakeSysfsFile(t, dir, "properties", "type "+strconv.Itoa(typ)+"\nnode_from "+strconv.Itoa(from)+
			"\nnode_to "+strconv.Itoa(to)+"\nweight 15\nmax_bandwidth "+strconv.Itoa(bw)+"\n")
	}
	link(1, 0, 0, kfdLinkPCIe, 32000) // GPU to CPU, both directions
	link(0, 0, 1, kfdLinkPCIe, 32000)
	link(1, 1, 2, kfdLinkXGMI, 50000) // GPU to GPU over xGMI, both directions
	link(2, 0, 1, kfdLinkXGMI, 50000)

	got := amdTopology([]string{"0000:0b:00.0", "0000:0a:00.0"})
	if len(got.Links) != 1 {
		t.Fatalf("%d links, want 1: the directed pair must collapse to one entry and the CPU links must go: %+v", len(got.Links), got.Links)
	}
	l := got.Links[0]
	if l.Type != "xgmi" || l.Bandwidth != 50_000_000_000 || l.BandwidthSource != "kfd_io_link" {
		t.Errorf("got %+v, want xgmi at 50 GB/s from the driver", l)
	}
	if got.Status != "measured" {
		t.Errorf("status = %q", got.Status)
	}
}
