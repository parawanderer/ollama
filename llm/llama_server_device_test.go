package llm

import (
	"context"
	"io"
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

// A model placed on the host's second device runs in a child restricted to that
// device, which therefore calls it CUDA0 and reports its buffers under that
// name. Looking the device up by its discovery name ("CUDA1") finds nothing and
// reports the device as unused.
func newSecondDeviceRunner(used uint64) *llamaServerRunner {
	gpus := []ml.DeviceInfo{{
		DeviceID:    ml.DeviceID{ID: "1", Library: "CUDA"},
		Name:        "CUDA1",
		TotalMemory: 100 << 30,
		FreeMemory:  100 << 30,
	}}

	return &llamaServerRunner{
		gpus:           gpus,
		deviceLogNames: ml.RunnerDeviceNames(gpus),
		vramByDevice:   map[string]uint64{"CUDA0": used},
	}
}

func TestVRAMByGPUFilteredChildRenumbers(t *testing.T) {
	const used = 15 << 30

	got := newSecondDeviceRunner(used).VRAMByGPU(ml.DeviceID{ID: "1", Library: "CUDA"})
	if got != used {
		t.Errorf("got %d, want %d: the child reports this device as CUDA0", got, uint64(used))
	}
}

// The same mismatch made a device holding a model look completely free, which is
// worse than a wrong readout: it invites placing more work on a full device.
func TestGetDeviceInfosFilteredChildRenumbers(t *testing.T) {
	const used = 15 << 30

	infos := newSecondDeviceRunner(used).GetDeviceInfos(context.Background())
	if len(infos) != 1 {
		t.Fatalf("expected 1 device, got %d", len(infos))
	}

	want := uint64(100<<30) - used
	if infos[0].FreeMemory != want {
		t.Errorf("free memory: got %d, want %d", infos[0].FreeMemory, want)
	}
}

// The layer placement has the same mismatch: the child logs its layers on "CUDA0", and
// /api/ps used to report exactly that for a model alone on the host's second card.
// The output layer on the CPU keeps the engine's name, which is also ollama's.
func TestLayerPlacementFilteredChildRenumbers(t *testing.T) {
	r := newSecondDeviceRunner(15 << 30)
	r.layerDevice, r.layerSWA = map[int]string{}, map[int]bool{}
	r.memBreakdownByDevice = map[string]api.MemoryBreakdown{}
	w := &memoryParsingWriter{inner: io.Discard, runner: r}
	for _, line := range []string{
		"load_tensors: layer   0 assigned to device CPU, is_swa = 0\n",
		"load_tensors: layer   1 assigned to device CUDA0, is_swa = 0\n",
		"load_tensors: layer   2 assigned to device CUDA0, is_swa = 0\n",
	} {
		w.Write([]byte(line))
	}

	got := r.LayerPlacement()
	want := []api.PlacementRange{
		{Device: "CPU", FirstLayer: 0, LastLayer: 0, Layers: 1},
		{Device: "CUDA1", GPUID: "1", FirstLayer: 1, LastLayer: 2, Layers: 2},
	}
	if got == nil || !reflect.DeepEqual(got.Devices, want) {
		t.Errorf("placement = %+v, want %+v", got, want)
	}
}

// A device the child never mentioned still reports zero rather than matching
// some other device's figure.
func TestVRAMByGPUUnknownDevice(t *testing.T) {
	r := newSecondDeviceRunner(15 << 30)
	if got := r.VRAMByGPU(ml.DeviceID{ID: "7", Library: "CUDA"}); got != 0 {
		t.Errorf("got %d, want 0", got)
	}
}
