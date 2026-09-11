//go:build linux

package discover

/*
#cgo linux LDFLAGS: -ldl

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void * ollama_dlopen(const char * path, int global) {
	return dlopen(path, RTLD_NOW | (global ? RTLD_GLOBAL : RTLD_LOCAL));
}

static void * ollama_dlsym(void * handle, const char * name) {
	return dlsym(handle, name);
}

static const char * ollama_dlerror(void) {
	const char * err = dlerror();
	return err ? err : "";
}

typedef void * (*ollama_ggml_backend_load_fn)(const char *);
typedef size_t (*ollama_ggml_backend_reg_dev_count_fn)(void *);
typedef void * (*ollama_ggml_backend_reg_dev_get_fn)(void *, size_t);
typedef const char * (*ollama_ggml_backend_reg_name_fn)(void *);
typedef void (*ollama_ggml_backend_dev_get_props_fn)(void *, void *);

static void * ollama_call_ggml_backend_load(void * fn, const char * path) {
	return ((ollama_ggml_backend_load_fn) fn)(path);
}

static size_t ollama_call_ggml_backend_reg_dev_count(void * fn, void * reg) {
	return ((ollama_ggml_backend_reg_dev_count_fn) fn)(reg);
}

static void * ollama_call_ggml_backend_reg_dev_get(void * fn, void * reg, size_t index) {
	return ((ollama_ggml_backend_reg_dev_get_fn) fn)(reg, index);
}

static const char * ollama_call_ggml_backend_reg_name(void * fn, void * reg) {
	return ((ollama_ggml_backend_reg_name_fn) fn)(reg);
}

static void ollama_call_ggml_backend_dev_get_props(void * fn, void * dev, void * props) {
	((ollama_ggml_backend_dev_get_props_fn) fn)(dev, props);
}

static const char * ollama_cstr_from_uintptr(uintptr_t ptr) {
	return (const char *) ptr;
}

typedef int (*ollama_cu_init_fn)(unsigned int);
typedef int (*ollama_cu_driver_get_version_fn)(int *);
typedef int (*ollama_cu_device_get_count_fn)(int *);
typedef int (*ollama_cu_device_get_fn)(int *, int);
typedef int (*ollama_cu_device_get_attribute_fn)(int *, int, int);
typedef int (*ollama_cu_device_get_name_fn)(char *, int, int);
typedef int (*ollama_cu_device_total_mem_fn)(size_t *, int);
typedef int (*ollama_cu_device_get_pci_bus_id_fn)(char *, int, int);

static int ollama_call_cu_init(void * fn) {
	return ((ollama_cu_init_fn) fn)(0);
}

static int ollama_call_cu_driver_get_version(void * fn, int * version) {
	return ((ollama_cu_driver_get_version_fn) fn)(version);
}

static int ollama_call_cu_device_get_count(void * fn, int * count) {
	return ((ollama_cu_device_get_count_fn) fn)(count);
}

static int ollama_call_cu_device_get(void * fn, int * device, int index) {
	return ((ollama_cu_device_get_fn) fn)(device, index);
}

static int ollama_call_cu_device_get_attribute(void * fn, int * value, int attr, int device) {
	return ((ollama_cu_device_get_attribute_fn) fn)(value, attr, device);
}

static int ollama_call_cu_device_get_name(void * fn, char * name, int len, int device) {
	return ((ollama_cu_device_get_name_fn) fn)(name, len, device);
}

static int ollama_call_cu_device_total_mem(void * fn, size_t * total, int device) {
	return ((ollama_cu_device_total_mem_fn) fn)(total, device);
}

static int ollama_call_cu_device_get_pci_bus_id(void * fn, char * pci, int len, int device) {
	return ((ollama_cu_device_get_pci_bus_id_fn) fn)(pci, len, device);
}

typedef int (*ollama_nvml_init_fn)(void);
typedef int (*ollama_nvml_shutdown_fn)(void);
typedef int (*ollama_nvml_system_get_driver_version_fn)(char *, unsigned int);

static int ollama_call_nvml_init(void * fn) {
	return ((ollama_nvml_init_fn) fn)();
}

static int ollama_call_nvml_shutdown(void * fn) {
	return ((ollama_nvml_shutdown_fn) fn)();
}

static int ollama_call_nvml_system_get_driver_version(void * fn, char * version, unsigned int len) {
	return ((ollama_nvml_system_get_driver_version_fn) fn)(version, len);
}

// nvmlMemory_t as of NVML 8. Only the leading three fields are read, and NVML has only
// ever appended to this structure, so a newer library filling more of it is harmless.
typedef struct {
	unsigned long long total;
	unsigned long long free;
	unsigned long long used;
} ollama_nvml_memory_t;

typedef int (*ollama_nvml_device_get_handle_by_pci_bus_id_fn)(const char *, void **);
typedef int (*ollama_nvml_device_get_memory_info_fn)(void *, ollama_nvml_memory_t *);

static int ollama_call_nvml_device_get_handle_by_pci_bus_id(void * fn, const char * pci, void ** device) {
	return ((ollama_nvml_device_get_handle_by_pci_bus_id_fn) fn)(pci, device);
}

// nvmlProcessInfo_t as of NVML v3, which is what nvmlDeviceGetComputeRunningProcesses_v3
// expects. The trailing instance ids exist for MIG and are read but unused here.
typedef struct {
	unsigned int pid;
	unsigned long long usedGpuMemory;
	unsigned int gpuInstanceId;
	unsigned int computeInstanceId;
} ollama_nvml_process_info_t;

typedef int (*ollama_nvml_device_get_compute_procs_fn)(void *, unsigned int *, ollama_nvml_process_info_t *);

// Returns the number of processes written into pids/used, or a negative NVML status.
static int ollama_call_nvml_device_get_compute_procs(void * fn, void * device, unsigned int max,
                                                     unsigned int * pids, unsigned long long * used) {
	ollama_nvml_process_info_t infos[64];
	unsigned int count = max > 64 ? 64 : max;
	int ret = ((ollama_nvml_device_get_compute_procs_fn) fn)(device, &count, infos);
	if (ret != 0) {
		return -ret;
	}
	for (unsigned int i = 0; i < count; i++) {
		pids[i] = infos[i].pid;
		used[i] = infos[i].usedGpuMemory;
	}
	return (int) count;
}

// nvmlPciInfo_t. The leading fields are stable and busIdLegacy is the "0000:01:00.0" form
// ollama already uses for PCIID. The layout was verified against the running library rather
// than transcribed from a header, and the trailing pad absorbs whatever NVML has appended
// since -- it only ever appends.
typedef struct {
	char busIdLegacy[16];
	unsigned int domain;
	unsigned int bus;
	unsigned int device;
	unsigned int pciDeviceId;
	unsigned int pciSubSystemId;
	char busId[32];
	char pad[64];
} ollama_nvml_pci_info_t;

typedef int (*ollama_nvml_device_get_count_fn)(unsigned int *);
typedef int (*ollama_nvml_device_get_handle_by_index_fn)(unsigned int, void **);
typedef int (*ollama_nvml_device_get_name_fn)(void *, char *, unsigned int);
typedef int (*ollama_nvml_device_get_uuid_fn)(void *, char *, unsigned int);
typedef int (*ollama_nvml_device_get_pci_info_fn)(void *, ollama_nvml_pci_info_t *);
typedef int (*ollama_nvml_device_get_temperature_fn)(void *, int, unsigned int *);
typedef const char * (*ollama_nvml_error_string_fn)(int);

static int ollama_call_nvml_device_get_count(void * fn, unsigned int * count) {
	return ((ollama_nvml_device_get_count_fn) fn)(count);
}

static int ollama_call_nvml_device_get_handle_by_index(void * fn, unsigned int index, void ** device) {
	return ((ollama_nvml_device_get_handle_by_index_fn) fn)(index, device);
}

static int ollama_call_nvml_device_get_name(void * fn, void * device, char * name, unsigned int len) {
	return ((ollama_nvml_device_get_name_fn) fn)(device, name, len);
}

static int ollama_call_nvml_device_get_uuid(void * fn, void * device, char * uuid, unsigned int len) {
	return ((ollama_nvml_device_get_uuid_fn) fn)(device, uuid, len);
}

static int ollama_call_nvml_device_get_pci_bus_id_legacy(void * fn, void * device, char * out, unsigned int len) {
	ollama_nvml_pci_info_t info;
	memset(&info, 0, sizeof(info));
	int ret = ((ollama_nvml_device_get_pci_info_fn) fn)(device, &info);
	if (ret == 0) {
		strncpy(out, info.busIdLegacy, len - 1);
		out[len - 1] = 0;
	}
	return ret;
}

// Temperature is the health probe. See the note on deviceProbe.Status in device_health.go
// for why it is this call and not memory, power or utilization.
//
// The 0 is NVML_TEMPERATURE_GPU. Do not write that as an inline C comment: this whole
// block is one Go comment, Go comments do not nest, and a nested terminator ends the cgo
// preamble right here. The errors then surface as Go syntax errors twenty lines below,
// pointing at code that is completely fine.
static int ollama_call_nvml_device_get_temperature(void * fn, void * device, unsigned int * temp) {
	return ((ollama_nvml_device_get_temperature_fn) fn)(device, 0, temp);
}

typedef int (*ollama_nvml_device_get_uint_fn)(void *, unsigned int *);
typedef int (*ollama_nvml_device_get_max_clock_fn)(void *, int, unsigned int *);

static int ollama_call_nvml_device_get_uint(void * fn, void * device, unsigned int * value) {
	return ((ollama_nvml_device_get_uint_fn) fn)(device, value);
}

// The 2 is NVML_CLOCK_MEM (graphics 0, sm 1, mem 2, video 3). Written as a literal because
// this block is one Go comment and an inline C comment would end it.
static int ollama_call_nvml_device_get_max_mem_clock(void * fn, void * device, unsigned int * mhz) {
	return ((ollama_nvml_device_get_max_clock_fn) fn)(device, 2, mhz);
}

static const char * ollama_call_nvml_error_string(void * fn, int status) {
	return ((ollama_nvml_error_string_fn) fn)(status);
}

static int ollama_call_nvml_device_get_memory_info(void * fn, void * device, unsigned long long * total, unsigned long long * free) {
	ollama_nvml_memory_t memory = {0};
	int ret = ((ollama_nvml_device_get_memory_info_fn) fn)(device, &memory);
	if (ret == 0) {
		*total = memory.total;
		*free = memory.free;
	}
	return ret;
}

*/
import "C"

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/ml"
)

const (
	cuSuccess                               = 0
	cuDeviceAttributeComputeCapabilityMajor = 75
	cuDeviceAttributeComputeCapabilityMinor = 76
	cuDeviceAttributeIntegrated             = 18
)

type dlHandle struct {
	ptr unsafe.Pointer
}

func runPlatformNativeProbe(ctx context.Context, libDirs []string) ([]nativeProbeDevice, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	ggmlDevices, ggmlErr := probeGGMLDevicesLinux(libDirs)
	var cudaDevices []nativeProbeDevice
	var cudaErr error
	if nativeProbeHasCUDA(libDirs) {
		cudaDevices, cudaErr = probeCUDADriverLinux()
	}
	var rocmDevices []nativeProbeDevice
	var rocmErr error
	if nativeProbeHasROCm(libDirs) {
		rocmDevices, rocmErr = probeROCmSysfsLinux()
	}

	devices := mergeNativeProbeDevices(mergeNativeProbeDevices(ggmlDevices, cudaDevices), rocmDevices)
	if len(devices) > 0 {
		return devices, nil
	}

	if ggmlErr != nil {
		return nil, ggmlErr
	}
	if rocmErr != nil {
		return nil, rocmErr
	}
	return nil, cudaErr
}

func probeGGMLDevicesLinux(libDirs []string) ([]nativeProbeDevice, error) {
	if len(libDirs) == 0 {
		return nil, errors.New("no library directories provided")
	}

	baseDir := libDirs[0]
	if baseDir == "" {
		return nil, errors.New("empty GGML library directory")
	}

	base, err := dlopen(ggmlLibraryFile(baseDir, "ggml-base"), true)
	if err != nil {
		return nil, err
	}

	ggml, err := dlopen(ggmlLibraryFile(baseDir, "ggml"), true)
	if err != nil {
		return nil, err
	}

	backendLoad, err := dlsym(ggml, "ggml_backend_load")
	if err != nil {
		return nil, err
	}
	regDevCount, err := dlsym(base, "ggml_backend_reg_dev_count")
	if err != nil {
		return nil, err
	}
	regDevGet, err := dlsym(base, "ggml_backend_reg_dev_get")
	if err != nil {
		return nil, err
	}
	regName, err := dlsym(base, "ggml_backend_reg_name")
	if err != nil {
		return nil, err
	}
	devGetProps, err := dlsym(base, "ggml_backend_dev_get_props")
	if err != nil {
		return nil, err
	}

	var devices []nativeProbeDevice
	for _, backendPath := range nativeProbeBackendFiles(libDirs) {
		reg := callGGMLBackendLoad(backendLoad, backendPath)
		if reg == nil {
			continue
		}

		library := ggmlProbeLibraryName(callGGMLRegName(regName, reg))
		count := int(callGGMLRegDevCount(regDevCount, reg))
		for i := range count {
			dev := callGGMLRegDevGet(regDevGet, reg, i)
			if dev == nil {
				continue
			}
			props := callGGMLDeviceProps(devGetProps, dev)
			if props.MemoryTotal == 0 {
				continue
			}
			devices = append(devices, nativeProbeDevice{
				Library:             library,
				Index:               i,
				IndexMatchesBackend: true,
				Name:                cString(props.Name),
				Description:         cString(props.Description),
				DeviceID:            cString(props.DeviceID),
				Integrated:          ggmlDeviceTypeIntegrated(props.Type),
				IntegratedKnown: props.Type == ggmlBackendDeviceTypeGPU ||
					props.Type == ggmlBackendDeviceTypeIGPU,
				TotalMemory: uint64(props.MemoryTotal),
				FreeMemory:  uint64(props.MemoryFree),
			})
			slog.Debug("GGML GPU device type", "library", library, "index", i, "ggml_type", props.Type, "integrated", ggmlDeviceTypeIntegrated(props.Type))
		}
	}

	return devices, nil
}

func probeCUDADriverLinux() ([]nativeProbeDevice, error) {
	cuda, err := dlopenFirst([]string{"libcuda.so.1", "libcuda.so"}, false)
	if err != nil {
		return nil, err
	}

	cuInit, err := dlsym(cuda, "cuInit")
	if err != nil {
		return nil, err
	}
	cuDriverGetVersion, err := dlsym(cuda, "cuDriverGetVersion")
	if err != nil {
		return nil, err
	}
	cuDeviceGetCount, err := dlsym(cuda, "cuDeviceGetCount")
	if err != nil {
		return nil, err
	}
	cuDeviceGet, err := dlsym(cuda, "cuDeviceGet")
	if err != nil {
		return nil, err
	}
	cuDeviceGetAttribute, err := dlsym(cuda, "cuDeviceGetAttribute")
	if err != nil {
		return nil, err
	}
	cuDeviceGetName, err := dlsym(cuda, "cuDeviceGetName")
	if err != nil {
		return nil, err
	}
	cuDeviceTotalMem, err := dlsymAny(cuda, "cuDeviceTotalMem_v2", "cuDeviceTotalMem")
	if err != nil {
		return nil, err
	}
	cuDeviceGetPCIBusID, _ := dlsym(cuda, "cuDeviceGetPCIBusId")

	if ret := C.ollama_call_cu_init(cuInit); ret != cuSuccess {
		return nil, fmt.Errorf("cuInit failed: %d", int(ret))
	}

	var driverVersion C.int
	driverMajor, driverMinor := 0, 0
	if ret := C.ollama_call_cu_driver_get_version(cuDriverGetVersion, &driverVersion); ret == cuSuccess {
		version := int(driverVersion)
		driverMajor = version / 1000
		driverMinor = (version - driverMajor*1000) / 10
	}

	nvidiaDriverMajor := 0
	if driver, err := probeNVIDIADriverMajorLinux(); err == nil {
		nvidiaDriverMajor = driver
	}

	var count C.int
	if ret := C.ollama_call_cu_device_get_count(cuDeviceGetCount, &count); ret != cuSuccess {
		return nil, fmt.Errorf("cuDeviceGetCount failed: %d", int(ret))
	}

	deviceCount := int(count)
	devices := make([]nativeProbeDevice, 0, deviceCount)
	for i := range deviceCount {
		var device C.int
		if ret := C.ollama_call_cu_device_get(cuDeviceGet, &device, C.int(i)); ret != cuSuccess {
			continue
		}

		major := cudaDeviceAttribute(cuDeviceGetAttribute, cuDeviceAttributeComputeCapabilityMajor, device)
		minor := cudaDeviceAttribute(cuDeviceGetAttribute, cuDeviceAttributeComputeCapabilityMinor, device)
		integrated := cudaDeviceAttribute(cuDeviceGetAttribute, cuDeviceAttributeIntegrated, device) == 1

		var name [128]C.char
		_ = C.ollama_call_cu_device_get_name(cuDeviceGetName, &name[0], C.int(len(name)), device)

		var total C.size_t
		_ = C.ollama_call_cu_device_total_mem(cuDeviceTotalMem, &total, device)

		pci := ""
		if cuDeviceGetPCIBusID != nil {
			var pciBuf [32]C.char
			if ret := C.ollama_call_cu_device_get_pci_bus_id(cuDeviceGetPCIBusID, &pciBuf[0], C.int(len(pciBuf)), device); ret == cuSuccess {
				pci = strings.ToLower(C.GoString(&pciBuf[0]))
			}
		}

		devices = append(devices, nativeProbeDevice{
			Library:             "CUDA",
			Index:               i,
			IndexMatchesBackend: true,
			Description:         C.GoString(&name[0]),
			DeviceID:            pci,
			Integrated:          integrated,
			IntegratedKnown:     true,
			TotalMemory:         uint64(total),
			ComputeMajor:        major,
			ComputeMinor:        minor,
			CUDADriverMajor:     driverMajor,
			CUDADriverMinor:     driverMinor,
			NVIDIADriverMajor:   nvidiaDriverMajor,
		})
	}

	// CUDA reports what a context can address; NVML reports what the card holds. Fill the
	// latter in where it is available, and leave it zero where it is not.
	pciIDs := make([]string, 0, len(devices))
	for _, d := range devices {
		pciIDs = append(pciIDs, d.DeviceID)
	}
	for pci, iface := range nvmlMemoryInterfaceByPCI(pciIDs) {
		for i := range devices {
			if devices[i].DeviceID == pci {
				devices[i].MemoryBusWidthBits = iface.busWidthBits
				devices[i].MemoryClockMaxMHz = iface.clockMaxMHz
			}
		}
	}
	if physical := nvmlPhysicalMemoryByPCI(pciIDs); len(physical) > 0 {
		for i := range devices {
			if total, ok := physical[devices[i].DeviceID]; ok && total >= devices[i].TotalMemory {
				devices[i].PhysicalMemory = total
			}
		}
	}

	return devices, nil
}

func probeROCmSysfsLinux() ([]nativeProbeDevice, error) {
	sysfsDevices, err := readROCmLinuxSysfsDevices("/sys")
	if err != nil {
		return nil, err
	}

	override := hsaOverrideGFXTarget()
	// Sysfs stays in physical KFD order; ROCm visibility envs can reindex the
	// backend device list, so filtered sysfs data must merge by PCI ID only.
	backendIndex := !rocmVisibleDevicesEnvSet()
	devices := make([]nativeProbeDevice, 0, len(sysfsDevices))
	for i, sysfsDevice := range sysfsDevices {
		gfxTarget := sysfsDevice.gfxTarget
		if override != "" {
			gfxTarget = override
		}
		devices = append(devices, nativeProbeDevice{
			Library:             "ROCm",
			Index:               i,
			IndexMatchesBackend: backendIndex,
			DeviceID:            sysfsDevice.pciID,
			Integrated:          sysfsDevice.integrated,
			IntegratedKnown:     sysfsDevice.known,
			GFXTarget:           gfxTarget,
		})
	}
	return devices, nil
}

func rocmVisibleDevicesEnvSet() bool {
	for _, name := range []string{"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"} {
		if os.Getenv(name) != "" {
			return true
		}
	}
	return false
}

func probeNVIDIADriverMajorLinux() (int, error) {
	nvml, err := dlopenFirst([]string{"libnvidia-ml.so.1", "libnvidia-ml.so"}, false)
	if err != nil {
		return 0, err
	}
	initFn, err := dlsym(nvml, "nvmlInit_v2")
	if err != nil {
		return 0, err
	}
	shutdownFn, err := dlsym(nvml, "nvmlShutdown")
	if err != nil {
		return 0, err
	}
	driverFn, err := dlsym(nvml, "nvmlSystemGetDriverVersion")
	if err != nil {
		return 0, err
	}
	if ret := C.ollama_call_nvml_init(initFn); ret != 0 {
		return 0, fmt.Errorf("nvmlInit_v2 failed: %d", int(ret))
	}
	defer C.ollama_call_nvml_shutdown(shutdownFn)

	var version [80]C.char
	if ret := C.ollama_call_nvml_system_get_driver_version(driverFn, &version[0], C.uint(len(version))); ret != 0 {
		return 0, fmt.Errorf("nvmlSystemGetDriverVersion failed: %d", int(ret))
	}
	return parseNVIDIADriverMajor(C.GoString(&version[0]))
}

// nvmlPhysicalMemoryByPCI reports the memory each device holds according to NVML, keyed by
// PCI bus ID.
//
// This is a different quantity from the one CUDA reports, and larger. cuDeviceTotalMem
// returns what a CUDA context can address, which excludes memory the driver has reserved
// for itself; NVML reports the device's framebuffer. On an RTX PRO 6000 the two are
// 94.97 GiB and 95.59 GiB — a 638 MiB gap that is not usable for models but is present on
// the card, and that a user reading nvidia-smi will expect to see accounted for.
//
// A failure here is not an error: NVML may be absent, too old, or refuse a device. Callers
// fall back to the CUDA figure, which is the one every decision is made from anyway.
// ComputeProcesses reports which processes hold memory on each device, keyed by PCI ID.
//
// It is deliberately not part of discovery. Discovery describes what devices exist and is
// cached; this changes whenever a process starts or stops, so it is read at the moment it
// is reported or it describes the past.
func ComputeProcesses(pciIDs []string) map[string][]ml.DeviceProcess {
	return nvmlComputeProcessesByPCI(pciIDs)
}

// nvmlComputeProcessesByPCI reports which processes hold memory on each device.
//
// This is what turns the difference between a device's used memory and the sum of what
// ollama has loaded from a number into an attribution. Without it a reporting client can
// only name that residual by magnitude -- small means the driver context, large means
// something else is running -- which is a guess that happens to hold on one machine.
func nvmlComputeProcessesByPCI(pciIDs []string) map[string][]ml.DeviceProcess {
	if len(pciIDs) == 0 {
		return nil
	}

	nvml, err := dlopenFirst([]string{"libnvidia-ml.so.1", "libnvidia-ml.so"}, false)
	if err != nil {
		return nil
	}

	initFn, initErr := dlsym(nvml, "nvmlInit_v2")
	shutdownFn, shutdownErr := dlsym(nvml, "nvmlShutdown")
	handleFn, handleErr := dlsym(nvml, "nvmlDeviceGetHandleByPciBusId_v2")
	procsFn, procsErr := dlsym(nvml, "nvmlDeviceGetComputeRunningProcesses_v3")
	if err := cmp.Or(initErr, shutdownErr, handleErr, procsErr); err != nil {
		slog.Debug("NVML cannot report compute processes", "error", err)
		return nil
	}

	if ret := C.ollama_call_nvml_init(initFn); ret != 0 {
		return nil
	}
	defer C.ollama_call_nvml_shutdown(shutdownFn)

	out := make(map[string][]ml.DeviceProcess, len(pciIDs))
	for _, pci := range pciIDs {
		if pci == "" {
			continue
		}

		cpci := C.CString(pci)
		var handle unsafe.Pointer
		ret := C.ollama_call_nvml_device_get_handle_by_pci_bus_id(handleFn, cpci, &handle)
		C.free(unsafe.Pointer(cpci))
		if ret != 0 {
			continue
		}

		const maxProcs = 64
		var pids [maxProcs]C.uint
		var used [maxProcs]C.ulonglong
		n := C.ollama_call_nvml_device_get_compute_procs(procsFn, handle, maxProcs, &pids[0], &used[0])
		if n < 0 {
			slog.Debug("nvmlDeviceGetComputeRunningProcesses failed", "pci_id", pci, "status", -int(n))
			continue
		}
		procs := make([]ml.DeviceProcess, 0, int(n))
		for i := 0; i < int(n); i++ {
			procs = append(procs, ml.DeviceProcess{PID: int(pids[i]), UsedMemory: uint64(used[i])})
		}
		if len(procs) > 0 {
			out[pci] = procs
		}
	}
	return out
}

func nvmlPhysicalMemoryByPCI(pciIDs []string) map[string]uint64 {
	if len(pciIDs) == 0 {
		return nil
	}

	nvml, err := dlopenFirst([]string{"libnvidia-ml.so.1", "libnvidia-ml.so"}, false)
	if err != nil {
		slog.Debug("NVML unavailable, reporting CUDA's device memory total", "error", err)
		return nil
	}

	initFn, initErr := dlsym(nvml, "nvmlInit_v2")
	shutdownFn, shutdownErr := dlsym(nvml, "nvmlShutdown")
	handleFn, handleErr := dlsym(nvml, "nvmlDeviceGetHandleByPciBusId_v2")
	memoryFn, memoryErr := dlsym(nvml, "nvmlDeviceGetMemoryInfo")
	if err := cmp.Or(initErr, shutdownErr, handleErr, memoryErr); err != nil {
		slog.Debug("NVML missing a required symbol", "error", err)
		return nil
	}

	if ret := C.ollama_call_nvml_init(initFn); ret != 0 {
		slog.Debug("nvmlInit_v2 failed", "status", int(ret))
		return nil
	}
	defer C.ollama_call_nvml_shutdown(shutdownFn)

	out := make(map[string]uint64, len(pciIDs))
	for _, pci := range pciIDs {
		if pci == "" {
			continue
		}

		cpci := C.CString(pci)
		var handle unsafe.Pointer
		ret := C.ollama_call_nvml_device_get_handle_by_pci_bus_id(handleFn, cpci, &handle)
		C.free(unsafe.Pointer(cpci))
		if ret != 0 {
			slog.Debug("NVML did not recognize device", "pci_id", pci, "status", int(ret))
			continue
		}

		var total, free C.ulonglong
		if ret := C.ollama_call_nvml_device_get_memory_info(memoryFn, handle, &total, &free); ret != 0 {
			slog.Debug("nvmlDeviceGetMemoryInfo failed", "pci_id", pci, "status", int(ret))
			continue
		}
		out[pci] = uint64(total)
	}
	return out
}

// FreeMemoryByPCI reports how much memory each device has free, read straight from NVML.
//
// It exists because the ordinary way to answer that question is device discovery, which
// costs a subprocess whenever any device lacks a runner to ask -- and during a load that is
// every device, since the runner being loaded is not serving yet. That is exactly the
// moment a client most wants the number: a load is the only time free memory moves by tens
// of gigabytes, and it moves in two steps a second or so apart. This call is a handful of
// microseconds, so it can be read as fast as anyone cares to sample it.
//
// It reports only what the driver says is free, so it is not a substitute for discovery: it
// cannot enumerate devices, and it says nothing about which process holds what. Callers use
// it to refresh a figure on devices they already know about.
func FreeMemoryByPCI(pciIDs []string) map[string]uint64 {
	if len(pciIDs) == 0 {
		return nil
	}

	session := liveMemorySession()
	if session == nil {
		return nil
	}

	out := make(map[string]uint64, len(pciIDs))
	for _, pci := range pciIDs {
		if pci == "" {
			continue
		}

		handle, ok := session.handle(pci)
		if !ok {
			continue
		}

		var total, free C.ulonglong
		if ret := C.ollama_call_nvml_device_get_memory_info(session.memoryFn, handle, &total, &free); ret != 0 {
			continue
		}
		out[pci] = uint64(free)
	}
	return out
}

// nvmlLiveSession keeps NVML loaded and initialised, with a device handle per PCI address.
//
// Opening the library, resolving symbols and calling nvmlInit on every read costs about
// 28ms, which is nothing against the 574ms subprocess this exists to avoid but is 25x the
// 1.1ms that serving a cached figure costs -- and this is now on the read path for every
// reported figure, so that would be a regression on the common case to fix the rare one.
// Held open, a read is a handful of microseconds.
//
// nvmlShutdown is deliberately never called. NVML reference-counts init, the session lives
// as long as the process, and there is no teardown point that is not simply exit.
type nvmlLiveSession struct {
	memoryFn unsafe.Pointer
	handleFn unsafe.Pointer

	mu      sync.Mutex
	handles map[string]unsafe.Pointer
}

var (
	liveMemoryOnce         sync.Once
	liveMemorySessionValue *nvmlLiveSession
)

func liveMemorySession() *nvmlLiveSession {
	liveMemoryOnce.Do(func() {
		nvml, err := dlopenFirst([]string{"libnvidia-ml.so.1", "libnvidia-ml.so"}, false)
		if err != nil {
			slog.Debug("NVML unavailable, live free memory will fall back to discovery", "error", err)
			return
		}

		initFn, initErr := dlsym(nvml, "nvmlInit_v2")
		handleFn, handleErr := dlsym(nvml, "nvmlDeviceGetHandleByPciBusId_v2")
		memoryFn, memoryErr := dlsym(nvml, "nvmlDeviceGetMemoryInfo")
		if err := cmp.Or(initErr, handleErr, memoryErr); err != nil {
			slog.Debug("NVML missing a required symbol", "error", err)
			return
		}
		if ret := C.ollama_call_nvml_init(initFn); ret != 0 {
			slog.Debug("nvmlInit_v2 failed", "status", int(ret))
			return
		}

		liveMemorySessionValue = &nvmlLiveSession{
			memoryFn: memoryFn,
			handleFn: handleFn,
			handles:  make(map[string]unsafe.Pointer),
		}
	})
	return liveMemorySessionValue
}

// handle resolves and remembers a device handle. Handles are stable for the life of the
// NVML session, so the lookup is done once per device rather than once per read.
func (s *nvmlLiveSession) handle(pci string) (unsafe.Pointer, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if h, ok := s.handles[pci]; ok {
		return h, h != nil
	}

	cpci := C.CString(pci)
	var h unsafe.Pointer
	ret := C.ollama_call_nvml_device_get_handle_by_pci_bus_id(s.handleFn, cpci, &h)
	C.free(unsafe.Pointer(cpci))
	if ret != 0 {
		// Remembered as absent so a device NVML does not know is not re-resolved on
		// every read.
		s.handles[pci] = nil
		return nil, false
	}
	s.handles[pci] = h
	return h, true
}

func cudaDeviceAttribute(fn unsafe.Pointer, attr int, device C.int) int {
	var value C.int
	if ret := C.ollama_call_cu_device_get_attribute(fn, &value, C.int(attr), device); ret != cuSuccess {
		return 0
	}
	return int(value)
}

func dlopenFirst(names []string, global bool) (dlHandle, error) {
	var errs []string
	for _, name := range names {
		handle, err := dlopen(name, global)
		if err == nil {
			return handle, nil
		}
		errs = append(errs, err.Error())
	}
	return dlHandle{}, errors.New(strings.Join(errs, "; "))
}

func dlopen(path string, global bool) (dlHandle, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	handle := C.ollama_dlopen(cpath, boolToCInt(global))
	if handle == nil {
		return dlHandle{}, fmt.Errorf("dlopen %s: %s", path, C.GoString(C.ollama_dlerror()))
	}
	return dlHandle{ptr: handle}, nil
}

func dlsym(handle dlHandle, name string) (unsafe.Pointer, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	sym := C.ollama_dlsym(handle.ptr, cname)
	if sym == nil {
		return nil, fmt.Errorf("dlsym %s: %s", name, C.GoString(C.ollama_dlerror()))
	}
	return sym, nil
}

func dlsymAny(handle dlHandle, names ...string) (unsafe.Pointer, error) {
	var errs []string
	for _, name := range names {
		sym, err := dlsym(handle, name)
		if err == nil {
			return sym, nil
		}
		errs = append(errs, err.Error())
	}
	return nil, errors.New(strings.Join(errs, "; "))
}

func callGGMLBackendLoad(fn unsafe.Pointer, path string) unsafe.Pointer {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	return C.ollama_call_ggml_backend_load(fn, cpath)
}

func callGGMLRegDevCount(fn unsafe.Pointer, reg unsafe.Pointer) uintptr {
	return uintptr(C.ollama_call_ggml_backend_reg_dev_count(fn, reg))
}

func callGGMLRegDevGet(fn unsafe.Pointer, reg unsafe.Pointer, index int) unsafe.Pointer {
	return C.ollama_call_ggml_backend_reg_dev_get(fn, reg, C.size_t(index))
}

func callGGMLRegName(fn unsafe.Pointer, reg unsafe.Pointer) string {
	return C.GoString(C.ollama_call_ggml_backend_reg_name(fn, reg))
}

func callGGMLDeviceProps(fn unsafe.Pointer, dev unsafe.Pointer) ggmlBackendDevProps {
	var props ggmlBackendDevProps
	C.ollama_call_ggml_backend_dev_get_props(fn, dev, unsafe.Pointer(&props))
	return props
}

func cString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	return C.GoString(C.ollama_cstr_from_uintptr(C.uintptr_t(ptr)))
}

func boolToCInt(v bool) C.int {
	if v {
		return 1
	}
	return 0
}

// UnavailableDevices reports GPUs the machine has that the compute backends did not offer,
// from every vendor this package knows how to ask.
func UnavailableDevices(known []string) []ml.UnavailableDevice {
	return append(nvidiaUnavailableDevices(known), amdUnavailableDevices(usableSet(known))...)
}

// nvidiaUnavailableDevices is the NVIDIA half of UnavailableDevices.
//
// known is the set of PCI addresses discovery DID return, in ml.DeviceInfo.PCIID form.
// Anything NVML enumerates and that set does not contain is a device that physically
// exists and cannot be used -- see the note on ml.UnavailableDevice for why that comparison
// is the whole detector, and why the absence of such a device is otherwise unreportable.
//
// When NVML cannot be asked, it falls back to the PCI bus rather than returning nothing,
// because the case NVML is least able to describe -- a card the driver has lost -- is the one
// most worth reporting. It never invents a reason it did not read. An empty result still
// means "nothing to report OR could not look", which is why the caller does not present it
// as "all devices healthy".
func nvidiaUnavailableDevices(known []string) []ml.UnavailableDevice {
	nvml, err := dlopenFirst([]string{"libnvidia-ml.so.1", "libnvidia-ml.so"}, false)
	if err != nil {
		slog.Debug("NVML unavailable, falling back to the PCI bus", "error", err)
		return gpusOnlySysfsCanSee(usableSet(known), nil)
	}

	initFn, initErr := dlsym(nvml, "nvmlInit_v2")
	shutdownFn, shutdownErr := dlsym(nvml, "nvmlShutdown")
	countFn, countErr := dlsym(nvml, "nvmlDeviceGetCount_v2")
	handleFn, handleErr := dlsym(nvml, "nvmlDeviceGetHandleByIndex_v2")
	nameFn, nameErr := dlsym(nvml, "nvmlDeviceGetName")
	uuidFn, uuidErr := dlsym(nvml, "nvmlDeviceGetUUID")
	pciFn, pciErr := dlsym(nvml, "nvmlDeviceGetPciInfo_v3")
	tempFn, tempErr := dlsym(nvml, "nvmlDeviceGetTemperature")
	if err := cmp.Or(initErr, shutdownErr, countErr, handleErr, nameErr, uuidErr, pciErr, tempErr); err != nil {
		slog.Debug("NVML missing a symbol needed to check device health", "error", err)
		return gpusOnlySysfsCanSee(usableSet(known), nil)
	}
	// nvmlErrorString is looked up separately: without it the reason is still known, only
	// the driver's own wording for it is missing, which is not worth losing the report for.
	errStrFn, _ := dlsym(nvml, "nvmlErrorString")

	if ret := C.ollama_call_nvml_init(initFn); ret != 0 {
		slog.Debug("nvmlInit_v2 failed", "status", int(ret))
		return gpusOnlySysfsCanSee(usableSet(known), nil)
	}
	defer C.ollama_call_nvml_shutdown(shutdownFn)

	usable := usableSet(known)

	var count C.uint
	if ret := C.ollama_call_nvml_device_get_count(countFn, &count); ret != 0 {
		slog.Debug("nvmlDeviceGetCount_v2 failed", "status", int(ret))
		return gpusOnlySysfsCanSee(usable, nil)
	}

	var out []ml.UnavailableDevice
	seen := map[string]bool{}
	for i := range uint(count) {
		var handle unsafe.Pointer
		if ret := C.ollama_call_nvml_device_get_handle_by_index(handleFn, C.uint(i), &handle); ret != 0 {
			slog.Debug("NVML could not open a device it had counted", "index", i, "status", int(ret))
			continue
		}

		pci := nvmlString(64, func(buf *C.char, n C.uint) C.int {
			return C.ollama_call_nvml_device_get_pci_bus_id_legacy(pciFn, handle, buf, n)
		})
		if pci == "" || usable[strings.ToLower(pci)] {
			continue
		}

		// This device exists and no backend offered it. Ask it something that reads live
		// state, because that is what a faulted device cannot answer.
		var temp C.uint
		status := int(C.ollama_call_nvml_device_get_temperature(tempFn, handle, &temp))

		detail := ""
		if errStrFn != nil && status != nvmlSuccess {
			if s := C.ollama_call_nvml_error_string(errStrFn, C.int(status)); s != nil {
				detail = C.GoString(s)
			}
		}

		probe := deviceProbe{
			PCIID: pci,
			Name: nvmlString(96, func(buf *C.char, n C.uint) C.int {
				return C.ollama_call_nvml_device_get_name(nameFn, handle, buf, n)
			}),
			UUID: nvmlString(96, func(buf *C.char, n C.uint) C.int {
				return C.ollama_call_nvml_device_get_uuid(uuidFn, handle, buf, n)
			}),
			Status:           status,
			DetailFromDriver: detail,
			Bus:              busStateFor(pci),
		}

		device := classifyUnavailable(probe)
		slog.Warn("a GPU is present but cannot be used",
			"pci_id", device.PCIID, "name", device.Name, "reason", device.Reason, "detail", device.Detail)
		seen[strings.ToLower(pci)] = true
		out = append(out, device)
	}

	return append(out, gpusOnlySysfsCanSee(usable, seen)...)
}

// gpusOnlySysfsCanSee reports NVIDIA GPUs the kernel enumerates that neither the compute
// backend nor NVML offered.
//
// It exists because NVML is not the superset it looks like. A faulted card was visible to
// NVML at first and vanished from it later, after the driver was torn down and reloaded
// without the device -- at which point NVML and CUDA agreed on one device and the detector
// fell silent with a broken card still in the machine. sysfs never lost it.
//
// Nothing can be said about WHY these are unusable: the driver is not talking about them,
// so there is no status code and no name. That is reported as such rather than guessed at,
// because "a GPU is here and nothing will tell me why it is unusable" is already the
// actionable fact, and inventing a reason would make it less trustworthy, not more.
func gpusOnlySysfsCanSee(usable, seen map[string]bool) []ml.UnavailableDevice {
	var out []ml.UnavailableDevice
	for _, pci := range nvidiaGPUsInSysfs() {
		key := strings.ToLower(pci)
		if usable[key] || seen[key] {
			continue
		}

		device := ml.UnavailableDevice{
			PCIID:  pci,
			Reason: "not_reported_by_driver",
			Detail: "the kernel enumerates this GPU but the driver does not report it",
			Recovery: "the driver has lost the device without releasing it. A cold power cycle " +
				"restores it; a warm reboot may not, since the card keeps power across one",
			Bus: busStateFor(pci),
		}
		slog.Warn("a GPU is on the PCI bus but the driver does not report it",
			"pci_id", device.PCIID, "reason", device.Reason)
		out = append(out, device)
	}
	return out
}

// nvmlString runs one of NVML's fill-a-buffer calls and returns what it wrote, or "".
func nvmlString(size int, fill func(*C.char, C.uint) C.int) string {
	buf := (*C.char)(C.malloc(C.size_t(size)))
	defer C.free(unsafe.Pointer(buf))
	if ret := fill(buf, C.uint(size)); ret != 0 {
		return ""
	}
	return C.GoString(buf)
}

// usableSet indexes the PCI addresses discovery did offer, lowercased so a comparison
// cannot fail on the hex casing of an address that came from a different source.
func usableSet(known []string) map[string]bool {
	usable := make(map[string]bool, len(known))
	for _, pci := range known {
		usable[strings.ToLower(pci)] = true
	}
	return usable
}

type nvmlMemoryInterface struct {
	busWidthBits int
	clockMaxMHz  int
}

// nvmlMemoryInterfaceByPCI reads each device's memory bus width and maximum memory clock,
// from which ml.DeviceInfo.MemoryBandwidth derives peak bandwidth. These are static
// properties of the card: read once at discovery, never a live reading.
//
// A device that answers only one of the two is left out entirely -- a bandwidth derived from
// half its inputs would be a number with the authority of a measurement and none of the basis.
func nvmlMemoryInterfaceByPCI(pciIDs []string) map[string]nvmlMemoryInterface {
	if len(pciIDs) == 0 {
		return nil
	}
	nvml, err := dlopenFirst([]string{"libnvidia-ml.so.1", "libnvidia-ml.so"}, false)
	if err != nil {
		return nil
	}
	initFn, initErr := dlsym(nvml, "nvmlInit_v2")
	shutdownFn, shutdownErr := dlsym(nvml, "nvmlShutdown")
	handleFn, handleErr := dlsym(nvml, "nvmlDeviceGetHandleByPciBusId_v2")
	busFn, busErr := dlsym(nvml, "nvmlDeviceGetMemoryBusWidth")
	clockFn, clockErr := dlsym(nvml, "nvmlDeviceGetMaxClockInfo")
	if err := cmp.Or(initErr, shutdownErr, handleErr, busErr, clockErr); err != nil {
		slog.Debug("NVML cannot report the memory interface", "error", err)
		return nil
	}
	if ret := C.ollama_call_nvml_init(initFn); ret != 0 {
		return nil
	}
	defer C.ollama_call_nvml_shutdown(shutdownFn)

	out := make(map[string]nvmlMemoryInterface, len(pciIDs))
	for _, pci := range pciIDs {
		if pci == "" {
			continue
		}
		cpci := C.CString(pci)
		var handle unsafe.Pointer
		ret := C.ollama_call_nvml_device_get_handle_by_pci_bus_id(handleFn, cpci, &handle)
		C.free(unsafe.Pointer(cpci))
		if ret != 0 {
			continue
		}
		var bus, clock C.uint
		if C.ollama_call_nvml_device_get_uint(busFn, handle, &bus) != 0 || bus == 0 {
			continue
		}
		if C.ollama_call_nvml_device_get_max_mem_clock(clockFn, handle, &clock) != 0 || clock == 0 {
			continue
		}
		out[pci] = nvmlMemoryInterface{busWidthBits: int(bus), clockMaxMHz: int(clock)}
	}
	return out
}
