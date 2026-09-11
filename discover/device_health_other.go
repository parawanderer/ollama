//go:build !linux

package discover

import "github.com/ollama/ollama/ml"

// UnavailableDevices reports nothing off Linux.
//
// The detector needs two things this code only has on Linux: NVML enumerating a device the
// compute backend hides, and sysfs to say whether the device is still on the PCIe bus.
// Returning nil is the same answer the Linux path gives when it cannot ask, and callers
// already treat empty as "nothing to report or could not look" rather than "all healthy".
func UnavailableDevices(known []string) []ml.UnavailableDevice { return nil }
