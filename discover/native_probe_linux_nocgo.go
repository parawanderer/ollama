//go:build linux && !cgo

package discover

import "github.com/ollama/ollama/ml"

import (
	"context"
	"errors"
)

func runPlatformNativeProbe(context.Context, []string) ([]nativeProbeDevice, error) {
	return nil, errors.New("native GPU discovery requires cgo on Linux")
}

// FreeMemoryByPCI is unavailable here; callers fall back to cached discovery.
func FreeMemoryByPCI([]string) map[string]uint64 { return nil }

func UtilizationByPCI([]string) map[string]ml.DeviceUtilization { return nil }
