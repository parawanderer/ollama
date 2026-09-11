//go:build !linux && !windows

package discover

import "github.com/ollama/ollama/ml"

import (
	"context"
	"errors"
)

func runPlatformNativeProbe(context.Context, []string) ([]nativeProbeDevice, error) {
	return nil, errors.New("native GPU discovery is not implemented on this platform")
}

// FreeMemoryByPCI is unavailable here; callers fall back to cached discovery.
func FreeMemoryByPCI([]string) map[string]uint64 { return nil }

func UtilizationByPCI([]string) map[string]ml.DeviceUtilization { return nil }
