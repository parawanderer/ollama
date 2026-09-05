//go:build linux && !cgo

package discover

import (
	"context"
	"errors"
)

func runPlatformNativeProbe(context.Context, []string) ([]nativeProbeDevice, error) {
	return nil, errors.New("native GPU discovery requires cgo on Linux")
}

// FreeMemoryByPCI is unavailable here; callers fall back to cached discovery.
func FreeMemoryByPCI([]string) map[string]uint64 { return nil }
