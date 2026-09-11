//go:build !linux

package discover

import "github.com/ollama/ollama/ml"

func Topology(devices []ml.DeviceInfo) *ml.Topology       { return nil }
func CachedTopology(devices []ml.DeviceInfo) *ml.Topology { return nil }
