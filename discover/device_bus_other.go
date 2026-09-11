//go:build !linux

package discover

import "github.com/ollama/ollama/ml"

func busStateFor(pciID string) *ml.DeviceBusState { return nil }

func PCIeMaxLink(pciID string) (generation, width int) { return 0, 0 }
