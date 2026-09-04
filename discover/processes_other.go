//go:build !linux

package discover

import "github.com/ollama/ollama/ml"

// ComputeProcesses reports which processes hold memory on each device. Only implemented
// where a management library can answer it; elsewhere the answer is "not known", which a
// caller must not read as "nothing is running".
func ComputeProcesses(pciIDs []string) map[string][]ml.DeviceProcess {
	return nil
}
