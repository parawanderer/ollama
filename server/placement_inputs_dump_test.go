package server

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

// TestDumpPlacementInputs prints, for each named model, what the scheduler places it from
// before anything has been measured: the metadata line (weights + bytes per token of
// context), whether the model is measured by probe instead, and the probe's context pair.
//
// It is not a test of anything. It exists so a notebook can draw ollama's own prediction
// rather than a re-implementation of it -- slop-zone/notebooks/context-vs-vram.ipynb runs
// it with the model store mounted and OLLAMA_MODELS pointing at it. Skipped unless
// SLOP_DUMP_MODELS names the models, comma-separated.
func TestDumpPlacementInputs(t *testing.T) {
	names := os.Getenv("SLOP_DUMP_MODELS")
	if names == "" {
		t.Skip("set SLOP_DUMP_MODELS to a comma-separated list of model names")
	}
	for _, name := range strings.Split(names, ",") {
		out := map[string]any{"name": name}
		m, err := GetModel(name)
		if err != nil {
			out["error"] = err.Error()
			emitPlacementInputs(out)
			continue
		}
		// 1024, as the scheduler loads it (sched.go). 0 skips arrays, and then every per-layer
		// value -- head counts, sliding-window patterns -- reads as absent.
		f, err := llm.LoadModel(m.ModelPath, 1024)
		if err != nil {
			out["error"] = err.Error()
			emitPlacementInputs(out)
			continue
		}
		weights, bytesPerToken := llm.PredictServerVRAMParts(m.ModelPath, m.ProjectorPaths, f)
		trainCtx := modelTrainContext(f)
		points, probeable := probePoints(trainCtx, 1)
		out["model_path"] = m.ModelPath
		out["projector_paths"] = m.ProjectorPaths
		out["architecture"] = f.KV().Architecture()
		out["train_ctx"] = trainCtx
		out["metadata_complete"] = f.KV().KVCacheModelIsComplete()
		out["measured_by_probe"] = modelNeedsMeasurement(f, m.ProjectorPaths)
		out["weights"] = weights
		out["bytes_per_token"] = bytesPerToken
		out["probe_points"] = points
		out["probeable"] = probeable
		emitPlacementInputs(out)
	}
}

func emitPlacementInputs(v map[string]any) {
	b, _ := json.Marshal(v)
	fmt.Printf("PLACEMENT_INPUTS %s\n", b)
}
