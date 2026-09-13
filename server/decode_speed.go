package server

import (
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

// Expected decode speed: what the box profile (box_profile.go) predicts for a model where it is
// placed now. It is the one piece of arithmetic behind both /api/ps's expected_decode and the
// prediction recorded beside every generation's measured speed in usage.db, so the two cannot
// disagree about what was predicted.
//
// One device's time per token is its per-token overhead, plus its per-layer overhead for the
// layers it holds, plus the bytes it reads at the bandwidth decode achieves. A layer split visits
// its devices in turn, so their times add; the per-token overhead is paid once.

// decodeInputs is what expected decode time depends on.
type decodeInputs struct {
	gpus     []ml.DeviceID
	memByGPU map[ml.DeviceID]api.MemoryBreakdown
	fits     map[ml.DeviceID]api.ProfileDevice

	layers int
	// activeWeights is the share of the weights one token reads: 1 for a dense model, less for
	// a mixture of experts, which reads only the experts each token is routed to.
	activeWeights float64
	// slidingWindow means the KV cache read per token does not grow at one rate per token of
	// context, so no context term is given.
	slidingWindow   bool
	grantedCtxTotal int
	// partlyOnCPU means some layers were spilled to the host, which nothing here measures.
	partlyOnCPU bool
}

// expectedDecode returns the milliseconds per token at an empty cache and the milliseconds each
// token of context adds, or a reason neither can be given. perCtxOK is false when the context
// term is not modelled (a sliding window).
func expectedDecode(in decodeInputs) (baseMs, perCtxMs float64, perCtxOK bool, unavailable string) {
	if len(in.gpus) == 0 {
		return 0, 0, false, "not_on_gpu"
	}
	if in.partlyOnCPU {
		return 0, 0, false, "partly_on_cpu"
	}
	var weights float64
	for _, d := range in.gpus {
		if _, ok := in.fits[d]; !ok {
			return 0, 0, false, "profile_pending"
		}
		m := in.memByGPU[d]
		weights += float64(max(m.Weights, 0))
	}
	if weights == 0 {
		return 0, 0, false, "memory_unknown"
	}
	active := in.activeWeights
	if active <= 0 || active > 1 {
		active = 1
	}

	perCtxOK = !in.slidingWindow && in.grantedCtxTotal > 0
	var tokenOverhead float64
	for _, d := range in.gpus {
		f, m := in.fits[d], in.memByGPU[d]
		bytesPerMs := float64(f.BandwidthBytesPerSec) / 1000
		w := float64(max(m.Weights, 0))
		// Recurrent state is read and rewritten every token whatever the routing, like a
		// dense weight.
		read := w*active + float64(max(m.RecurrentState, 0))
		tokenOverhead = max(tokenOverhead, f.TokenOverheadMs) // paid once, not per device
		// A device holds layers in proportion to its weights, near enough.
		baseMs += f.LayerOverheadUs / 1000 * float64(in.layers) * (w / weights)
		baseMs += read / bytesPerMs
		if perCtxOK {
			perCtxMs += float64(max(m.KVCache, 0)) / float64(in.grantedCtxTotal) / bytesPerMs
		}
	}
	return tokenOverhead + baseMs, perCtxMs, perCtxOK, ""
}

// activeWeightsFraction is the share of a model's weights one decoded token reads. For a mixture
// of experts, the expert tensors count at expert_used_count / expert_count; everything else is
// read in full. The token embedding is left out of both sides: decode reads one row of it.
func activeWeightsFraction(f *ggml.GGML) float64 {
	if f == nil {
		return 1
	}
	experts, used := f.KV().Uint("expert_count"), f.KV().Uint("expert_used_count")
	if experts == 0 || used == 0 || used >= experts {
		return 1
	}
	var total, exps float64
	for _, t := range f.Tensors().Items() {
		if strings.HasPrefix(t.Name, "token_embd") {
			continue
		}
		size := float64(t.Size())
		total += size
		if strings.Contains(t.Name, "_exps") {
			exps += size
		}
	}
	if total == 0 {
		return 1
	}
	return (total - exps + exps*float64(used)/float64(experts)) / total
}

// expectedDecodeReport is the /api/ps form: tokens per second at an empty cache, and how much
// longer each token takes per thousand tokens already in the cache.
func expectedDecodeReport(in decodeInputs) *api.ExpectedDecode {
	baseMs, perCtxMs, perCtxOK, unavailable := expectedDecode(in)
	switch unavailable {
	case "not_on_gpu":
		return nil
	case "":
	default:
		return &api.ExpectedDecode{Unavailable: unavailable}
	}
	out := &api.ExpectedDecode{TokensPerSec: round3(1000 / baseMs)}
	if perCtxOK {
		out.MsPerTokenPer1kContext = round3(perCtxMs * 1000)
	}
	if in.activeWeights > 0 && in.activeWeights < 1 {
		out.ActiveWeightsFraction = round3(in.activeWeights)
	}
	return out
}

// predictedEvalMs is the prediction recorded beside a generation's measured decode: milliseconds
// per token at the context the decode ran through on average (the prompt plus half the tokens
// decoded). basis says what the number includes, and is computed from the same inputs at the
// same point, so a row can never claim a context term it does not contain:
//   - "profile": weights, overheads and the KV cache read at that context
//   - "profile_no_kv": the cache read is not modelled -- a sliding-window model, or a context
//     the engine did not report -- so the prediction is a lower bound on the time
func predictedEvalMs(in decodeInputs, promptTokens, decoded int) (ms float64, basis string, ok bool) {
	baseMs, perCtxMs, perCtxOK, unavailable := expectedDecode(in)
	if unavailable != "" || decoded <= 0 {
		return 0, "", false
	}
	if !perCtxOK {
		return baseMs, "profile_no_kv", true
	}
	occupied := float64(promptTokens) + float64(decoded)/2
	return baseMs + perCtxMs*occupied, "profile", true
}

// decodeInputsFor is the /api/ps row's inputs: the runner as last read, and the profile now.
func decodeInputsFor(v loadedModel, fits map[ml.DeviceID]api.ProfileDevice) decodeInputs {
	return decodeInputs{
		gpus: v.gpus, memByGPU: v.memByGPU, fits: fits,
		layers: v.layers, activeWeights: v.activeWeights,
		slidingWindow: v.slidingWindow, grantedCtxTotal: v.grantedCtxTotal,
		partlyOnCPU: v.size > v.sizeVRAM,
	}
}

// profileFits returns the measured profile's device fits for the devices ollama knows now, keyed
// by device id. Empty until the profile is measured. It reads the cached device list only.
func (s *Scheduler) profileFits() map[ml.DeviceID]api.ProfileDevice {
	if s.profiler == nil {
		return nil
	}
	s.deviceCacheMu.Lock()
	devices := s.deviceCache
	s.deviceCacheMu.Unlock()
	r := s.profiler.report(devices)
	if r == nil || r.State != "measured" {
		return nil
	}
	byPCI := make(map[string]api.ProfileDevice, len(r.Devices))
	for _, d := range r.Devices {
		byPCI[d.PCIID] = d
	}
	out := make(map[ml.DeviceID]api.ProfileDevice, len(devices))
	for _, d := range devices {
		if f, ok := byPCI[d.PCIID]; ok {
			out[d.DeviceID] = f
		}
	}
	return out
}
