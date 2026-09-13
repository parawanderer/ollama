package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
)

// decodeTimerWarmups and decodeTimerTokens are the two warm-up requests and the forced tokens
// of each timed one. The first request on a fresh server pays for clocks ramping up and graph
// capture, so it is never timed.
const (
	decodeTimerWarmups = 2
	decodeTimerReps    = 3
	decodeTimerTokens  = 128
	decodeTimerReady   = 2 * time.Minute
)

// decodeTimerPrompt is bos and 31 arbitrary tokens, passed as ids so no tokenizer is involved.
var decodeTimerPrompt = func() []int {
	p := []int{1}
	for t := 3; t < 34; t++ {
		p = append(p, t)
	}
	return p
}()

// TimeDecode starts llama-server on gpus with the model at modelPath, forces decode, and
// returns the engine's own time per decoded token: the median of three timed requests of 128
// tokens after two warm-ups.
//
// tensorSplit selects --split-mode tensor across gpus; otherwise gpus must be one device and
// the model runs on it alone. Everything that could move the figure between runs is fixed: no
// fit pass (tensor split does not support one), every layer on the device, flash attention
// (tensor split requires it), one slot, a 4096 context, no prompt cache.
//
// The time comes from llama-server's timings, not from the wall clock, so the HTTP round and
// the prompt are not in it.
func TimeDecode(ctx context.Context, gpus []ml.DeviceInfo, modelPath string, tensorSplit bool) (float64, error) {
	if len(gpus) == 0 || (!tensorSplit && len(gpus) != 1) {
		return 0, fmt.Errorf("time decode: %d devices for tensor split %v", len(gpus), tensorSplit)
	}
	exe, err := FindLlamaServer()
	if err != nil {
		return 0, err
	}
	port, err := freePort()
	if err != nil {
		return 0, err
	}

	mode := "none"
	if tensorSplit {
		mode = "tensor"
	}
	params := []string{
		"--model", modelPath,
		"--port", strconv.Itoa(port),
		"--host", "127.0.0.1",
		"--no-webui",
		"--offline",
		"-c", "4096",
		"-np", "1",
		"-ngl", "999",
		"--fit", "off",
		"--flash-attn", "on",
		"--cache-ram", "0",
		"--split-mode", mode,
	}
	if !tensorSplit {
		params = append(params, "--main-gpu", "0")
	}

	cmd := exec.Command(exe, params...)
	unlock := dieWithParent(cmd)
	defer unlock() // registered first, so it runs after the child is killed and reaped
	SetupLlamaServerCommandEnv(cmd, exe, ml.LibraryPaths(gpus), ml.GetDevicesEnv(gpus))
	tail := &lastLines{max: 20}
	cmd.Stdout, cmd.Stderr = tail, tail
	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("starting llama-server: %w", err)
	}
	exited := make(chan struct{})
	go func() { _ = cmd.Wait(); close(exited) }()
	defer func() {
		_ = cmd.Process.Kill()
		<-exited
	}()

	base := "http://127.0.0.1:" + strconv.Itoa(port)
	if err := waitHealthy(ctx, base, exited); err != nil {
		return 0, fmt.Errorf("%w; llama-server said: %s", err, tail.String())
	}

	for range decodeTimerWarmups {
		if _, err := timedCompletion(ctx, base, 32); err != nil {
			return 0, fmt.Errorf("%w; llama-server said: %s", err, tail.String())
		}
	}
	times := make([]float64, 0, decodeTimerReps)
	for range decodeTimerReps {
		ms, err := timedCompletion(ctx, base, decodeTimerTokens)
		if err != nil {
			return 0, fmt.Errorf("%w; llama-server said: %s", err, tail.String())
		}
		times = append(times, ms)
	}
	slices.Sort(times)
	return times[len(times)/2], nil
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// waitHealthy polls /health, which answers 503 until the model is loaded. It gives up when
// the process exits, the context ends or decodeTimerReady passes.
func waitHealthy(ctx context.Context, base string, exited <-chan struct{}) error {
	deadline := time.Now().Add(decodeTimerReady)
	for {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, base+"/health", nil)
		if resp, err := http.DefaultClient.Do(req); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-exited:
			return errors.New("llama-server exited before it was ready")
		case <-time.After(100 * time.Millisecond):
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("llama-server not ready after %s", decodeTimerReady)
		}
	}
}

// timedCompletion forces n tokens and returns the engine's milliseconds per decoded token.
func timedCompletion(ctx context.Context, base string, n int) (float64, error) {
	body, _ := json.Marshal(map[string]any{
		"prompt":       decodeTimerPrompt,
		"n_predict":    n,
		"ignore_eos":   true,
		"cache_prompt": false,
		"temperature":  0,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/completion", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return 0, fmt.Errorf("completion: %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}
	var out struct {
		Timings struct {
			PredictedN  int     `json:"predicted_n"`
			PredictedMs float64 `json:"predicted_ms"`
		} `json:"timings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, err
	}
	if out.Timings.PredictedN <= 0 {
		return 0, errors.New("completion reported no decoded tokens")
	}
	return out.Timings.PredictedMs / float64(out.Timings.PredictedN), nil
}

// lastLines keeps the last few lines written to it, so a failure can say what the process
// said without holding its whole log.
type lastLines struct {
	mu    sync.Mutex
	max   int
	lines []string
	part  []byte
}

func (l *lastLines) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.part = append(l.part, p...)
	for {
		i := bytes.IndexByte(l.part, '\n')
		if i < 0 {
			break
		}
		l.lines = append(l.lines, string(l.part[:i]))
		l.part = l.part[i+1:]
	}
	l.part = slices.Clone(l.part[max(0, len(l.part)-4096):])
	if len(l.lines) > l.max {
		l.lines = slices.Clone(l.lines[len(l.lines)-l.max:])
	}
	return len(p), nil
}

func (l *lastLines) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return strings.Join(append(slices.Clone(l.lines), string(l.part)), " | ")
}
