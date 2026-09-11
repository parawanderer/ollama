package llm

import (
	"bytes"
	"regexp"
	"strconv"
	"sync"

	"github.com/ollama/ollama/api"
)

// llama-server's host-RAM prompt cache (--cache-ram) reports itself only in its log. These
// are trace lines from tools/server/server-context.cpp and server-task.cpp; the wording is
// matched from real output, kept in testdata/prompt_cache_*.log.
var (
	promptCacheUpdateBeginRegex = regexp.MustCompile(`updating prompt cache`)
	promptCacheSavedRegex       = regexp.MustCompile(`saving prompt with length (\d+), total state size = ([0-9.]+) MiB`)
	promptCacheRestoredRegex    = regexp.MustCompile(`found better prompt with f_keep`)
	// Three wordings drop an entry: making room for a save, and the two limits enforced
	// afterwards. All three end in the size of what went.
	promptCacheEvictedRegex    = regexp.MustCompile(`removing oldest entry \(size = ([0-9.]+) MiB\)`)
	promptCacheTooLargeRegex   = regexp.MustCompile(`prompt state size [0-9.]+ MiB exceeds cache size limit`)
	promptCacheStateRegex      = regexp.MustCompile(`cache state: (\d+) prompts, ([0-9.]+) MiB \(limits: ([0-9.]+) MiB`)
	promptCacheEntryRegex      = regexp.MustCompile(`- prompt 0x[0-9a-f]+:\s+(\d+) tokens`)
	promptCacheUpdateTookRegex = regexp.MustCompile(`prompt cache update took ([0-9.]+) ms`)
)

// promptCacheTracker turns those log lines into the swap each request paid for and the
// cache's current occupancy.
//
// A swap is attributed to a request by a window: begin when the request is handed to the
// engine, take when it completes. That is only sound when the runner serves one request at
// a time, which is how ollama runs it (-np 1, one semaphore slot), and the caller must not
// use it otherwise. Lines that arrive after take -- a swap logged late -- are discarded by
// the next begin, so a late line is lost rather than pinned on the wrong request.
type promptCacheTracker struct {
	mu      sync.Mutex
	pending *api.PromptCacheSwap // lines seen since the engine started an update
	done    *api.PromptCacheSwap // the last completed update inside the current window
	state   *api.PromptCacheState
}

func mibToBytes(s []byte) uint64 {
	mib, err := strconv.ParseFloat(string(s), 64)
	if err != nil || mib < 0 {
		return 0
	}
	return uint64(mib * 1024 * 1024)
}

// observe reads a chunk of the runner's log. Lines are handled in order, which matters: a
// swap is the sequence begin, save, evict, restore, state, took.
func (t *promptCacheTracker) observe(chunk []byte) {
	// Cheap rejection of the other 99% of the log. Both words are needed: two of the eviction
	// wordings ("cache size limit reached", "cache token limit") never say "prompt".
	if !bytes.Contains(chunk, []byte("prompt")) && !bytes.Contains(chunk, []byte("cache")) {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	for line := range bytes.SplitSeq(chunk, []byte("\n")) {
		t.observeLineLocked(line)
	}
}

func (t *promptCacheTracker) observeLineLocked(line []byte) {
	switch {
	case promptCacheUpdateBeginRegex.Match(line):
		t.pending = &api.PromptCacheSwap{}
	case promptCacheUpdateTookRegex.Match(line):
		if t.pending == nil {
			return
		}
		m := promptCacheUpdateTookRegex.FindSubmatch(line)
		t.pending.Ms, _ = strconv.ParseFloat(string(m[1]), 64)
		t.done, t.pending = t.pending, nil
	case promptCacheStateRegex.Match(line):
		m := promptCacheStateRegex.FindSubmatch(line)
		entries, _ := strconv.Atoi(string(m[1]))
		t.state = &api.PromptCacheState{Entries: entries, Bytes: mibToBytes(m[2]), LimitBytes: mibToBytes(m[3])}
	case promptCacheEntryRegex.Match(line):
		// One line per entry follows the state line; they are what carries the lengths.
		if t.state != nil {
			n, _ := strconv.Atoi(string(promptCacheEntryRegex.FindSubmatch(line)[1]))
			t.state.Tokens += n
		}
	}

	if t.pending == nil {
		return
	}
	switch {
	case promptCacheSavedRegex.Match(line):
		m := promptCacheSavedRegex.FindSubmatch(line)
		t.pending.SavedTokens, _ = strconv.Atoi(string(m[1]))
		t.pending.SavedBytes = mibToBytes(m[2])
	case promptCacheEvictedRegex.Match(line):
		t.pending.Evicted++
		t.pending.EvictedBytes += mibToBytes(promptCacheEvictedRegex.FindSubmatch(line)[1])
	case promptCacheTooLargeRegex.Match(line):
		// The save was refused, so what the save line said is not in the cache.
		t.pending.TooLarge = true
		t.pending.SavedTokens, t.pending.SavedBytes = 0, 0
	case promptCacheRestoredRegex.Match(line):
		t.pending.Restored = true
	}
}

// begin opens the window for one request.
func (t *promptCacheTracker) begin() {
	t.mu.Lock()
	t.pending, t.done = nil, nil
	t.mu.Unlock()
}

// take returns the swap the current request caused, or nil if it caused none.
func (t *promptCacheTracker) take() *api.PromptCacheSwap {
	t.mu.Lock()
	defer t.mu.Unlock()
	swap := t.done
	t.done = nil
	return swap
}

// snapshot returns the cache's occupancy as last reported, or nil before the first report.
func (t *promptCacheTracker) snapshot() *api.PromptCacheState {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.state == nil {
		return nil
	}
	state := *t.state
	return &state
}
