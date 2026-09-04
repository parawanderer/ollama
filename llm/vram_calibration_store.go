package llm

import (
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// calibrationFileVersion guards the on-disk format. A file written by a different version
// is discarded rather than migrated: the samples are cheap to re-measure and a
// misinterpreted one is a wrong placement.
const calibrationFileVersion = 1

type persistedSample struct {
	NumCtx int    `json:"num_ctx"`
	VRAM   uint64 `json:"vram"`
}

type persistedEntry struct {
	Key     CalibrationKey    `json:"key"`
	Samples []persistedSample `json:"samples"`
}

type persistedFile struct {
	Version int              `json:"version"`
	Written time.Time        `json:"written"`
	Entries []persistedEntry `json:"entries"`
}

// Load reads previously measured samples from path.
//
// Without this every restart returns every model to a cold estimate, so the measurement
// has to be re-earned by a load that was already paid for once. A model whose memory use
// is not derivable from its metadata is exactly the case that benefits, and exactly the
// case where re-earning it means one wrong placement per restart.
//
// A file that cannot be read or understood is ignored rather than fatal. Starting cold is
// the behaviour without this file at all, so it is always a safe outcome.
func (c *VRAMCalibration) Load(path string) {
	if c == nil || path == "" {
		return
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Debug("could not read VRAM calibration", "path", path, "error", err)
		}
		return
	}

	var file persistedFile
	if err := json.Unmarshal(data, &file); err != nil {
		slog.Info("discarding unreadable VRAM calibration", "path", path, "error", err)
		return
	}
	if file.Version != calibrationFileVersion {
		slog.Info("discarding VRAM calibration written by another version",
			"path", path, "found", file.Version, "want", calibrationFileVersion)
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	for _, entry := range file.Entries {
		samples := make([]calibrationSample, 0, len(entry.Samples))
		for _, s := range entry.Samples {
			if s.NumCtx > 0 && s.VRAM > 0 {
				samples = append(samples, calibrationSample{numCtx: s.NumCtx, vram: s.VRAM})
			}
		}
		if len(samples) > 0 {
			c.samples[entry.Key] = samples
		}
	}
	slog.Debug("loaded VRAM calibration", "path", path, "models", len(c.samples))
}

// Save writes the samples to path, atomically, so a crash mid-write leaves the previous
// file rather than a truncated one.
func (c *VRAMCalibration) Save(path string) error {
	if c == nil || path == "" {
		return nil
	}

	c.mu.Lock()
	file := persistedFile{Version: calibrationFileVersion, Written: time.Now().UTC()}
	for key, samples := range c.samples {
		entry := persistedEntry{Key: key}
		for _, s := range samples {
			entry.Samples = append(entry.Samples, persistedSample{NumCtx: s.numCtx, VRAM: s.vram})
		}
		file.Entries = append(file.Entries, entry)
	}
	c.mu.Unlock()

	data, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// persistOnce serialises concurrent saves so a burst of loads does not have several
// writers racing for the same file.
var persistOnce sync.Mutex

// Persist writes the samples, logging rather than returning an error: losing calibration
// costs a cold estimate on the next start, which is not worth failing a load over.
func (c *VRAMCalibration) Persist(path string) {
	persistOnce.Lock()
	defer persistOnce.Unlock()
	if err := c.Save(path); err != nil {
		slog.Debug("could not write VRAM calibration", "path", path, "error", err)
	}
}
