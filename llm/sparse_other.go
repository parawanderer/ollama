//go:build !windows

package llm

import (
	"os"
	"syscall"
)

// markSparse is a no-op where the filesystem keeps holes in any file (ext4, xfs, btrfs, tmpfs,
// overlayfs, APFS).
func markSparse(*os.File) error { return nil }

// AllocatedBytes is the disk a file actually occupies, which for a sparse file is far less than
// its size.
func AllocatedBytes(path string) (int64, bool) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, false
	}
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, false
	}
	return int64(st.Blocks) * 512, true
}
