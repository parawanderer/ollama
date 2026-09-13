package server

import (
	"os"
	"syscall"

	"golang.org/x/sys/unix"
)

// allocatedBytes is the disk a file actually occupies, which for a sparse file is far less
// than its size.
func allocatedBytes(path string) (int64, bool) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, false
	}
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, false
	}
	return st.Blocks * 512, true
}

// releasePageCache asks the kernel to drop a file's cached pages. After a synthetic model has
// been loaded they are gigabytes of zeros, and they would otherwise stay until the file goes.
func releasePageCache(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	_ = unix.Fadvise(int(f.Fd()), 0, 0, unix.FADV_DONTNEED)
}
