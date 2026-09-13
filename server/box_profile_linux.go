package server

import (
	"os"

	"golang.org/x/sys/unix"
)

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
