package llm

import (
	"os"

	"golang.org/x/sys/windows"
)

// markSparse lets NTFS keep the file's unwritten ranges as holes. Without it, extending the file
// allocates every byte, and a synthetic model of 11 GiB takes 11 GiB of disk while it exists.
func markSparse(f *os.File) error {
	var n uint32
	return windows.DeviceIoControl(windows.Handle(f.Fd()), windows.FSCTL_SET_SPARSE, nil, 0, nil, 0, &n, nil)
}

// AllocatedBytes is not read on Windows; marking the file sparse is what is relied on there.
func AllocatedBytes(string) (int64, bool) { return 0, false }
