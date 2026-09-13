//go:build !linux

package server

func allocatedBytes(string) (int64, bool) { return 0, false }

func releasePageCache(string) {}
