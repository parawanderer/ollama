package discover

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ollama/ollama/format"
)

func GetCPUMem() (memInfo, error) {
	mem, err := getCPUMem()
	if err != nil {
		return memInfo{}, err
	}
	return getCPUMemByCgroupsAt("/sys/fs/cgroup", mem), nil
}

func getCPUMem() (memInfo, error) {
	var mem memInfo
	var total, available, free, buffers, cached, freeSwap uint64
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return mem, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		switch {
		case strings.HasPrefix(line, "MemTotal:"):
			_, err = fmt.Sscanf(line, "MemTotal:%d", &total)
		case strings.HasPrefix(line, "MemAvailable:"):
			_, err = fmt.Sscanf(line, "MemAvailable:%d", &available)
		case strings.HasPrefix(line, "MemFree:"):
			_, err = fmt.Sscanf(line, "MemFree:%d", &free)
		case strings.HasPrefix(line, "Buffers:"):
			_, err = fmt.Sscanf(line, "Buffers:%d", &buffers)
		case strings.HasPrefix(line, "Cached:"):
			_, err = fmt.Sscanf(line, "Cached:%d", &cached)
		case strings.HasPrefix(line, "SwapFree:"):
			_, err = fmt.Sscanf(line, "SwapFree:%d", &freeSwap)
		default:
			continue
		}
		if err != nil {
			return mem, err
		}
	}
	mem.TotalMemory = total * format.KibiByte
	mem.FreeSwap = freeSwap * format.KibiByte
	if available > 0 {
		mem.FreeMemory = available * format.KibiByte
	} else {
		mem.FreeMemory = (free + buffers + cached) * format.KibiByte
	}
	return mem, nil
}

// getCPUMemByCgroupsAt narrows host memory to a cgroup v2 limit, when there is one.
//
// memory.current counts the page cache charged to the cgroup, and for a model server that is
// the model files it has read: 111 GB of this box's 111.3 GB "current" was file cache, which the
// kernel reclaims on demand. Counting it as used reported 21 GiB free on a host with 114 GiB
// available, and made CPU-only loads fail their fit checks. So file cache is not usage here,
// and with no limit at all (memory.max is "max") the host's MemAvailable is already the answer.
func getCPUMemByCgroupsAt(root string, mem memInfo) memInfo {
	limit, err := getUint64ValueFromFile(filepath.Join(root, "memory.max"))
	if err != nil {
		return mem // unlimited ("max") or no cgroup v2: the host figures stand
	}
	used, err := getUint64ValueFromFile(filepath.Join(root, "memory.current"))
	if err != nil {
		mem.TotalMemory = min(mem.TotalMemory, limit)
		return mem
	}
	if file, ok := memoryStatValue(filepath.Join(root, "memory.stat"), "file"); ok && file <= used {
		used -= file
	}
	free := uint64(0)
	if limit > used {
		free = limit - used
	}
	mem.TotalMemory = min(mem.TotalMemory, limit)
	mem.FreeMemory = min(mem.FreeMemory, free)
	return mem
}

// memoryStatValue reads one "key value" line from a cgroup memory.stat file.
func memoryStatValue(path, key string) (uint64, bool) {
	f, err := os.Open(path)
	if err != nil {
		return 0, false
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		k, v, ok := strings.Cut(s.Text(), " ")
		if ok && k == key {
			n, err := strconv.ParseUint(strings.TrimSpace(v), 10, 64)
			return n, err == nil
		}
	}
	return 0, false
}

func getUint64ValueFromFile(path string) (uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		return strconv.ParseUint(line, 10, 64)
	}
	return 0, errors.New("empty file content")
}

func IsNUMA() bool {
	ids := map[string]any{}
	packageIds, _ := filepath.Glob("/sys/devices/system/cpu/cpu*/topology/physical_package_id")
	for _, packageId := range packageIds {
		id, err := os.ReadFile(packageId)
		if err == nil {
			ids[strings.TrimSpace(string(id))] = struct{}{}
		}
	}
	return len(ids) > 1
}
