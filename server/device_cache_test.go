package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
)

// cacheTestScheduler counts discoveries, which on a real host each spawn a llama-server on
// every GPU.
func cacheTestScheduler(unavailable *[]ml.UnavailableDevice) (*Scheduler, *int) {
	discoveries := 0
	s := &Scheduler{
		deviceCacheTTL:     2 * time.Second,
		deviceCacheLiveTTL: 10 * time.Minute,
		getGpuFn: func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
			discoveries++
			return []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, PCIID: "0000:01:00.0"},
				{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}, PCIID: "0000:03:00.0"},
			}
		},
		unavailableFn: func([]string) []ml.UnavailableDevice { return *unavailable },
	}
	return s, &discoveries
}

// readsLive makes the driver answer free memory, as NVML does on this box.
func readsLive(s *Scheduler) {
	s.freeMemoryFn = func(ids []string) map[string]uint64 {
		free := map[string]uint64{}
		for _, id := range ids {
			free[id] = 90 << 30
		}
		return free
	}
}

// The sampler reads every 15 s when idle. With free memory read live, that must not
// rediscover: it used to, starting a llama-server on both cards every ~16 s.
func TestIdleReadsDoNotRediscoverWhileFreeMemoryIsLive(t *testing.T) {
	var none []ml.UnavailableDevice
	s, discoveries := cacheTestScheduler(&none)
	readsLive(s)

	s.cachedDevices(t.Context())
	s.deviceCacheMu.Lock()
	s.deviceCacheAt = time.Now().Add(-15 * time.Second) // one idle sampler tick later
	s.deviceCacheMu.Unlock()
	s.cachedDevices(t.Context())

	if *discoveries != 1 {
		t.Fatalf("%d discoveries across an idle tick, want 1", *discoveries)
	}
}

// Without a live reading the short TTL still stands, since it is then the only thing
// keeping free memory current.
func TestWithoutLiveFreeMemoryTheShortTTLStands(t *testing.T) {
	var none []ml.UnavailableDevice
	s, discoveries := cacheTestScheduler(&none)

	s.cachedDevices(t.Context())
	s.deviceCacheMu.Lock()
	s.deviceCacheAt = time.Now().Add(-15 * time.Second)
	s.deviceCacheMu.Unlock()
	s.cachedDevices(t.Context())

	if *discoveries != 2 {
		t.Fatalf("%d discoveries, want 2: without a live reading the cache must expire", *discoveries)
	}
}

// The churn used to be what noticed a card faulting while cached. Now the health verdict
// does, and a cached card that faulted drops the cache at once -- but only once per
// verdict, so a faulted card discovery goes on returning cannot make every read rediscover.
func TestACachedCardThatFaultsDropsTheCache(t *testing.T) {
	var unavailable []ml.UnavailableDevice
	s, discoveries := cacheTestScheduler(&unavailable)
	readsLive(s)

	s.cachedDevices(t.Context())
	unavailable = []ml.UnavailableDevice{{PCIID: "0000:03:00.0", Reason: "reset_required"}}
	s.cachedDevices(t.Context())
	if *discoveries != 2 {
		t.Fatalf("%d discoveries after a cached card faulted, want 2", *discoveries)
	}

	s.cachedDevices(t.Context())
	s.cachedDevices(t.Context())
	if *discoveries != 2 {
		t.Fatalf("%d discoveries: a fault discovery still reports must not rediscover on every read", *discoveries)
	}
}

// A device that was never offered is unavailable_gpus' business, not a reason to rediscover.
func TestAnUnofferedDeviceDoesNotDropTheCache(t *testing.T) {
	unavailable := []ml.UnavailableDevice{{PCIID: "0000:79:00.0", Reason: "not_offered_by_backend"}}
	s, discoveries := cacheTestScheduler(&unavailable)
	readsLive(s)

	s.cachedDevices(t.Context())
	s.cachedDevices(t.Context())
	if *discoveries != 1 {
		t.Fatalf("%d discoveries, want 1", *discoveries)
	}
}
