package discover

import (
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
)

// unavailableTTL is how long a health verdict is reused.
//
// Measured on this box: opening libnvidia-ml, calling nvmlInit, enumerating and asking each
// device for its temperature costs a median of 16.9 ms, against 1.1 ms for the whole of
// /api/info. The sampler reads that endpoint as often as twice a second, so probing per call
// would make the endpoint 15x more expensive to answer a question whose answer changes maybe
// once in the life of a machine.
//
// The sysfs half is free by comparison -- 0.042 ms for every device -- but it is not useful
// on its own, because the fault this exists to catch leaves the device enumerated and the
// link up. So the expensive half is the load-bearing half, and caching is the only lever.
//
// 30 seconds is chosen against the failure it replaces: a GPU sat faulted for five and a
// half hours with nothing reporting it. Thirty seconds of staleness is not a meaningful
// difference to an operator, and it costs 0.06% of one core.
const unavailableTTL = 30 * time.Second

var unavailable struct {
	sync.Mutex
	key    string
	at     time.Time
	result []ml.UnavailableDevice
}

// CachedUnavailableDevices is UnavailableDevices with the probe cost amortised.
//
// The cache is keyed on the set of devices discovery returned, so a device appearing or
// disappearing re-probes immediately rather than waiting out the TTL. That is the case that
// matters: a GPU going missing from discovery is exactly when the question "is it broken or
// was it never there" is being asked.
func CachedUnavailableDevices(known []string) []ml.UnavailableDevice {
	key := strings.Join(known, ",")

	unavailable.Lock()
	defer unavailable.Unlock()

	if unavailable.key == key && time.Since(unavailable.at) < unavailableTTL {
		return unavailable.result
	}

	unavailable.key = key
	unavailable.at = time.Now()
	unavailable.result = UnavailableDevices(known)
	return unavailable.result
}
