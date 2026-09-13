package server

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
)

func cudaDev(pci, name string) ml.DeviceInfo {
	return ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "CUDA"}, PCIID: pci, Name: name}
}

// With the first of two cards gone the survivor is the new CUDA0. The faulted card must
// keep the name it had, not inherit the shift.
func TestDeviceNamesKeepWhatAFaultedCardWasCalled(t *testing.T) {
	d := newDeviceNames(nil)
	d.observe([]ml.DeviceInfo{cudaDev("0000:01:00.0", "CUDA0"), cudaDev("0000:03:00.0", "CUDA1")})
	d.observe([]ml.DeviceInfo{cudaDev("0000:03:00.0", "CUDA0")}) // 01:00.0 faulted and left

	if s, ok := d.lookup("0000:01:00.0"); !ok || s.Name != "CUDA0" {
		t.Errorf("faulted card: %+v %v, want its last healthy name CUDA0", s, ok)
	}
	if s, _ := d.lookup("0000:03:00.0"); s.Name != "CUDA0" {
		t.Errorf("survivor: %+v, want its current name", s)
	}
	if _, ok := d.lookup("0000:09:00.0"); ok {
		t.Error("an address never seen healthy must have no name")
	}
}

// The case that happened: the server restarted after the card had already faulted.
func TestDeviceNamesSurviveARestart(t *testing.T) {
	dir := t.TempDir()
	db := testServerDB(t, dir)
	newDeviceNames(db).observe([]ml.DeviceInfo{cudaDev("0000:01:00.0", "CUDA0"), cudaDev("0000:03:00.0", "CUDA1")})
	db.close()

	restarted := newDeviceNames(testServerDB(t, dir))
	if s, ok := restarted.lookup("0000:03:00.0"); !ok || s.Name != "CUDA1" || s.LastSeen.IsZero() {
		t.Fatalf("after a restart: %+v %v, want CUDA1 with a time", s, ok)
	}
}

// Enumeration runs every few seconds while anything polls; the file is rewritten only when a
// name changes or the last write is older than the persist interval.
func TestDeviceNamesDoNotRewriteOnEveryRefresh(t *testing.T) {
	now := time.Date(2026, 9, 12, 8, 0, 0, 0, time.UTC)
	d := newDeviceNames(testServerDB(t, t.TempDir()))
	d.now = func() time.Time { return now }
	devs := []ml.DeviceInfo{cudaDev("0000:01:00.0", "CUDA0")}

	d.observe(devs)
	first := d.persisted
	now = now.Add(time.Minute)
	d.observe(devs)
	if d.persisted != first {
		t.Fatal("rewrote an unchanged record a minute later")
	}
	d.observe([]ml.DeviceInfo{cudaDev("0000:01:00.0", "CUDA1")})
	if d.persisted == first {
		t.Fatal("a changed name was not written")
	}
	written := d.persisted
	now = now.Add(deviceNamesPersistEvery)
	d.observe([]ml.DeviceInfo{cudaDev("0000:01:00.0", "CUDA1")})
	if d.persisted == written {
		t.Fatal("last_seen was never refreshed on disk")
	}
}

func TestUnavailableGPUCarriesItsLastName(t *testing.T) {
	d := newDeviceNames(nil)
	d.observe([]ml.DeviceInfo{cudaDev("0000:03:00.0", "CUDA1")})

	got := unavailableGPU(ml.UnavailableDevice{PCIID: "0000:03:00.0", Reason: "reset_required"}, d)
	if got.LastName != "CUDA1" || got.LastSeen == nil {
		t.Fatalf("got %+v, want last_name CUDA1 with last_seen", got)
	}
	for _, names := range []*deviceNames{nil, newDeviceNames(nil)} {
		b, _ := json.Marshal(unavailableGPU(ml.UnavailableDevice{PCIID: "0000:03:00.0", Reason: "reset_required"}, names))
		if strings.Contains(string(b), "last_name") || strings.Contains(string(b), "last_seen") {
			t.Errorf("a card never seen healthy was given a name: %s", b)
		}
	}
}
