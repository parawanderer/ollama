package ml

import "testing"

// The formula is validated against published figures on three memory technologies, so a
// factor that happens to fit one card cannot pass all three.
func TestMemoryBandwidthMatchesPublishedFigures(t *testing.T) {
	for _, tt := range []struct {
		name      string
		bits, mhz int
		wantGBps  uint64
	}{
		{"RTX PRO 6000 (GDDR7)", 512, 14001, 1792},
		{"RTX 3090 (GDDR6X)", 384, 9751, 936},
		{"A100 80GB (HBM2e)", 5120, 1512, 1935},
	} {
		d := DeviceInfo{MemoryBusWidthBits: tt.bits, MemoryClockMaxMHz: tt.mhz}
		if got := d.MemoryBandwidth() / 1_000_000_000; got != tt.wantGBps {
			t.Errorf("%s: %d GB/s, want the published %d", tt.name, got, tt.wantGBps)
		}
	}
	if (DeviceInfo{MemoryBusWidthBits: 512}).MemoryBandwidth() != 0 {
		t.Error("a bandwidth was derived from half its inputs")
	}
}
