package power

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func writeRAPL(t *testing.T, root string, uj int64) {
	t.Helper()
	dir := filepath.Join(root, "intel-rapl:0")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, v := range map[string]string{"name": "package-0\n", "max_energy_range_uj": "1000000\n", "energy_uj": strconv.FormatInt(uj, 10) + "\n"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(v), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// a core domain beside it must not be picked
	core := filepath.Join(root, "intel-rapl:0:0")
	os.MkdirAll(core, 0o755)
	os.WriteFile(filepath.Join(core, "name"), []byte("core\n"), 0o644)
}

func TestRAPLPowerFromCounterAcrossWrap(t *testing.T) {
	root := t.TempDir()
	writeRAPL(t, root, 900000)
	p, err := newRAPL(root)
	if err != nil {
		t.Fatal(err)
	}
	r := p.(*rapl)
	first, err := r.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if first.PowerMW != nil {
		t.Fatalf("first read has no interval, so no power; got %d", *first.PowerMW)
	}
	if *first.EnergyMJ != 900 || *first.EnergyWrapMJ != 1000 {
		t.Fatalf("energy %d wrap %d, want 900 and 1000 mJ", *first.EnergyMJ, *first.EnergyWrapMJ)
	}
	// 200000 uJ later, across the wrap at 1000000, over a pinned 1 s interval: 200 mW.
	r.lastAt = time.Now().Add(-time.Second)
	writeRAPL(t, root, 100000)
	second, err := r.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if second.PowerMW == nil || *second.PowerMW < 195 || *second.PowerMW > 200 {
		t.Fatalf("power across the wrap = %v, want ~200 mW", second.PowerMW)
	}
}

func TestRAPLUnreadableIsAnError(t *testing.T) {
	if _, err := newRAPL(t.TempDir()); err == nil {
		t.Fatal("an empty powercap directory must not make a source")
	}
}

// recorder is a fake plug that keeps every request it was sent.
type recorder struct {
	mu   sync.Mutex
	reqs []*http.Request
	h    http.HandlerFunc
}

func (rc *recorder) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	rc.mu.Lock()
	rc.reqs = append(rc.reqs, r)
	rc.mu.Unlock()
	rc.h(w, r)
}

const tasmotaReply = `{"StatusSNS":{"Time":"2026-10-10T11:12:13","ENERGY":{"TotalStartTime":"2026-10-01T00:00:00","Total":12.345,"Yesterday":1.2,"Today":0.3,"Power":187,"ApparentPower":190,"ReactivePower":30,"Factor":0.98,"Voltage":231,"Current":0.82}}}`

// TestPlugsOnlyRead is the guard on the one rule that matters most here: a plug switches
// its socket, and the socket may feed this machine. Every request either provider ever sends
// must be a GET of its fixed status call, and nothing from the configuration may reach the
// command.
func TestPlugsOnlyRead(t *testing.T) {
	rc := &recorder{h: func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/rpc/") {
			w.Write([]byte(`{"apower":42.5,"aenergy":{"total":11.679,"minute_ts":1654511972}}`))
			return
		}
		w.Write([]byte(tasmotaReply))
	}}
	srv := httptest.NewServer(rc)
	defer srv.Close()

	for _, kind := range []string{"tasmota", "shelly"} {
		p, err := factories[kind](srv.URL)
		if err != nil {
			t.Fatal(kind, err)
		}
		for range 3 {
			if _, err := p.Read(context.Background()); err != nil {
				t.Fatal(kind, err)
			}
		}
	}
	if len(rc.reqs) != 6 {
		t.Errorf("%d requests, want 6", len(rc.reqs))
	}
	for _, r := range rc.reqs {
		if r.Method != http.MethodGet {
			t.Errorf("%s %s: only GET is allowed", r.Method, r.URL)
		}
		switch r.URL.Path {
		case "/cm":
			if c := r.URL.Query().Get("cmnd"); c != "Status 10" && c != "Status 8" {
				t.Errorf("tasmota sent command %q: only Status 10 or 8 is allowed", c)
			}
		case "/rpc/Switch.GetStatus":
		default:
			t.Errorf("request to %s: not a status call", r.URL.Path)
		}
	}

	// An address carrying a path or a query is refused, so a command cannot be smuggled in.
	for _, bad := range []string{srv.URL + "/cm?cmnd=Power%20Off", srv.URL + "/?cmnd=Power%200", srv.URL + "#x", "ftp://plug"} {
		if _, err := newTasmota(bad); err == nil {
			t.Errorf("tasmota accepted %q", bad)
		}
		if _, err := newShelly(bad); err == nil {
			t.Errorf("shelly accepted %q", bad)
		}
	}
}

func TestTasmotaReading(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(tasmotaReply))
	}))
	defer srv.Close()
	p, _ := newTasmota(srv.URL)
	r, err := p.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if *r.PowerMW != 187000 || *r.EnergyMJ != 44442000000 || r.SourceTime != "2026-10-10T11:12:13" || r.EnergyWrapMJ != nil {
		t.Fatalf("got power %d energy %d time %q", *r.PowerMW, *r.EnergyMJ, r.SourceTime)
	}
	if p.Scope() != ScopeWall {
		t.Fatalf("scope %s", p.Scope())
	}
}

func TestTasmotaFallsBackToStatus8AndSendsPassword(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		seen = append(seen, q.Get("cmnd"))
		if q.Get("user") != "admin" || q.Get("password") != "s3cret" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if q.Get("cmnd") == "Status 10" {
			w.Write([]byte(`{"Command":"Unknown"}`)) // an older firmware
			return
		}
		w.Write([]byte(tasmotaReply))
	}))
	defer srv.Close()
	p, err := newTasmota(strings.Replace(srv.URL, "http://", "http://admin:s3cret@", 1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Read(context.Background()); err != nil {
		t.Fatal(err)
	}
	if strings.Join(seen, ",") != "Status 10,Status 8" {
		t.Fatalf("commands %v", seen)
	}
}

// The error is published on /api/info, and a Tasmota password travels in the URL's query.
func TestPlugErrorDoesNotLeakPassword(t *testing.T) {
	p, _ := newTasmota("http://admin:s3cret@127.0.0.1:1") // nothing listens on port 1
	_, err := p.Read(context.Background())
	if err == nil {
		t.Fatal("expected a connection error")
	}
	if strings.Contains(err.Error(), "s3cret") {
		t.Fatalf("error leaks the password: %v", err)
	}
}

func TestParse(t *testing.T) {
	s, err := Parse("none")
	if err != nil || len(s.list) != 0 {
		t.Fatal("none must configure nothing")
	}
	s, err = Parse("tasmota@http://plug.lan, nosuch, tasmota")
	if err == nil {
		t.Fatal("an unknown kind and a missing address must be reported")
	}
	if len(s.list) != 1 || s.list[0].kind != "tasmota" {
		t.Fatalf("the valid entry must still run, got %d", len(s.list))
	}
	s2, err := Parse("box=tasmota@http://plug1.lan,monitor=tasmota@http://plug2.lan")
	if err != nil || len(s2.list) != 2 || s2.list[0].name != "box" || s2.list[1].name != "monitor" || s2.list[1].kind != "tasmota" {
		t.Fatalf("labels: %v %+v", err, s2.list)
	}
	if s.Latest() == nil || len(s.Latest()) != 0 {
		t.Fatal("a source not yet read is left out")
	}
}

func TestSetIntervals(t *testing.T) {
	s, err := Parse("rapl@/nonexistent,box=tasmota@http://plug1.lan,monitor=tasmota@http://plug2.lan")
	_ = err // rapl may fail to open here; the plugs are what is tested
	var box, monitor *source
	for _, src := range s.list {
		switch src.name {
		case "box":
			box = src
		case "monitor":
			monitor = src
		}
	}
	if box.interval() != 200*time.Millisecond {
		t.Fatalf("tasmota default %v, want 200ms", box.interval())
	}
	err = s.SetIntervals("tasmota=1s, box=300ms, nosuch=1s, monitor=10ms, box2=fast")
	if err == nil {
		t.Fatal("an unmatched key, a too-short and an unparsable duration must be reported")
	}
	for _, want := range []string{"nosuch", "below the minimum", "box2"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
	if box.interval() != 300*time.Millisecond {
		t.Errorf("a label must win over its kind: box %v, want 300ms", box.interval())
	}
	if monitor.interval() != time.Second {
		t.Errorf("a rejected label entry must leave the kind's: monitor %v, want 1s", monitor.interval())
	}
	if err := s.SetIntervals(""); err != nil {
		t.Errorf("empty spec: %v", err)
	}
}

func TestFailedReadClearsFigures(t *testing.T) {
	ok := true
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !ok {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Write([]byte(tasmotaReply))
	}))
	defer srv.Close()
	s, err := Parse("tasmota@" + srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	s.list[0].readOnce(context.Background())
	if l := s.Latest()[0]; l.PowerMW == nil || l.Error != "" || l.Scope != ScopeWall || l.Provider != "tasmota" {
		t.Fatalf("first sample %+v", l)
	}
	ok = false
	s.list[0].readOnce(context.Background())
	if l := s.Latest()[0]; l.PowerMW != nil || l.EnergyMJ != nil || l.Error == "" {
		t.Fatalf("a failed read must not repeat the old figures as current: %+v", l)
	}
}
