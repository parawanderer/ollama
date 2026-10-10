package power

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"time"
)

// The smart-plug providers. Both only ever READ. A metering plug usually also switches its
// socket, and the machine reading it may be the very load it switches, so each provider
// sends exactly one fixed status request and nothing configurable reaches the request:
// the argument is a base URL, and a URL carrying a path, query or fragment is refused rather
// than passed on. TestPlugsOnlyRead holds both to this.

var plugClient = &http.Client{Timeout: 2 * time.Second}

// plugBase validates a plug's address: scheme and host, optionally user:password, nothing else.
func plugBase(arg string) (*url.URL, error) {
	if arg == "" {
		return nil, errors.New("needs the plug's address, e.g. tasmota@http://192.168.1.50")
	}
	u, err := url.Parse(arg)
	if err != nil {
		return nil, err
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("%q: want http:// or https://", arg)
	}
	if u.Host == "" || (u.Path != "" && u.Path != "/") || u.RawQuery != "" || u.Fragment != "" {
		return nil, fmt.Errorf("%q: give the plug's address only, with no path or query", arg)
	}
	return u, nil
}

func getJSON(ctx context.Context, u string, v any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	resp, err := plugClient.Do(req)
	if err != nil {
		// A *url.Error prints the whole URL, and a Tasmota password travels in its query.
		// The error is published on /api/info, so drop the URL and keep the cause.
		var ue *url.Error
		if errors.As(err, &ue) {
			return fmt.Errorf("%s: %w", ue.Op, ue.Err)
		}
		return err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	return json.Unmarshal(body, v)
}

func milli(v float64) *int64 { return ptr(int64(math.Round(v * 1000))) }

// tasmota reads a Tasmota plug's energy sensor with "Status 10" ("connected sensor
// information"; it replaced "Status 8", which is tried when a firmware predates it).
// A web password is given in the URL, tasmota@http://admin:secret@192.168.1.50, and sent
// the way Tasmota's HTTP API takes it, as user and password query parameters.
type tasmota struct {
	base *url.URL
}

func newTasmota(arg string) (Provider, error) {
	u, err := plugBase(arg)
	if err != nil {
		return nil, err
	}
	return &tasmota{base: u}, nil
}

func (t *tasmota) Scope() Scope            { return ScopeWall }
func (t *tasmota) Interval() time.Duration { return time.Second }

func (t *tasmota) statusURL(n int) string {
	q := url.Values{}
	if t.base.User != nil {
		q.Set("user", t.base.User.Username())
		if p, ok := t.base.User.Password(); ok {
			q.Set("password", p)
		}
	}
	q.Set("cmnd", fmt.Sprintf("Status %d", n))
	u := url.URL{Scheme: t.base.Scheme, Host: t.base.Host, Path: "/cm", RawQuery: q.Encode()}
	return u.String()
}

// tasmotaStatus is the part of the reply read. Power is a number on a one-channel plug and
// an array on a multi-channel one; Total is kWh since the plug's counter was last reset.
type tasmotaStatus struct {
	StatusSNS *struct {
		Time   string `json:"Time"`
		ENERGY *struct {
			Power json.RawMessage `json:"Power"`
			Total *float64        `json:"Total"`
		} `json:"ENERGY"`
	} `json:"StatusSNS"`
}

func (t *tasmota) Read(ctx context.Context) (Reading, error) {
	var st tasmotaStatus
	err := getJSON(ctx, t.statusURL(10), &st)
	if err == nil && st.StatusSNS == nil {
		st = tasmotaStatus{}
		err = getJSON(ctx, t.statusURL(8), &st)
	}
	if err != nil {
		return Reading{}, err
	}
	if st.StatusSNS == nil || st.StatusSNS.ENERGY == nil {
		return Reading{}, errors.New("reply has no StatusSNS.ENERGY: not a metering plug, or the energy driver is off")
	}
	e := st.StatusSNS.ENERGY
	r := Reading{SourceTime: st.StatusSNS.Time}
	if w, ok := sumPower(e.Power); ok {
		r.PowerMW = milli(w)
	}
	if e.Total != nil {
		r.EnergyMJ = ptr(int64(math.Round(*e.Total * 3.6e9))) // kWh to mJ
	}
	if r.PowerMW == nil && r.EnergyMJ == nil {
		return Reading{}, errors.New("ENERGY carries neither Power nor Total")
	}
	return r, nil
}

func sumPower(raw json.RawMessage) (float64, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	var one float64
	if json.Unmarshal(raw, &one) == nil {
		return one, true
	}
	var many []float64
	if json.Unmarshal(raw, &many) == nil && len(many) > 0 {
		var s float64
		for _, v := range many {
			s += v
		}
		return s, true
	}
	return 0, false
}

// shelly reads a Shelly Gen2/Gen3 plug's first switch with Switch.GetStatus. Authentication
// is not supported: Shelly's RPC uses digest auth, which is not implemented here, so a plug
// with a password set answers 401 and the source reports that error.
type shelly struct {
	base *url.URL
}

func newShelly(arg string) (Provider, error) {
	u, err := plugBase(arg)
	if err != nil {
		return nil, err
	}
	if u.User != nil {
		return nil, errors.New("authentication is not supported for shelly (it needs digest auth)")
	}
	return &shelly{base: u}, nil
}

func (s *shelly) Scope() Scope            { return ScopeWall }
func (s *shelly) Interval() time.Duration { return time.Second }

func (s *shelly) Read(ctx context.Context) (Reading, error) {
	u := url.URL{Scheme: s.base.Scheme, Host: s.base.Host, Path: "/rpc/Switch.GetStatus", RawQuery: "id=0"}
	var st struct {
		APower  *float64 `json:"apower"`
		AEnergy *struct {
			Total    *float64 `json:"total"` // Wh
			MinuteTS *int64   `json:"minute_ts"`
		} `json:"aenergy"`
	}
	if err := getJSON(ctx, u.String(), &st); err != nil {
		return Reading{}, err
	}
	var r Reading
	if st.APower != nil {
		r.PowerMW = milli(*st.APower)
	}
	if st.AEnergy != nil && st.AEnergy.Total != nil {
		r.EnergyMJ = ptr(int64(math.Round(*st.AEnergy.Total * 3.6e6))) // Wh to mJ
		if st.AEnergy.MinuteTS != nil {
			r.SourceTime = fmt.Sprint(*st.AEnergy.MinuteTS)
		}
	}
	if r.PowerMW == nil && r.EnergyMJ == nil {
		return Reading{}, errors.New("reply has neither apower nor aenergy: not a metering device")
	}
	return r, nil
}

func init() {
	Register("tasmota", newTasmota)
	Register("shelly", newShelly)
}
