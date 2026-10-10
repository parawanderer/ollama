# Power sources

ollama reports GPU board power per device (`supported_gpus[].power`, from NVML). Everything
else -- the CPU package, the wall socket, a UPS -- comes from a **power source**, and each
configured source appears in `/api/info` (and so on every `/api/events` sample) under
`compute.power_sources`:

```json
{"provider":"tasmota","scope":"wall","power_mw":187000,"energy_mj":44442000000,
 "source_time":"2026-10-10T11:12:13","at_ms":1791633133000}
```

`scope` says how much of the machine the reading covers: `wall` (at the socket, PSU losses
included), `system` (the whole machine, DC side), `cpu` or `gpu`. **Never add readings of
different scopes together**: a `wall` reading already contains the CPU and the GPUs.

Every figure is optional; absent means not reported, never zero. A source whose latest read
failed carries `error` and no figures. `at_ms` is when ollama read it; `source_time` is the
source's own clock, verbatim, which may be in any timezone.

`energy_mj` is a cumulative counter. Integrating it (differences between samples) is exact;
integrating `power_mw` samples is an approximation. It can wrap (at `energy_wrap_mj`, if
given) or be reset at the source, so treat a decrease as one of those.

## Configuring

`OLLAMA_POWER_SOURCES` is a comma-separated list of `[label=]kind[@argument]`:

```
OLLAMA_POWER_SOURCES=rapl@/host/powercap,box=tasmota@http://admin:secret@192.168.1.50
```

The label is optional and appears as `name`; give one when two sources share a kind (two plugs).

`none` disables all of them. Unset, ollama tries `rapl` and says nothing if it is unreadable.

| kind | scope | argument | read every |
|---|---|---|---|
| `rapl` | cpu | powercap directory (default: tries `/sys/class/powercap`, then `/host/powercap`) | 250 ms |
| `tasmota` | wall | `http://[user:password@]host`; reads `Status 10` (falls back to `Status 8`) | 1 s |
| `shelly` | wall | `http://host`, Gen2/Gen3 `Switch.GetStatus?id=0`; no password support (Shelly uses digest auth) | 1 s |

**The plug providers only read.** A metering plug switches its socket, and it may be feeding
this machine. Each sends one fixed status request; an argument with a path or a query is
refused. `TestPlugsOnlyRead` enforces this, and a new plug provider must be added to it.

### RAPL in a container

The CPU's energy counter is root-only (a fine-grained energy counter is a side channel,
CVE-2020-8694), and Docker masks `/sys/devices/virtual/powercap` in every container. Bind-mount
the host's copy read-only elsewhere, in a container running as root:

```
docker run ... -v /sys/devices/virtual/powercap:/host/powercap:ro -e OLLAMA_POWER_SOURCES=rapl@/host/powercap ...
```

AMD exposes only `package-0` and `core`; there is no DRAM domain. Only the package is reported,
since `core` is part of it.

## Writing a provider

One file in this package:

```go
package power

type myUPS struct{ addr string }

func newMyUPS(arg string) (Provider, error) {
	if arg == "" {
		return nil, errors.New("needs the UPS address")
	}
	return &myUPS{addr: arg}, nil
}

func (u *myUPS) Scope() Scope            { return ScopeWall }      // what the reading covers
func (u *myUPS) Interval() time.Duration { return 2 * time.Second } // how often to read

// Read takes one reading. It runs on its own goroutine with a timeout, so a slow source
// never delays a sample. It must not change anything on the device.
func (u *myUPS) Read(ctx context.Context) (Reading, error) {
	w, err := queryMyUPS(ctx, u.addr)
	if err != nil {
		return Reading{}, err // published as "error"; do not put credentials in it
	}
	return Reading{PowerMW: ptr(int64(w * 1000))}, nil
}

func init() { Register("myups", newMyUPS) }
```

Then `OLLAMA_POWER_SOURCES=myups@10.0.0.5`. Set whichever of `PowerMW`, `EnergyMJ` (and
`EnergyWrapMJ`) and `SourceTime` the device gives you and leave the rest nil. Errors are
published on the API, so keep secrets out of them (see `getJSON` for the URL case).
