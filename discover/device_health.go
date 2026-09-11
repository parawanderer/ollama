package discover

import "github.com/ollama/ollama/ml"

// NVML status codes, from nvml.h. Only the ones that describe a device fault are named:
// the rest are call-level problems that say nothing about the hardware.
const (
	nvmlSuccess               = 0
	nvmlErrorNotSupported     = 3
	nvmlErrorNoPermission     = 4
	nvmlErrorInsufficientPwr  = 8
	nvmlErrorIRQIssue         = 11
	nvmlErrorCorruptedInfoROM = 14
	nvmlErrorGPUIsLost        = 15
	nvmlErrorResetRequired    = 16
	nvmlErrorOSBlocked        = 17
)

// deviceProbe is what was learned about one device the compute backend did not offer,
// separated from how it was learned so the classification can be tested without a GPU --
// and specifically without a BROKEN GPU, which is not a thing a test can arrange.
type deviceProbe struct {
	PCIID string
	Name  string
	UUID  string

	// Status is the NVML return code from a call that reads live state from the device,
	// and DetailFromDriver is nvmlErrorString of that code.
	//
	// WHICH call this came from matters, and the obvious choice is wrong. Measured on a
	// GPU that the driver had marked as needing a reset:
	//
	//   nvmlDeviceGetTemperature      -> NVML_ERROR_RESET_REQUIRED   (the signal)
	//   nvmlDeviceGetPowerUsage       -> NVML_ERROR_NOT_SUPPORTED    (indistinguishable from
	//   nvmlDeviceGetUtilizationRates -> NVML_ERROR_NOT_SUPPORTED     a card lacking the sensor)
	//   nvmlDeviceGetMemoryInfo       -> SUCCESS, with free memory identical to the
	//                                    HEALTHY card -- so a health check built on memory
	//                                    silently never fires
	//
	// Temperature is the one that reports the fault, so that is what fills this field.
	Status           int
	DetailFromDriver string

	Bus *ml.DeviceBusState
}

// classifyUnavailable turns a probe into the reason a device cannot be used.
//
// The bus state is deliberately allowed to OVERRIDE the driver's code in one direction. A
// driver that has lost a device reports it as lost whether the card fell off the bus or
// merely stopped answering, and those have different causes and different fixes -- reseat
// and power on one hand, a reset on the other. sysfs still knows which, because it does not
// need the driver's cooperation to see whether the device is enumerated.
func classifyUnavailable(p deviceProbe) ml.UnavailableDevice {
	d := ml.UnavailableDevice{
		PCIID:  p.PCIID,
		Name:   p.Name,
		UUID:   p.UUID,
		Detail: p.DetailFromDriver,
		Bus:    p.Bus,
	}

	// Gone from the bus outranks anything the driver says, because no GPU-level fix applies
	// to a device that is not electrically there.
	if p.Bus != nil && !p.Bus.Present {
		d.Reason = "not_enumerated"
		d.Recovery = "check the card is seated and powered; the device is not on the PCIe bus"
		return d
	}

	switch p.Status {
	case nvmlErrorResetRequired:
		d.Reason = "reset_required"
		// Deliberately does not name nvidia-smi -r. That is the documented fix and it is
		// REFUSED on consumer and workstation cards -- an RTX PRO 6000 answers "GPU
		// 00000000:03:00.0: Not Supported" while its PCI reset_method reads "flr bus", so
		// the kernel can reset a device the vendor tool will not. Naming one tool that
		// fails on the commonest hardware reads as a dead end rather than a first step.
		d.Recovery = "a cold power cycle: shut down, wait for the rails to drain, power on. " +
			"nvidia-smi -r is refused on many consumer and workstation cards, and unloading the " +
			"kernel modules is worse than useless here -- the unload has to talk to the dead GPU " +
			"to release it, so it blocks for minutes and takes the healthy cards down with it"
	case nvmlErrorGPUIsLost:
		d.Reason = "lost"
		d.Recovery = "the driver can no longer reach the device; a reset may work, otherwise reseat and power-cycle"
	case nvmlErrorInsufficientPwr:
		d.Reason = "insufficient_power"
		d.Recovery = "check the power connectors are fully seated at both ends"
	case nvmlErrorIRQIssue:
		d.Reason = "irq_issue"
		d.Recovery = "the device's interrupt line is not working; check the kernel log"
	case nvmlErrorCorruptedInfoROM:
		d.Reason = "corrupted_inforom"
		d.Recovery = "the device's on-board configuration store is damaged; this is a vendor RMA"
	case nvmlErrorOSBlocked:
		d.Reason = "blocked_by_os"
		d.Recovery = "the operating system is denying access to the device; check cgroups, containers and permissions"
	case nvmlErrorNoPermission:
		d.Reason = "no_permission"
		d.Recovery = "this process may not query the device; check device node permissions"
	case nvmlSuccess:
		// The device answers every question put to it and the compute backend still will
		// not offer it. That is a real state and it must not be reported as a GPU fault:
		// the usual causes are a visibility filter (CUDA_VISIBLE_DEVICES) or a backend
		// that does not support this architecture.
		d.Reason = "not_offered_by_backend"
		d.Recovery = "the device is healthy but no compute backend claimed it; check device visibility filters"
	default:
		d.Reason = "unknown"
	}

	// A link reporting errors is worth saying out loud whatever the driver's verdict, since
	// it points somewhere else entirely -- the slot, the riser, the cabling.
	if p.Bus != nil && (p.Bus.FatalErrors > 0 || p.Bus.NonFatalErrors > 0) && d.Reason != "not_enumerated" {
		d.Recovery = "PCIe errors are logged against this device; suspect the slot or riser before the card. " + d.Recovery
	}

	return d
}
