"""MIDI output via python-rtmidi.

Same stack as the DDJ bridge project: ``python-rtmidi`` for the actual bytes,
because it is the lowest-latency option on macOS (CoreMIDI directly, no Python
object churn per message).

Two ways to get the notes out:

* **virtual port** (default, macOS/Linux only) — the controller shows up in
  Logic, Ableton, GarageBand or rekordbox as a MIDI device called
  "Gesture MIDI Controller". Nothing to configure in Audio MIDI Setup.
* **existing port** (``--port``) — open a port that already exists, e.g. the
  ``IAC Driver Bus 1`` loopback or a USB-MIDI interface feeding a digital piano.

``NullOutput`` implements the same interface without touching any hardware and
backs ``--dry-run``, which is how you check a mapping file on a machine with no
MIDI at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

log = logging.getLogger(__name__)

DEFAULT_PORT_NAME = "Gesture MIDI Controller"

# MIDI status bytes (channel is added to the low nibble).
NOTE_OFF = 0x80
NOTE_ON = 0x90
AFTERTOUCH = 0xD0
CONTROL_CHANGE = 0xB0
PITCH_BEND = 0xE0

PITCH_BEND_CENTER = 8192
PITCH_BEND_MAX = 16383


def clamp7(value: int) -> int:
    """Clamp to a 7-bit MIDI data byte."""
    return max(0, min(127, int(round(value))))


def clamp14(value: int) -> int:
    """Clamp to a 14-bit MIDI value (pitch bend)."""
    return max(0, min(PITCH_BEND_MAX, int(round(value))))


@dataclass(frozen=True)
class MidiEvent:
    """One MIDI message, produced by the mapper and consumed by an output.

    ``channel`` is 1-16 as musicians count it; the wire format's 0-15 is applied
    in :meth:`to_bytes`.  ``value`` is 0-127 except for ``pitch_bend`` where it
    is the raw 14-bit value.
    """

    kind: str            # "cc" | "note_on" | "note_off" | "pitch_bend" | "aftertouch"
    channel: int = 1
    number: int = 0      # CC number or note number; unused for pitch bend
    value: int = 0
    label: str = ""      # mapping name, for logging and the on-screen overlay

    def to_bytes(self) -> List[int]:
        """Encode as raw MIDI bytes."""
        channel = max(0, min(15, int(self.channel) - 1))
        if self.kind == "cc":
            return [CONTROL_CHANGE | channel, clamp7(self.number), clamp7(self.value)]
        if self.kind == "note_on":
            return [NOTE_ON | channel, clamp7(self.number), clamp7(self.value)]
        if self.kind == "note_off":
            return [NOTE_OFF | channel, clamp7(self.number), clamp7(self.value)]
        if self.kind == "aftertouch":
            return [AFTERTOUCH | channel, clamp7(self.value)]
        if self.kind == "pitch_bend":
            bend = clamp14(self.value)
            return [PITCH_BEND | channel, bend & 0x7F, (bend >> 7) & 0x7F]
        raise ValueError(f"unknown MIDI event kind {self.kind!r}")

    def __str__(self) -> str:
        if self.kind == "cc":
            return f"CC{self.number:>3} = {self.value:>3} ch{self.channel} ({self.label})"
        if self.kind == "pitch_bend":
            return f"BEND  = {self.value:>5} ch{self.channel} ({self.label})"
        if self.kind == "aftertouch":
            return f"AT    = {self.value:>3} ch{self.channel} ({self.label})"
        return f"{self.kind.upper():<8} {self.number:>3} vel {self.value:>3} ch{self.channel} ({self.label})"


class BaseOutput:
    """Interface shared by the real and the no-op outputs."""

    name: str = "base"

    def send(self, event: MidiEvent) -> None:
        raise NotImplementedError

    def send_all(self, events: Iterable[MidiEvent]) -> None:
        for event in events:
            self.send(event)

    def close(self) -> None:
        pass

    def __enter__(self) -> "BaseOutput":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class NullOutput(BaseOutput):
    """Swallows (and optionally logs) events. Backs ``--dry-run``."""

    name = "null"

    def __init__(self, echo: bool = False) -> None:
        self.echo = echo
        self.events: List[MidiEvent] = []

    def send(self, event: MidiEvent) -> None:
        self.events.append(event)
        if self.echo:
            print(f"[dry-run] {event}", flush=True)


class MidiUnavailableError(RuntimeError):
    """Raised when the platform MIDI backend itself cannot be initialised."""


def _new_midi_out():
    """Construct an ``rtmidi.MidiOut``, turning backend errors into clear advice."""
    import rtmidi

    try:
        return rtmidi.MidiOut()
    except (rtmidi.SystemError, SystemError) as exc:
        raise MidiUnavailableError(
            f"could not initialise the system MIDI backend ({exc}). On macOS this "
            "usually means CoreMIDI is unavailable; on Linux, that ALSA sequencer "
            "support is missing. Use --dry-run to test a mapping without MIDI."
        ) from exc


def list_output_ports() -> Sequence[str]:
    """Names of the MIDI output ports currently available on this machine."""
    probe = _new_midi_out()
    try:
        return list(probe.get_ports())
    finally:
        del probe


class MidiOutput(BaseOutput):
    """python-rtmidi output: a virtual port, or an existing one matched by name.

    Args:
        port: substring of an existing port name, or its integer index. When
            ``None`` a virtual port named ``port_name`` is created instead.
        port_name: name of the virtual port other applications will see.
        verbose: log every message (useful while building a mapping).
    """

    name = "rtmidi"

    def __init__(
        self,
        port: Optional[object] = None,
        port_name: str = DEFAULT_PORT_NAME,
        verbose: bool = False,
    ) -> None:
        self._out = _new_midi_out()
        self.verbose = verbose
        self.port_name = port_name
        available = list(self._out.get_ports())

        if port is None:
            try:
                self._out.open_virtual_port(port_name)
            except (NotImplementedError, SystemError) as exc:
                raise RuntimeError(
                    "virtual MIDI ports are not supported on this platform. "
                    "Open an existing port instead, e.g. --port 'IAC Driver'"
                ) from exc
            self.description = f"virtual port {port_name!r}"
        else:
            index = self._resolve_port(port, available)
            self._out.open_port(index)
            self.description = f"port {index}: {available[index]!r}"
        log.info("MIDI output on %s", self.description)

    @staticmethod
    def _resolve_port(port: object, available: Sequence[str]) -> int:
        """Resolve an index or a case-insensitive name fragment to a port index."""
        if not available:
            raise RuntimeError(
                "no MIDI output ports found. On macOS enable the IAC Driver in "
                "Audio MIDI Setup, or run with --virtual to create one."
            )
        if isinstance(port, int) or str(port).isdigit():
            index = int(port)
            if not 0 <= index < len(available):
                raise ValueError(
                    f"MIDI port index {index} out of range; available: "
                    + ", ".join(f"{i}: {n}" for i, n in enumerate(available))
                )
            return index

        needle = str(port).lower()
        matches = [i for i, candidate in enumerate(available) if needle in candidate.lower()]
        if not matches:
            raise ValueError(
                f"no MIDI output port matching {port!r}; available: "
                + ", ".join(f"{i}: {n}" for i, n in enumerate(available))
            )
        if len(matches) > 1:
            log.warning(
                "%r matches several ports (%s); using the first",
                port,
                ", ".join(available[i] for i in matches),
            )
        return matches[0]

    def send(self, event: MidiEvent) -> None:
        self._out.send_message(event.to_bytes())
        if self.verbose:
            log.info("%s", event)

    def close(self) -> None:
        if getattr(self, "_out", None) is not None:
            self._out.close_port()
            del self._out
            self._out = None


def open_output(
    port: Optional[object] = None,
    port_name: str = DEFAULT_PORT_NAME,
    dry_run: bool = False,
    verbose: bool = False,
) -> BaseOutput:
    """Open the requested output, falling back to :class:`NullOutput` on ``dry_run``."""
    if dry_run:
        log.info("dry run: MIDI messages will be printed, not sent")
        return NullOutput(echo=True)
    return MidiOutput(port=port, port_name=port_name, verbose=verbose)
