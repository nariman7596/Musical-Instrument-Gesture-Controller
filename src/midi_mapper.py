"""Feature -> MIDI translation, driven entirely by a JSON file.

Nothing about *which* gesture controls *which* parameter lives in the code: the
mapping file is the instrument's patch.  A mapping entry names a feature (see
:mod:`gesture_features`), a MIDI message type, and how to scale one into the
other.

Five message types are supported:

===========  =====================================================
``cc``       continuous feature -> control change (0-127)
``gate``     pose gate -> control change on/off (sustain pedal, ...)
``note``     pose gate -> note on/off (trigger a drum or sample)
``pitch_bend``  continuous feature -> 14-bit pitch bend
``aftertouch``  continuous feature -> channel pressure
===========  =====================================================

Two details matter for the result sounding musical rather than robotic:

* **Deadband.** A CC is only sent when the rounded value actually moves far
  enough, so a still hand produces no traffic at all.
* **Hysteresis.** Gates use a Schmitt trigger (``threshold`` / ``release``)
  instead of one threshold, so a hand hovering at the edge of "fist" cannot
  machine-gun the sustain pedal.

The file can be edited while the controller is running: :class:`ConfigWatcher`
re-reads it when the mtime changes and keeps the previous configuration if the
new one does not parse.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping as TMapping, Optional, Sequence, Tuple

from .gesture_features import FeatureCalibration, clamp01, feature_names, normalise
from .midi_output import DEFAULT_PORT_NAME, MidiEvent, PITCH_BEND_MAX
from .smoother import EMASmoother, SchmittTrigger

log = logging.getLogger(__name__)

CONFIG_VERSION = 1


class ConfigError(ValueError):
    """Raised for an invalid mapping file, with a message aimed at the user."""


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------
def _check_keys(data: TMapping[str, Any], allowed: Sequence[str], context: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{context}: unknown key(s) {', '.join(unknown)}; "
            f"allowed keys are {', '.join(sorted(allowed))}"
        )


def _get_number(data: TMapping[str, Any], key: str, default: float, context: str) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context}: {key!r} must be a number, got {value!r}")
    return float(value)


def _get_int(data: TMapping[str, Any], key: str, default: Optional[int], context: str) -> int:
    value = data.get(key, default)
    if value is None:
        raise ConfigError(f"{context}: {key!r} is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise ConfigError(f"{context}: {key!r} must be an integer, got {value!r}")
    return int(value)


def _get_range(
    data: TMapping[str, Any], key: str, default: Tuple[float, float], context: str
) -> Tuple[float, float]:
    value = data.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{context}: {key!r} must be [low, high], got {value!r}")
    low, high = float(value[0]), float(value[1])
    if low == high:
        raise ConfigError(f"{context}: {key!r} low and high must differ")
    return low, high


# --------------------------------------------------------------------------
# Mapping types
# --------------------------------------------------------------------------
@dataclass
class MappingState:
    """Snapshot of one mapping, used by the on-screen overlay."""

    name: str
    kind: str
    label: str          # e.g. "CC 74" or "Note 36"
    value: int          # last value sent
    normalised: float   # 0..1, for drawing a bar
    active: bool        # gate/note currently on
    tracked: bool       # the driving feature was present in this frame


@dataclass
class BaseMapping:
    """Common fields of every mapping entry."""

    name: str
    feature: str
    channel: int = 1
    enabled: bool = True
    smoothing: Optional[float] = None  # per-mapping EMA window override

    kind: str = field(init=False, default="base")

    def __post_init__(self) -> None:
        self.tracked = False
        self.reset()

    # -- interface ---------------------------------------------------------
    def update(self, features: TMapping[str, float], now: float) -> List[MidiEvent]:
        raw = features.get(self.feature)
        self.tracked = raw is not None
        if not self.enabled or raw is None:
            return []
        return self._update(float(raw), now)

    def _update(self, value: float, now: float) -> List[MidiEvent]:
        raise NotImplementedError

    def reset(self) -> None:
        self.last_value: Optional[int] = None
        self.display: float = 0.0
        self.active: bool = False
        self._last_sent_at: float = -math.inf

    def panic(self) -> List[MidiEvent]:
        """Events needed to leave the receiver in a clean state on shutdown."""
        return []

    def state(self) -> MappingState:
        return MappingState(
            name=self.name,
            kind=self.kind,
            label=self.label,
            value=self.last_value if self.last_value is not None else 0,
            normalised=clamp01(self.display),
            active=self.active,
            tracked=self.tracked,
        )

    @property
    def label(self) -> str:
        return self.kind


@dataclass
class ContinuousMapping(BaseMapping):
    """Continuous feature -> CC / pitch bend / aftertouch."""

    number: int = 0                              # CC number (unused for bend/AT)
    input_range: Tuple[float, float] = (0.0, 1.0)
    output_range: Optional[Tuple[float, float]] = None
    invert: bool = False
    curve: float = 1.0                           # >1 = finer control near the bottom
    deadband: float = 1.0                        # min change (in output units) to send
    rate_limit_hz: Optional[float] = None

    kind: str = field(init=False, default="cc")

    @property
    def value_max(self) -> int:
        return PITCH_BEND_MAX if self.kind == "pitch_bend" else 127

    @property
    def label(self) -> str:
        if self.kind == "cc":
            return f"CC {self.number}"
        if self.kind == "pitch_bend":
            return "Pitch Bend"
        return "Aftertouch"

    def _range(self) -> Tuple[float, float]:
        return self.output_range if self.output_range is not None else (0.0, float(self.value_max))

    def _update(self, raw: float, now: float) -> List[MidiEvent]:
        position = normalise(raw, *self.input_range)
        if self.invert:
            position = 1.0 - position
        if self.curve != 1.0:
            position = position ** self.curve
        self.display = position

        low, high = self._range()
        value = int(round(low + position * (high - low)))
        value = max(0, min(self.value_max, value))

        if not self._should_send(value, now, low, high):
            return []
        self.last_value = value
        self._last_sent_at = now
        return [MidiEvent(self.kind, self.channel, self.number, value, self.name)]

    def _should_send(self, value: int, now: float, low: float, high: float) -> bool:
        if self.last_value is None:
            return True
        if value == self.last_value:
            return False
        # Always let the endpoints through, otherwise a large deadband would keep
        # a filter from ever fully closing.
        at_endpoint = value in (int(round(low)), int(round(high)), 0, self.value_max)
        if not at_endpoint and abs(value - self.last_value) < self.deadband:
            return False
        if self.rate_limit_hz and not at_endpoint:
            if (now - self._last_sent_at) < (1.0 / self.rate_limit_hz):
                return False
        return True


@dataclass
class GateMapping(BaseMapping):
    """Pose gate -> control change on/off (sustain pedal, effect kill, ...)."""

    number: int = 64
    threshold: float = 0.6
    release: float = 0.4
    value_on: int = 127
    value_off: int = 0

    kind: str = field(init=False, default="gate")

    @property
    def label(self) -> str:
        return f"CC {self.number}"

    def reset(self) -> None:
        super().reset()
        self._trigger = SchmittTrigger(self.threshold, self.release)

    def _update(self, raw: float, now: float) -> List[MidiEvent]:
        self.display = clamp01(raw)
        state = self._trigger.update(raw)
        if state == self.active and self.last_value is not None:
            return []
        self.active = state
        self.last_value = self.value_on if state else self.value_off
        self._last_sent_at = now
        return [MidiEvent("cc", self.channel, self.number, self.last_value, self.name)]

    def panic(self) -> List[MidiEvent]:
        if not self.active:
            return []
        self.active = False
        self._trigger.reset(False)
        return [MidiEvent("cc", self.channel, self.number, self.value_off, self.name)]


#: Scales a hand position can be quantised to. Whatever the hand does, the note
#: that comes out belongs to the scale — which is what makes waving your arm
#: around sound like playing rather than like a siren.
SCALES = {
    "pentatonic_minor": (0, 3, 5, 7, 10),
    "pentatonic_major": (0, 2, 4, 7, 9),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "chromatic": tuple(range(12)),
}


def scale_notes(root: int, scale: str, octaves: float) -> List[int]:
    """Every MIDI note of ``scale`` from ``root`` upwards, spanning ``octaves``."""
    try:
        degrees = SCALES[scale]
    except KeyError:
        raise ConfigError(
            f"unknown scale {scale!r}; expected one of {', '.join(sorted(SCALES))}"
        ) from None
    count = max(int(round(len(degrees) * octaves)), 1)
    notes = []
    for step in range(count + 1):
        octave, degree = divmod(step, len(degrees))
        note = root + octave * 12 + degrees[degree]
        if 0 <= note <= 127:
            notes.append(note)
    return notes or [max(0, min(127, root))]


@dataclass
class ScaleMapping(BaseMapping):
    """Hand position -> a note from a musical scale: a playable lead voice.

    This is the mapping that turns the controller from a bank of knobs into an
    instrument. The driving feature picks a rung on a ladder of scale notes, a
    gate feature decides when it should sound at all, and the note is re-struck
    only when the hand crosses properly into the next rung.

    That last part is the whole trick. Quantising on the nearest note alone
    makes a hand resting on a boundary stutter between two pitches many times a
    second; ``hysteresis`` requires the hand to move past the halfway point by a
    margin before the note changes.
    """

    root: int = 57                          # A3
    scale: str = "pentatonic_minor"
    octaves: float = 2.0
    velocity: int = 100
    velocity_feature: Optional[str] = None
    gate_feature: Optional[str] = None      # defaults to "the hand is tracked"
    threshold: float = 0.5
    release: float = 0.3
    hysteresis: float = 0.2                 # extra travel needed, in note steps
    input_range: Tuple[float, float] = (0.0, 1.0)
    invert: bool = False

    kind: str = field(init=False, default="scale")

    def __post_init__(self) -> None:
        self.notes = scale_notes(self.root, self.scale, self.octaves)
        if self.gate_feature is None and "." in self.feature:
            self.gate_feature = f"{self.feature.split('.', 1)[0]}.present"
        super().__post_init__()

    @property
    def label(self) -> str:
        return f"{self.scale.replace('_', ' ')} {self.notes[0]}-{self.notes[-1]}"

    def reset(self) -> None:
        super().reset()
        self._trigger = SchmittTrigger(self.threshold, self.release)
        self._index: Optional[int] = None
        self._velocity_source = 1.0
        self.sounding_note: Optional[int] = None

    def _position(self, raw: float) -> float:
        """Feature value -> continuous position along the ladder of notes."""
        amount = normalise(raw, *self.input_range)
        if self.invert:
            amount = 1.0 - amount
        return amount * (len(self.notes) - 1)

    def _quantise(self, position: float) -> int:
        """Nearest note index, holding the current one inside the dead band."""
        if self._index is None:
            return int(round(position))
        if abs(position - self._index) <= 0.5 + self.hysteresis:
            return self._index
        return int(round(position))

    def update(self, features: TMapping[str, float], now: float) -> List[MidiEvent]:
        if self.velocity_feature is not None:
            source = features.get(self.velocity_feature)
            if source is not None:
                self._velocity_source = clamp01(float(source))

        raw = features.get(self.feature)
        self.tracked = raw is not None
        if not self.enabled:
            return []

        gate_value = features.get(self.gate_feature) if self.gate_feature else None
        if gate_value is None:
            gate_value = 1.0 if raw is not None else 0.0
        open_gate = self._trigger.update(float(gate_value))

        if not open_gate:
            self.active = False
            return self._release()

        if raw is None:                       # gate open but no fresh position
            return []

        position = self._position(float(raw))
        self.display = position / max(len(self.notes) - 1, 1)
        index = self._quantise(position)
        note = self.notes[max(0, min(index, len(self.notes) - 1))]

        if note == self.sounding_note:
            return []

        events = self._release()
        self._index = index
        self.sounding_note = note
        self.active = True
        self.last_value = note
        self._last_sent_at = now
        velocity = self.velocity
        if self.velocity_feature is not None:
            velocity = max(1, int(round(self.velocity * self._velocity_source)))
        events.append(MidiEvent("note_on", self.channel, note, velocity, self.name))
        return events

    def _release(self) -> List[MidiEvent]:
        if self.sounding_note is None:
            return []
        note, self.sounding_note = self.sounding_note, None
        return [MidiEvent("note_off", self.channel, note, 0, self.name)]

    def panic(self) -> List[MidiEvent]:
        self.active = False
        self._trigger.reset(False)
        self._index = None
        return self._release()


@dataclass
class NoteMapping(BaseMapping):
    """Pose gate -> note on / note off (trigger a drum pad or a sample)."""

    number: int = 36
    velocity: int = 100
    threshold: float = 0.6
    release: float = 0.4
    velocity_feature: Optional[str] = None  # optional 0..1 feature scaling velocity
    retrigger: bool = False                 # re-fire while held (machine-gun mode)

    kind: str = field(init=False, default="note")

    @property
    def label(self) -> str:
        return f"Note {self.number}"

    def reset(self) -> None:
        super().reset()
        self._trigger = SchmittTrigger(self.threshold, self.release)
        self._velocity_source: float = 1.0

    def update(self, features: TMapping[str, float], now: float) -> List[MidiEvent]:
        if self.velocity_feature is not None:
            source = features.get(self.velocity_feature)
            if source is not None:
                self._velocity_source = clamp01(float(source))
        return super().update(features, now)

    def _update(self, raw: float, now: float) -> List[MidiEvent]:
        self.display = clamp01(raw)
        state = self._trigger.update(raw)
        if state and (not self.active or self.retrigger):
            velocity = self.velocity
            if self.velocity_feature is not None:
                velocity = max(1, int(round(self.velocity * self._velocity_source)))
            self.active = True
            self.last_value = velocity
            self._last_sent_at = now
            events = [MidiEvent("note_on", self.channel, self.number, velocity, self.name)]
            if self.retrigger:
                events.insert(0, MidiEvent("note_off", self.channel, self.number, 0, self.name))
            return events
        if not state and self.active:
            self.active = False
            self.last_value = 0
            return [MidiEvent("note_off", self.channel, self.number, 0, self.name)]
        return []

    def panic(self) -> List[MidiEvent]:
        if not self.active:
            return []
        self.active = False
        self._trigger.reset(False)
        return [MidiEvent("note_off", self.channel, self.number, 0, self.name)]


# --------------------------------------------------------------------------
# Mapping construction from JSON
# --------------------------------------------------------------------------
_COMMON_KEYS = ("name", "feature", "type", "channel", "enabled", "smoothing", "comment")
_CONTINUOUS_KEYS = _COMMON_KEYS + (
    "cc", "number", "input_range", "output_range", "invert", "curve",
    "deadband", "rate_limit_hz",
)
_GATE_KEYS = _COMMON_KEYS + ("cc", "number", "threshold", "release", "value_on", "value_off")
_NOTE_KEYS = _COMMON_KEYS + (
    "note", "number", "velocity", "threshold", "release", "velocity_feature", "retrigger",
)
_SCALE_KEYS = _COMMON_KEYS + (
    "root", "scale", "octaves", "velocity", "velocity_feature", "gate_feature",
    "threshold", "release", "hysteresis", "input_range", "invert",
)

#: Accepted ``type`` spellings -> canonical kind.
TYPE_ALIASES = {
    "cc": "cc",
    "control_change": "cc",
    "gate": "gate",
    "toggle": "gate",
    "switch": "gate",
    "note": "note",
    "trigger": "note",
    "pitch_bend": "pitch_bend",
    "pitchbend": "pitch_bend",
    "bend": "pitch_bend",
    "aftertouch": "aftertouch",
    "pressure": "aftertouch",
    "channel_pressure": "aftertouch",
    "scale": "scale",
    "pitch": "scale",
    "lead": "scale",
}


def _build_mapping(entry: TMapping[str, Any], index: int, default_channel: int) -> BaseMapping:
    if not isinstance(entry, dict):
        raise ConfigError(f"mappings[{index}] must be an object, got {type(entry).__name__}")

    name = str(entry.get("name") or f"mapping {index}")
    context = f"mapping {name!r}"

    raw_type = str(entry.get("type", "cc")).lower()
    if raw_type not in TYPE_ALIASES:
        raise ConfigError(
            f"{context}: unknown type {raw_type!r}; expected one of "
            + ", ".join(sorted(set(TYPE_ALIASES.values())))
        )
    kind = TYPE_ALIASES[raw_type]

    feature = entry.get("feature")
    if not feature or not isinstance(feature, str):
        raise ConfigError(f"{context}: 'feature' is required and must be a string")

    channel = _get_int(entry, "channel", default_channel, context) if entry.get("channel") is not None else default_channel
    if not 1 <= channel <= 16:
        raise ConfigError(f"{context}: 'channel' must be between 1 and 16, got {channel}")

    enabled = bool(entry.get("enabled", True))
    smoothing = entry.get("smoothing")
    if smoothing is not None:
        smoothing = _get_number(entry, "smoothing", 5.0, context)

    common = dict(name=name, feature=feature, channel=channel, enabled=enabled, smoothing=smoothing)

    if kind in ("cc", "pitch_bend", "aftertouch"):
        _check_keys(entry, _CONTINUOUS_KEYS, context)
        number = 0
        if kind == "cc":
            if "cc" not in entry and "number" not in entry:
                raise ConfigError(f"{context}: a cc mapping needs a 'cc' number")
            number = _get_int(entry, "cc", entry.get("number"), context)
            if not 0 <= number <= 127:
                raise ConfigError(f"{context}: 'cc' must be between 0 and 127, got {number}")
        mapping = ContinuousMapping(
            number=number,
            input_range=_get_range(entry, "input_range", (0.0, 1.0), context),
            output_range=(
                _get_range(entry, "output_range", (0.0, 1.0), context)
                if "output_range" in entry else None
            ),
            invert=bool(entry.get("invert", False)),
            curve=_get_number(entry, "curve", 1.0, context),
            deadband=_get_number(entry, "deadband", 1.0 if kind != "pitch_bend" else 64.0, context),
            rate_limit_hz=(
                _get_number(entry, "rate_limit_hz", 0.0, context)
                if entry.get("rate_limit_hz") else None
            ),
            **common,
        )
        # ``kind`` is init=False so the three continuous flavours share one class.
        object.__setattr__(mapping, "kind", kind)
        if mapping.curve <= 0:
            raise ConfigError(f"{context}: 'curve' must be positive, got {mapping.curve}")
        return mapping

    if kind == "scale":
        _check_keys(entry, _SCALE_KEYS, context)
        root = _get_int(entry, "root", 57, context)
        if not 0 <= root <= 127:
            raise ConfigError(f"{context}: 'root' must be between 0 and 127, got {root}")
        octaves = _get_number(entry, "octaves", 2.0, context)
        if octaves <= 0:
            raise ConfigError(f"{context}: 'octaves' must be positive, got {octaves}")
        threshold = _get_number(entry, "threshold", 0.5, context)
        release = _get_number(entry, "release", 0.3, context)
        if release > threshold:
            raise ConfigError(
                f"{context}: 'release' ({release}) must not be above 'threshold' ({threshold})"
            )
        scale = str(entry.get("scale", "pentatonic_minor"))
        if scale not in SCALES:
            raise ConfigError(
                f"{context}: unknown scale {scale!r}; expected one of {', '.join(sorted(SCALES))}"
            )
        return ScaleMapping(
            root=root,
            scale=scale,
            octaves=octaves,
            velocity=_get_int(entry, "velocity", 100, context),
            velocity_feature=entry.get("velocity_feature"),
            gate_feature=entry.get("gate_feature"),
            threshold=threshold,
            release=release,
            hysteresis=_get_number(entry, "hysteresis", 0.2, context),
            input_range=_get_range(entry, "input_range", (0.0, 1.0), context),
            invert=bool(entry.get("invert", False)),
            **common,
        )

    # Both remaining types are gates, and both need a valid hysteresis band.
    # Validated before construction: the Schmitt trigger rejects an inverted band
    # with an exception aimed at a programmer, not at someone editing JSON.
    threshold = _get_number(entry, "threshold", 0.6, context)
    release = _get_number(entry, "release", 0.4, context)
    if release > threshold:
        raise ConfigError(
            f"{context}: 'release' ({release}) must not be above "
            f"'threshold' ({threshold}); the gap between them is what debounces the gesture"
        )

    if kind == "gate":
        _check_keys(entry, _GATE_KEYS, context)
        number = _get_int(entry, "cc", entry.get("number", 64), context)
        if not 0 <= number <= 127:
            raise ConfigError(f"{context}: 'cc' must be between 0 and 127, got {number}")
        return GateMapping(
            number=number,
            threshold=threshold,
            release=release,
            value_on=_get_int(entry, "value_on", 127, context),
            value_off=_get_int(entry, "value_off", 0, context),
            **common,
        )

    _check_keys(entry, _NOTE_KEYS, context)
    number = _get_int(entry, "note", entry.get("number", 36), context)
    if not 0 <= number <= 127:
        raise ConfigError(f"{context}: 'note' must be between 0 and 127, got {number}")
    return NoteMapping(
        number=number,
        velocity=_get_int(entry, "velocity", 100, context),
        threshold=threshold,
        release=release,
        velocity_feature=entry.get("velocity_feature"),
        retrigger=bool(entry.get("retrigger", False)),
        **common,
    )


# --------------------------------------------------------------------------
# Whole-file configuration
# --------------------------------------------------------------------------
@dataclass
class MidiSettings:
    """The ``"midi"`` block of the config file."""

    channel: int = 1
    port_name: str = DEFAULT_PORT_NAME
    port: Optional[str] = None  # existing port to open instead of a virtual one


@dataclass
class SmoothingSettings:
    """The ``"smoothing"`` block of the config file."""

    window: float = 5.0
    reset_after: Optional[int] = 30


@dataclass
class SynthSettings:
    """The ``"synth"`` block: defaults for the built-in monitor synth.

    A patch knows whether it needs a drone. One built from control changes does
    (there must be something for the filter to shape); a playable one does not,
    because it makes its own notes. Command-line flags still win.
    """

    drone: Optional[int] = 45
    gain: float = 0.28


@dataclass
class ControllerConfig:
    """A parsed mapping file."""

    midi: MidiSettings = field(default_factory=MidiSettings)
    smoothing: SmoothingSettings = field(default_factory=SmoothingSettings)
    calibration: FeatureCalibration = field(default_factory=FeatureCalibration)
    synth: SynthSettings = field(default_factory=SynthSettings)
    mappings: List[BaseMapping] = field(default_factory=list)
    path: Optional[Path] = None

    def enabled_mappings(self) -> List[BaseMapping]:
        return [mapping for mapping in self.mappings if mapping.enabled]

    def describe(self) -> str:
        lines = [f"{len(self.enabled_mappings())} active mapping(s), MIDI channel {self.midi.channel}"]
        for mapping in self.mappings:
            flag = " " if mapping.enabled else "-"
            lines.append(f"  {flag} {mapping.name:<22} {mapping.feature:<28} -> {mapping.label}")
        return "\n".join(lines)


def parse_config(data: TMapping[str, Any], path: Optional[Path] = None) -> ControllerConfig:
    """Validate a decoded JSON document and build a :class:`ControllerConfig`."""
    if not isinstance(data, dict):
        raise ConfigError("the mapping file must contain a JSON object")
    _check_keys(
        data,
        ("version", "name", "description", "midi", "smoothing", "calibration", "synth", "mappings"),
        "config",
    )

    version = data.get("version", CONFIG_VERSION)
    if version != CONFIG_VERSION:
        raise ConfigError(f"unsupported config version {version!r}; this build reads version {CONFIG_VERSION}")

    midi_block = data.get("midi", {}) or {}
    _check_keys(midi_block, ("channel", "port_name", "port"), "config.midi")
    midi = MidiSettings(
        channel=_get_int(midi_block, "channel", 1, "config.midi"),
        port_name=str(midi_block.get("port_name", DEFAULT_PORT_NAME)),
        port=midi_block.get("port"),
    )
    if not 1 <= midi.channel <= 16:
        raise ConfigError(f"config.midi: 'channel' must be between 1 and 16, got {midi.channel}")

    smoothing_block = data.get("smoothing", {}) or {}
    _check_keys(smoothing_block, ("window", "reset_after"), "config.smoothing")
    smoothing = SmoothingSettings(
        window=_get_number(smoothing_block, "window", 5.0, "config.smoothing"),
        reset_after=(
            None if smoothing_block.get("reset_after") is None
            else _get_int(smoothing_block, "reset_after", 30, "config.smoothing")
        ),
    )

    try:
        calibration = FeatureCalibration.from_dict(data.get("calibration"))
    except ValueError as exc:
        raise ConfigError(f"config.calibration: {exc}") from exc

    synth_block = data.get("synth", {}) or {}
    _check_keys(synth_block, ("drone", "gain"), "config.synth")
    drone = _get_int(synth_block, "drone", 45, "config.synth")
    synth = SynthSettings(
        drone=None if drone < 0 else drone,
        gain=_get_number(synth_block, "gain", 0.28, "config.synth"),
    )

    raw_mappings = data.get("mappings")
    if not isinstance(raw_mappings, list) or not raw_mappings:
        raise ConfigError("config: 'mappings' must be a non-empty list")

    mappings = [_build_mapping(entry, i, midi.channel) for i, entry in enumerate(raw_mappings)]

    known = set(feature_names())
    for mapping in mappings:
        for feature in filter(None, (mapping.feature, getattr(mapping, "velocity_feature", None))):
            if feature not in known:
                log.warning(
                    "mapping %r uses unknown feature %r (run --list-features to see them all)",
                    mapping.name,
                    feature,
                )
    return ControllerConfig(midi=midi, smoothing=smoothing, calibration=calibration,
                            synth=synth, mappings=mappings,
                            path=Path(path) if path else None)


def load_config(path) -> ControllerConfig:
    """Read and validate a mapping file."""
    config_path = Path(path).expanduser()
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read mapping file {config_path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{config_path} is not valid JSON: {exc}") from exc
    return parse_config(data, config_path)


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------
def build_smoother(config: ControllerConfig) -> EMASmoother:
    """Create an :class:`EMASmoother` honouring per-mapping ``smoothing`` overrides."""
    smoother = EMASmoother(
        window=config.smoothing.window, reset_after=config.smoothing.reset_after
    )
    for mapping in config.mappings:
        if mapping.smoothing is not None:
            smoother.set_window(mapping.feature, mapping.smoothing)
    return smoother


class MidiMapper:
    """Turn a frame of features into the MIDI messages that should be sent."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        # Mapping objects carry per-run state (last value sent, gate latches).
        # A fresh mapper must start from a clean slate, otherwise reloading a
        # config that reuses the same objects would silently swallow the first
        # values because they "have not changed".
        self.reset()

    @property
    def mappings(self) -> List[BaseMapping]:
        return self.config.mappings

    def update(
        self, features: TMapping[str, float], now: Optional[float] = None
    ) -> List[MidiEvent]:
        """Process one frame of (smoothed) features."""
        timestamp = time.monotonic() if now is None else now
        events: List[MidiEvent] = []
        for mapping in self.config.mappings:
            events.extend(mapping.update(features, timestamp))
        return events

    def panic(self) -> List[MidiEvent]:
        """All notes off, all gates released — sent on exit and on ``p``."""
        events: List[MidiEvent] = []
        for mapping in self.config.mappings:
            events.extend(mapping.panic())
        channels = sorted({mapping.channel for mapping in self.config.mappings} or {1})
        for channel in channels:
            events.append(MidiEvent("cc", channel, 123, 0, "all notes off"))
        return events

    def reset(self) -> None:
        for mapping in self.config.mappings:
            mapping.reset()

    def states(self) -> List[MappingState]:
        """Per-mapping snapshot for the visual overlay."""
        return [mapping.state() for mapping in self.config.mappings if mapping.enabled]


class ConfigWatcher:
    """Reload the mapping file when it changes on disk (hot-reload).

    Polling the mtime is enough here — the check runs once per frame at 30 fps
    and costs one ``stat`` — and it avoids a dependency on ``watchdog``.  A file
    that fails to parse is reported once and the running configuration is kept,
    so a typo mid-set does not silence the instrument.
    """

    def __init__(self, path, min_interval: float = 0.5) -> None:
        self.path = Path(path).expanduser()
        self.min_interval = min_interval
        self._signature = self._stat()
        self._last_check = 0.0
        self._last_error: Optional[str] = None

    def _stat(self) -> Optional[Tuple[float, int]]:
        try:
            info = self.path.stat()
        except OSError:
            return None
        return (info.st_mtime, info.st_size)

    def poll(self, now: Optional[float] = None) -> Optional[ControllerConfig]:
        """Return a freshly parsed config if the file changed, else ``None``."""
        timestamp = time.monotonic() if now is None else now
        if timestamp - self._last_check < self.min_interval:
            return None
        self._last_check = timestamp

        signature = self._stat()
        if signature is None or signature == self._signature:
            return None
        self._signature = signature

        try:
            config = load_config(self.path)
        except ConfigError as exc:
            message = str(exc)
            if message != self._last_error:
                log.error("keeping previous mapping, new file is invalid: %s", message)
                self._last_error = message
            return None
        self._last_error = None
        log.info("reloaded mapping file %s", self.path)
        return config
