"""A small built-in synthesiser, so the controller makes sound on its own.

The controller is a MIDI *control surface*: it sends control changes, and a
control change only shapes a note that something else is already playing. That
makes the first run confusing — everything works, nothing is audible.

This module removes the setup step. It is an ordinary MIDI receiver, fed the
very same :class:`~src.midi_output.MidiEvent` objects that go out of the port,
so what you hear is exactly what a DAW would receive. Signal path:

    3 detuned saw oscillators  ->  resonant low-pass  ->  delay  ->  soft clip

A **drone** is held by default, because that is what makes a filter sweep
audible the moment you raise your hand; note-on messages add voices on top.

It is deliberately small (no dependency beyond NumPy and ``sounddevice``, which
MediaPipe already installs) and it is not the point of the project — it is a
monitor, the way a metronome is a monitor. Use a real instrument for real work.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from .midi_output import MidiEvent

log = logging.getLogger(__name__)

# Which controllers the synth listens to. These match the shipped presets and
# the General MIDI conventions they follow.
CC_MODULATION = 1
CC_VOLUME = 7
CC_PAN = 10
CC_RESONANCE = 71
CC_CUTOFF = 74
CC_REVERB = 91
CC_SUSTAIN = 64
CC_ALL_NOTES_OFF = 123

#: Filter cutoff range in Hz, swept exponentially so the sweep sounds even.
CUTOFF_RANGE = (80.0, 9000.0)
#: Filter resonance, from gentle to nearly self-oscillating.
RESONANCE_RANGE = (0.7, 6.0)
#: Pitch bend depth in semitones, matching a synth's usual default.
BEND_SEMITONES = 2.0
MAX_VOICES = 8


def note_to_hz(note: float) -> float:
    """MIDI note number -> frequency in Hz (A4 = note 69 = 440 Hz)."""
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


def _poly_blep(phase: np.ndarray, increment: float) -> np.ndarray:
    """Anti-aliasing correction for a naive sawtooth.

    A raw saw has a hard discontinuity every cycle, which folds high harmonics
    back down as inharmonic whistling — very audible once the filter opens.
    PolyBLEP rounds the corner over one sample either side of the wrap.
    """
    correction = np.zeros_like(phase)
    if increment <= 0.0:
        return correction

    rising = phase < increment
    if rising.any():
        t = phase[rising] / increment
        correction[rising] = t + t - t * t - 1.0

    falling = phase > (1.0 - increment)
    if falling.any():
        t = (phase[falling] - 1.0) / increment
        correction[falling] = t * t + t + t + 1.0
    return correction


@dataclass
class Voice:
    """One sounding note: three saws detuned against each other."""

    note: int
    velocity: float = 1.0
    held: bool = True
    sustained: bool = False       # held by the pedal after the gesture released
    level: float = 0.0            # envelope state
    phases: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.33, 0.66]))

    #: Detune in semitones for the three oscillators; the spread is what makes
    #: a single note sound wide rather than thin.
    DETUNE = (-0.11, 0.0, 0.13)

    def render(self, frames: int, samplerate: float, bend: float) -> np.ndarray:
        """Render ``frames`` samples of this voice's raw oscillator sum."""
        output = np.zeros(frames, dtype=np.float64)
        base = note_to_hz(self.note) * bend
        for index, detune in enumerate(self.DETUNE):
            frequency = base * (2.0 ** (detune / 12.0))
            increment = frequency / samplerate
            if not 0.0 < increment < 0.5:      # skip anything past Nyquist
                continue
            phase = (self.phases[index] + increment * np.arange(1, frames + 1)) % 1.0
            saw = 2.0 * phase - 1.0 - _poly_blep(phase, increment)
            output += saw
            self.phases[index] = phase[-1] if frames else self.phases[index]
        return output / len(self.DETUNE)

    @property
    def sounding(self) -> bool:
        return self.held or self.sustained or self.level > 1e-4


class StateVariableFilter:
    """Chamberlin state-variable low-pass — the classic cheap resonant filter.

    Runs per sample because the cutoff moves continuously; at 256 frames a
    block that is a few hundred loop iterations, which Python handles inside the
    audio deadline with room to spare.
    """

    def __init__(self) -> None:
        self.low = 0.0
        self.band = 0.0

    def process(
        self, signal: np.ndarray, cutoff: np.ndarray, resonance: float, samplerate: float
    ) -> np.ndarray:
        output = np.empty_like(signal)
        low, band = self.low, self.band
        damping = 1.0 / max(resonance, 0.5)
        nyquist_limit = samplerate / 3.0     # keep the topology stable

        for index in range(signal.shape[0]):
            frequency = min(float(cutoff[index]), nyquist_limit)
            f = 2.0 * math.sin(math.pi * frequency / samplerate)
            high = signal[index] - low - damping * band
            band += f * high
            low += f * band
            output[index] = low

        # A denormal or a NaN here would poison every later block.
        if not (math.isfinite(low) and math.isfinite(band)):
            low = band = 0.0
        self.low, self.band = low, band
        return output


class Delay:
    """One short feedback delay, standing in for a reverb send."""

    def __init__(self, samplerate: float, seconds: float = 0.19, feedback: float = 0.38) -> None:
        self.buffer = np.zeros(max(int(samplerate * seconds), 1), dtype=np.float64)
        self.index = 0
        self.feedback = feedback

    def process(self, signal: np.ndarray, mix: float) -> np.ndarray:
        if mix <= 0.0:
            # Still advance the line, so switching the send back on is not abrupt.
            for sample in signal:
                self.buffer[self.index] = self.buffer[self.index] * self.feedback
                self.index = (self.index + 1) % self.buffer.shape[0]
            return signal

        output = np.empty_like(signal)
        buffer, size = self.buffer, self.buffer.shape[0]
        index, feedback = self.index, self.feedback
        for position in range(signal.shape[0]):
            delayed = buffer[index]
            buffer[index] = signal[position] + delayed * feedback
            index = (index + 1) % size
            output[position] = signal[position] + delayed * mix
        self.index = index
        return output


class Smoothed:
    """One-pole smoothing for a control value, so steps do not click.

    ``time_constant`` is in seconds: how long the value takes to cover most of
    the distance to a new target. Cutoff needs a short one to stay responsive,
    master volume a longer one to stay quiet.
    """

    def __init__(self, value: float, time_constant: float = 0.02) -> None:
        self.value = float(value)
        self.target = float(value)
        self.time_constant = time_constant

    def block(self, frames: int, samplerate: float) -> np.ndarray:
        """Glide towards the target across one block, returning every step."""
        if frames <= 0:
            return np.empty(0, dtype=np.float64)
        alpha = 1.0 - math.exp(-frames / max(self.time_constant * samplerate, 1.0))
        end = self.value + (self.target - self.value) * alpha
        ramp = np.linspace(self.value, end, frames)
        self.value = float(end)
        return ramp

    def step(self, frames: int, samplerate: float) -> float:
        """Advance one block and return just the final value."""
        block = self.block(frames, samplerate)
        return float(block[-1]) if block.size else self.value


class GestureSynth:
    """A MIDI-driven monitor synth: feed it events, it makes noise.

    Args:
        samplerate: audio sample rate.
        blocksize: frames per callback. 256 at 44.1 kHz is ~6 ms of audio
            latency, which keeps the whole gesture-to-ear path well under the
            point where playing stops feeling direct.
        drone_note: MIDI note held continuously so filter moves are audible
            with no keyboard attached. ``None`` waits for note-on messages.
        gain: master output level before the soft clipper.
        device: sounddevice output device, or ``None`` for the system default.
    """

    def __init__(
        self,
        samplerate: int = 44100,
        blocksize: int = 256,
        drone_note: Optional[int] = 45,
        gain: float = 0.28,
        device: Optional[object] = None,
    ) -> None:
        self.samplerate = float(samplerate)
        self.blocksize = int(blocksize)
        self.gain = float(gain)
        self.device = device

        self._events: "queue.SimpleQueue[MidiEvent]" = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._stream = None
        self.underruns = 0

        self.voices: Dict[int, Voice] = {}
        self.drone_note = drone_note
        if drone_note is not None:
            self.voices[drone_note] = Voice(note=drone_note, velocity=0.9)

        self.filter = StateVariableFilter()
        self.delay = Delay(self.samplerate)
        self.sustain = False
        self.bend = 1.0
        self.modulation = 0.0
        self._lfo_phase = 0.0

        # Start half open: the first hand movement is then audible in either
        # direction, rather than only when it happens to open the filter.
        self.cutoff = Smoothed(_exp_scale(0.5, *CUTOFF_RANGE), time_constant=0.012)
        self.resonance = Smoothed(1.4, time_constant=0.03)
        self.volume = Smoothed(0.8, time_constant=0.05)
        self.pan = Smoothed(0.5, time_constant=0.05)
        self.reverb = Smoothed(0.18, time_constant=0.08)

    # -- MIDI input --------------------------------------------------------
    def handle(self, event: MidiEvent) -> None:
        """Queue one event; it is applied at the start of the next block."""
        self._events.put(event)

    def handle_all(self, events: Iterable[MidiEvent]) -> None:
        for event in events:
            self._events.put(event)

    def _drain(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            self._apply(event)

    def _apply(self, event: MidiEvent) -> None:
        if event.kind == "cc":
            self._apply_cc(event.number, event.value)
        elif event.kind == "note_on" and event.value > 0:
            self._note_on(event.number, event.value / 127.0)
        elif event.kind in ("note_off",) or (event.kind == "note_on" and event.value == 0):
            self._note_off(event.number)
        elif event.kind == "pitch_bend":
            semitones = (event.value - 8192) / 8192.0 * BEND_SEMITONES
            self.bend = 2.0 ** (semitones / 12.0)
        elif event.kind == "aftertouch":
            self.volume.target = 0.25 + 0.75 * (event.value / 127.0)

    def _apply_cc(self, number: int, value: int) -> None:
        amount = value / 127.0
        if number == CC_CUTOFF:
            self.cutoff.target = _exp_scale(amount, *CUTOFF_RANGE)
        elif number == CC_RESONANCE:
            self.resonance.target = RESONANCE_RANGE[0] + amount * (
                RESONANCE_RANGE[1] - RESONANCE_RANGE[0]
            )
        elif number == CC_VOLUME:
            self.volume.target = amount
        elif number == CC_PAN:
            self.pan.target = amount
        elif number == CC_REVERB:
            self.reverb.target = amount * 0.6
        elif number == CC_MODULATION:
            self.modulation = amount
        elif number == CC_SUSTAIN:
            self.sustain = value >= 64
            if not self.sustain:
                for voice in list(self.voices.values()):
                    if voice.sustained and not voice.held:
                        voice.sustained = False
        elif number == CC_ALL_NOTES_OFF:
            self._all_notes_off()

    def _note_on(self, note: int, velocity: float) -> None:
        with self._lock:
            voice = self.voices.get(note)
            if voice is None:
                if len(self.voices) >= MAX_VOICES:
                    self._steal_voice()
                voice = Voice(note=note, velocity=max(velocity, 0.05))
                self.voices[note] = voice
            voice.held = True
            voice.sustained = self.sustain
            voice.velocity = max(velocity, 0.05)

    def _note_off(self, note: int) -> None:
        with self._lock:
            voice = self.voices.get(note)
            if voice is None or note == self.drone_note:
                return
            voice.held = False
            voice.sustained = self.sustain

    def _all_notes_off(self) -> None:
        with self._lock:
            self.sustain = False
            for note, voice in self.voices.items():
                if note != self.drone_note:
                    voice.held = False
                    voice.sustained = False

    def _steal_voice(self) -> None:
        """Drop the quietest non-drone voice to make room for a new one."""
        candidates = [
            (voice.level, note) for note, voice in self.voices.items() if note != self.drone_note
        ]
        if candidates:
            self.voices.pop(min(candidates)[1], None)

    # -- audio -------------------------------------------------------------
    def render(self, frames: int) -> np.ndarray:
        """Render one block as ``(frames, 2)`` float32. Pure — safe to test."""
        self._drain()
        if frames <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        mix = np.zeros(frames, dtype=np.float64)
        bend = self.bend * self._vibrato(frames)

        with self._lock:
            for note in list(self.voices):
                voice = self.voices[note]
                envelope = self._envelope(voice, frames)
                if envelope is None:
                    if note != self.drone_note:
                        self.voices.pop(note, None)
                    continue
                mix += voice.render(frames, self.samplerate, bend) * envelope

        cutoff = self.cutoff.block(frames, self.samplerate)
        resonance = self.resonance.step(frames, self.samplerate)
        mix = self.filter.process(mix, cutoff, resonance, self.samplerate)
        mix = self.delay.process(mix, self.reverb.step(frames, self.samplerate))

        volume = self.volume.block(frames, self.samplerate)
        mix = np.tanh(mix * volume * self.gain * 1.6)

        angle = self.pan.step(frames, self.samplerate) * (math.pi / 2.0)
        stereo = np.empty((frames, 2), dtype=np.float64)
        stereo[:, 0] = mix * math.cos(angle)
        stereo[:, 1] = mix * math.sin(angle)
        return np.nan_to_num(stereo, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _vibrato(self, frames: int) -> float:
        """A 5 Hz wobble scaled by the modulation wheel."""
        if self.modulation <= 0.0:
            self._lfo_phase = (self._lfo_phase + 5.0 * frames / self.samplerate) % 1.0
            return 1.0
        self._lfo_phase = (self._lfo_phase + 5.0 * frames / self.samplerate) % 1.0
        semitones = math.sin(2.0 * math.pi * self._lfo_phase) * 0.4 * self.modulation
        return 2.0 ** (semitones / 12.0)

    def _envelope(self, voice: Voice, frames: int) -> Optional[np.ndarray]:
        """Block-rate AR envelope; ``None`` once the voice has faded out."""
        target = voice.velocity if (voice.held or voice.sustained) else 0.0
        # Time constants, not durations: a release of 0.18 s is fully quiet in
        # about a second. Short enough for a drum trigger to feel tight, long
        # enough that letting go of a gesture is not a click.
        seconds = 0.008 if target > voice.level else 0.18
        alpha = 1.0 - math.exp(-frames / max(seconds * self.samplerate, 1.0))
        end = voice.level + (target - voice.level) * alpha
        if end < 1e-4 and target == 0.0:
            voice.level = 0.0
            return None
        envelope = np.linspace(voice.level, end, frames)
        voice.level = float(end)
        return envelope

    # -- stream ------------------------------------------------------------
    def start(self) -> "GestureSynth":
        """Open the audio output. Raises RuntimeError with advice on failure."""
        try:
            import sounddevice as sd
        except OSError as exc:   # PortAudio missing (common on bare Linux)
            raise RuntimeError(
                f"audio output unavailable: {exc}. On Linux install libportaudio2; "
                "otherwise run without --synth and use a DAW."
            ) from exc

        def callback(outdata, frames, _time, status):
            if status:
                self.underruns += 1
            outdata[:] = self.render(frames)

        try:
            self._stream = sd.OutputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=2,
                dtype="float32",
                device=self.device,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(
                f"could not open the audio output ({exc}). Check the output device "
                "in System Settings > Sound, or run without --synth."
            ) from exc
        log.info(
            "built-in synth running (%.0f Hz, %d-frame blocks, ~%.1f ms audio latency)",
            self.samplerate, self.blocksize, self.blocksize / self.samplerate * 1000.0,
        )
        return self

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        if self.underruns:
            log.info("audio underruns during the session: %d", self.underruns)

    def __enter__(self) -> "GestureSynth":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()


def _exp_scale(amount: float, low: float, high: float) -> float:
    """Map ``0..1`` onto ``low..high`` geometrically (how pitch and cutoff are heard)."""
    amount = min(max(amount, 0.0), 1.0)
    return float(low * ((high / low) ** amount))
