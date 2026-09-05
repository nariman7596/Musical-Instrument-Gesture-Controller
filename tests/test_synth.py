"""The built-in monitor synth: DSP correctness and real-time behaviour.

No audio device is needed — ``render()`` is a pure function of the events fed
in, which is also what makes it safe to assert on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.midi_output import MidiEvent
from src.synth import (
    CUTOFF_RANGE,
    GestureSynth,
    StateVariableFilter,
    _exp_scale,
    note_to_hz,
)

RATE, BLOCK = 44100, 256


def render(synth: GestureSynth, blocks: int = 120) -> np.ndarray:
    return np.concatenate([synth.render(BLOCK) for _ in range(blocks)])


def fundamental(audio: np.ndarray, skip: float = 0.4) -> float:
    """Strongest frequency in the settled part of a rendered buffer."""
    mono = audio[:, 0] + audio[:, 1]
    mono = mono[int(len(mono) * skip):]
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return float(np.fft.rfftfreq(len(mono), 1 / RATE)[np.argmax(spectrum)])


def brightness(audio: np.ndarray, skip: float = 0.4) -> float:
    """Spectral centroid — how open the filter sounds."""
    mono = audio[:, 0] + audio[:, 1]
    mono = mono[int(len(mono) * skip):]
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1 / RATE)
    return float((freqs * spectrum).sum() / max(spectrum.sum(), 1e-12))


def synth(drone=45, **kwargs) -> GestureSynth:
    return GestureSynth(samplerate=RATE, blocksize=BLOCK, drone_note=drone, **kwargs)


def audible(instrument, *events, blocks=120):
    instrument.handle_all([MidiEvent("cc", 1, 7, 110), MidiEvent("cc", 1, 74, 110), *events])
    return render(instrument, blocks)


class TestOutputContract:
    def test_shape_and_dtype(self):
        block = synth().render(BLOCK)
        assert block.shape == (BLOCK, 2)
        assert block.dtype == np.float32

    def test_zero_frames_is_handled(self):
        assert synth().render(0).shape == (0, 2)

    def test_output_never_clips_or_goes_wild(self):
        audio = audible(synth(), MidiEvent("cc", 1, 71, 127))  # full resonance
        assert np.isfinite(audio).all()
        assert np.abs(audio).max() < 1.0

    def test_a_stack_of_voices_stays_bounded(self):
        instrument = synth()
        audio = audible(instrument, *[MidiEvent("note_on", 1, 48 + i * 3, 120) for i in range(8)])
        assert np.abs(audio).max() < 1.0
        assert np.isfinite(audio).all()


class TestPitch:
    def test_the_drone_sounds_at_its_note(self):
        assert fundamental(audible(synth(45))) == pytest.approx(note_to_hz(45), rel=0.03)

    def test_note_on_sounds_at_its_note(self):
        instrument = synth(drone=None)
        assert fundamental(audible(instrument, MidiEvent("note_on", 1, 60, 100))) == pytest.approx(
            note_to_hz(60), rel=0.03
        )

    def test_pitch_bend_moves_by_two_semitones(self):
        up = audible(synth(45), MidiEvent("pitch_bend", 1, value=16383))
        assert fundamental(up) == pytest.approx(note_to_hz(47), rel=0.03)

    def test_centre_bend_leaves_pitch_alone(self):
        centred = audible(synth(45), MidiEvent("pitch_bend", 1, value=8192))
        assert fundamental(centred) == pytest.approx(note_to_hz(45), rel=0.03)


class TestControls:
    def test_cutoff_changes_brightness(self):
        dark = audible(synth(), MidiEvent("cc", 1, 74, 5))
        bright = audible(synth(), MidiEvent("cc", 1, 74, 127))
        assert brightness(bright) > brightness(dark) * 3

    def test_volume_zero_fades_to_silence(self):
        instrument = synth()
        instrument.handle(MidiEvent("cc", 1, 7, 0))
        tail = render(instrument, 200)[-RATE // 10:]
        assert np.abs(tail).max() < 1e-3

    def test_pan_moves_the_signal_across_the_stereo_field(self):
        left = audible(synth(), MidiEvent("cc", 1, 10, 0))[-RATE // 4:]
        right = audible(synth(), MidiEvent("cc", 1, 10, 127))[-RATE // 4:]
        assert left[:, 0].std() > left[:, 1].std() * 5
        assert right[:, 1].std() > right[:, 0].std() * 5

    def test_exp_scale_is_geometric(self):
        assert _exp_scale(0.0, *CUTOFF_RANGE) == pytest.approx(CUTOFF_RANGE[0])
        assert _exp_scale(1.0, *CUTOFF_RANGE) == pytest.approx(CUTOFF_RANGE[1])
        middle = _exp_scale(0.5, *CUTOFF_RANGE)
        assert middle == pytest.approx(math_sqrt(CUTOFF_RANGE[0] * CUTOFF_RANGE[1]), rel=1e-6)


def math_sqrt(value: float) -> float:
    return value ** 0.5


class TestVoices:
    def test_note_off_releases_the_voice(self):
        instrument = synth(drone=None)
        audible(instrument, MidiEvent("note_on", 1, 60, 100), blocks=20)
        instrument.handle(MidiEvent("note_off", 1, 60, 0))
        tail = render(instrument, 400)[-RATE // 8:]   # ~2.3 s, past the release tail
        assert np.abs(tail).max() < 1e-3

    def test_the_sustain_pedal_holds_a_released_note(self):
        instrument = synth(drone=None)
        instrument.handle_all([MidiEvent("cc", 1, 7, 110), MidiEvent("cc", 1, 74, 110),
                               MidiEvent("cc", 1, 64, 127), MidiEvent("note_on", 1, 60, 100)])
        render(instrument, 20)
        instrument.handle(MidiEvent("note_off", 1, 60, 0))
        assert np.abs(render(instrument, 60)).max() > 1e-3   # still ringing
        instrument.handle(MidiEvent("cc", 1, 64, 0))         # pedal up
        assert np.abs(render(instrument, 400)[-RATE // 8:]).max() < 1e-3

    def test_all_notes_off_silences_played_notes_but_keeps_the_drone(self):
        instrument = synth(45)
        audible(instrument, MidiEvent("note_on", 1, 72, 110), blocks=20)
        instrument.handle(MidiEvent("cc", 1, 123, 0))
        render(instrument, 400)
        assert 45 in instrument.voices     # the drone is not a played note
        assert 72 not in instrument.voices

    def test_voice_count_is_capped(self):
        instrument = synth(45)
        instrument.handle_all([MidiEvent("note_on", 1, 40 + i, 100) for i in range(20)])
        instrument.render(BLOCK)
        assert len(instrument.voices) <= 9      # MAX_VOICES plus the drone

    def test_the_drone_ignores_note_off(self):
        instrument = synth(45)
        instrument.handle(MidiEvent("note_off", 1, 45, 0))
        assert np.abs(audible(instrument)).max() > 1e-3


class TestRealTimeBudget:
    def test_a_block_renders_well_inside_the_audio_deadline(self):
        """An overrun here is an audible click, so keep a wide margin."""
        import time

        instrument = synth(45)
        instrument.handle_all([MidiEvent("note_on", 1, 60 + i, 100) for i in range(6)])
        for _ in range(20):
            instrument.render(BLOCK)

        start = time.perf_counter()
        for _ in range(200):
            instrument.render(BLOCK)
        per_block = (time.perf_counter() - start) / 200
        assert per_block < (BLOCK / RATE) * 0.5


class TestFilterStability:
    def test_the_filter_recovers_from_a_non_finite_state(self):
        state = StateVariableFilter()
        state.low, state.band = float("nan"), float("inf")
        out = state.process(np.ones(64), np.full(64, 1000.0), 2.0, RATE)
        assert out.shape == (64,)
        assert state.low == 0.0 and state.band == 0.0

    def test_extreme_cutoff_stays_stable(self):
        state = StateVariableFilter()
        signal = np.random.default_rng(0).normal(0, 0.3, 4096)
        out = state.process(signal, np.full(4096, 20000.0), 6.0, RATE)
        assert np.isfinite(out).all()
        assert np.abs(out).max() < 50.0
