"""The standalone example must keep working — it is what people run first.

It deliberately duplicates the project in miniature, so these tests guard the
two things that make it playable at all: notes locked to a scale, and a dead
band that stops a resting hand stuttering between pitches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

one_file = pytest.importorskip("one_file_gesture_music")

RATE, BLOCK = 44100, 256


def render(synth, blocks=80):
    return np.concatenate([synth.render(BLOCK) for _ in range(blocks)])


def fundamental(audio, skip=0.4):
    mono = audio[:, 0]
    mono = mono[int(len(mono) * skip):]
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return float(np.fft.rfftfreq(len(mono), 1 / RATE)[np.argmax(spectrum)])


class TestNoteLadder:
    def test_notes_are_a_pentatonic_scale(self):
        assert one_file.NOTES == [57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81]

    def test_every_note_is_a_valid_midi_note(self):
        assert all(0 <= note <= 127 for note in one_file.NOTES)

    def test_the_ladder_ascends(self):
        assert one_file.NOTES == sorted(one_file.NOTES)


class TestSynth:
    def test_output_contract(self):
        block = one_file.Synth().render(BLOCK)
        assert block.shape == (BLOCK, 2)
        assert block.dtype == np.float32

    def test_silence_before_any_note(self):
        assert np.abs(render(one_file.Synth(), 20)).max() < 1e-6

    def test_a_note_sounds_at_its_pitch(self):
        synth = one_file.Synth()
        synth.note_on(69)              # A4 = 440 Hz
        synth.set_cutoff(6000.0)
        assert fundamental(render(synth)) == pytest.approx(440.0, rel=0.05)

    def test_note_off_fades_to_silence(self):
        synth = one_file.Synth()
        synth.note_on(69)
        render(synth, 20)
        synth.note_off()
        assert np.abs(render(synth, 200)[-RATE // 8:]).max() < 1e-3

    def test_cutoff_changes_brightness(self):
        def centroid(audio):
            mono = audio[int(len(audio) * 0.4):, 0]
            spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
            freqs = np.fft.rfftfreq(len(mono), 1 / RATE)
            return float((freqs * spectrum).sum() / max(spectrum.sum(), 1e-12))

        dark, bright = one_file.Synth(), one_file.Synth()
        for synth, cutoff in ((dark, 200.0), (bright, 7000.0)):
            synth.note_on(45)
            synth.set_cutoff(cutoff)
        assert centroid(render(bright)) > centroid(render(dark)) * 2

    def test_output_never_clips(self):
        synth = one_file.Synth()
        synth.note_on(45)
        synth.set_cutoff(9000.0)
        audio = render(synth)
        assert np.abs(audio).max() < 1.0
        assert np.isfinite(audio).all()

    def test_a_block_renders_inside_the_audio_deadline(self):
        import time

        synth = one_file.Synth()
        synth.note_on(60)
        for _ in range(10):
            synth.render(BLOCK)
        start = time.perf_counter()
        for _ in range(200):
            synth.render(BLOCK)
        assert (time.perf_counter() - start) / 200 < (BLOCK / RATE) * 0.5


class TestPlayability:
    """The note-picking logic copied from main(), which is what makes it music."""

    @staticmethod
    def pick(position, index):
        spot = min(max((position - 0.12) / 0.76, 0.0), 1.0) * (len(one_file.NOTES) - 1)
        if index is None or abs(spot - index) > 0.5 + one_file.HYSTERESIS:
            return int(round(spot))
        return index

    def test_sweeping_plays_the_ladder_in_order(self):
        index, played = None, []
        for step in range(60):
            new = self.pick(step / 59, index)
            if new != index:
                played.append(one_file.NOTES[new])
            index = new
        assert played == one_file.NOTES

    def test_a_shaky_hand_holds_its_note(self):
        index = self.pick(0.5, None)
        changes = 0
        for step in range(40):
            new = self.pick(0.5 + 0.004 * (-1) ** step, index)
            changes += new != index
            index = new
        assert changes == 0
