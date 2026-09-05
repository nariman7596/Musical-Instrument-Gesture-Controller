"""End-to-end behaviour of landmarks -> features -> smoothing -> MIDI.

These tests skip MediaPipe entirely (landmarks are synthesised or replayed from
the fixtures) so the musical behaviour of the whole chain can be asserted
deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.gesture_features import build_feature_vector
from src.midi_mapper import MidiMapper, build_smoother, load_config, parse_config
from src.midi_output import NullOutput

from .conftest import make_hand
from .test_midi_mapper import DEFAULT_CONFIG_PATH, config_dict

WIDE = 16 / 9


def run_frames(hands_per_frame, config, aspect=WIDE):
    """Push a list of ``{"left": lm|None, "right": lm|None}`` through the pipeline."""
    return [event for _, event in run_frames_indexed(hands_per_frame, config, aspect)]


def run_frames_indexed(hands_per_frame, config, aspect=WIDE):
    """As :func:`run_frames`, but pairs every event with the frame it came from."""
    mapper = MidiMapper(config)
    smoother = build_smoother(config)
    output = NullOutput()
    timeline = []
    for index, hands in enumerate(hands_per_frame):
        features = smoother.update(build_feature_vector(hands, config.calibration, aspect))
        events = mapper.update(features, now=index / 30.0)
        output.send_all(events)
        timeline.extend((index, event) for event in events)
    return timeline


def cc_values(events, number):
    return [event.value for event in events if event.kind == "cc" and event.number == number]


class TestFullChain:
    def test_raising_the_hand_opens_the_filter(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        frames = [
            {"right": make_hand(center=(0.5, y), aspect=WIDE), "left": None}
            for y in np.linspace(0.85, 0.15, 40)  # hand travels upwards
        ]
        frames += [frames[-1]] * 10          # hold still so the EMA can settle
        values = cc_values(run_frames(frames, config), 74)
        assert len(values) > 10
        assert values == sorted(values)      # monotonic sweep, no jumps back
        assert values[0] < 20 and values[-1] > 110

    def test_a_still_hand_stops_sending(self):
        """No movement must mean no MIDI traffic at all."""
        config = load_config(DEFAULT_CONFIG_PATH)
        still = {"right": make_hand(center=(0.5, 0.5), aspect=WIDE), "left": None}
        timeline = run_frames_indexed([still] * 60, config)
        assert timeline, "the first frames should announce the initial values"
        # Everything is said in the first few frames; after that, silence.
        assert max(index for index, _ in timeline) < 10

    def test_closing_the_left_fist_presses_the_sustain_pedal(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        open_hand = make_hand(curl=0.0, thumb=1.0, center=(0.3, 0.5), aspect=WIDE)
        fist = make_hand(curl=1.0, thumb=0.0, center=(0.3, 0.5), aspect=WIDE)
        frames = [{"left": open_hand, "right": None}] * 10 + [{"left": fist, "right": None}] * 10
        assert cc_values(run_frames(frames, config), 64) == [0, 127]

    def test_pointing_the_left_index_fires_a_note(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        point = make_hand(curls={"index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
                          thumb=0.0, center=(0.3, 0.4), aspect=WIDE)
        rest = make_hand(curl=1.0, thumb=0.0, center=(0.3, 0.4), aspect=WIDE)
        events = run_frames([{"left": rest}] * 5 + [{"left": point}] * 10 + [{"left": rest}] * 10,
                            config)
        kinds = [event.kind for event in events if event.kind.startswith("note")]
        assert kinds == ["note_on", "note_off"]

    def test_losing_a_hand_freezes_its_controllers(self):
        """Tracking dropping out must not slam every parameter to zero."""
        config = load_config(DEFAULT_CONFIG_PATH)
        visible = {"right": make_hand(center=(0.5, 0.35), aspect=WIDE), "left": None}
        with_gap = [visible] * 20 + [{"right": None, "left": None}] * 20
        events = run_frames(with_gap, config)
        assert cc_values(events, 74)[-1] > 40  # last value is the held one, not 0

    def test_two_hand_features_only_fire_with_both_hands(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        one = [{"right": make_hand(center=(0.6, 0.5), aspect=WIDE), "left": None}] * 10
        both = [{"right": make_hand(center=(0.7, 0.5), aspect=WIDE),
                 "left": make_hand(center=(0.3, 0.5), aspect=WIDE)}] * 10
        assert cc_values(run_frames(one, config), 93) == []       # chorus depth silent
        assert cc_values(run_frames(one + both, config), 93) != []

    def test_smoothing_reduces_the_message_rate_on_a_jittery_hand(self):
        """The point of the EMA: fewer, calmer messages from a shaky hand."""
        rng = np.random.default_rng(7)
        frames = [
            {"right": make_hand(center=(0.5, 0.5 + rng.normal(0, 0.004)), aspect=WIDE)}
            for _ in range(120)
        ]
        entry = {"name": "cut", "feature": "right.height", "type": "cc", "cc": 74}
        smoothed = parse_config(config_dict(entry, smoothing={"window": 8}))
        raw = parse_config(config_dict(entry, smoothing={"window": 1}))
        assert len(run_frames(frames, smoothed)) < len(run_frames(frames, raw)) * 0.75


class TestRealHandsThroughTheDefaultPreset:
    def test_photographed_gestures_produce_sensible_midi(self, reference_cases):
        config = load_config(DEFAULT_CONFIG_PATH)
        for case in reference_cases:
            hands = {case["handedness"]: np.asarray(case["landmarks"])}
            events = run_frames([hands] * 5, config, aspect=case["aspect"])
            assert events, f"{case['image']} produced no MIDI at all"
            for event in events:
                assert 0 <= event.value <= (16383 if event.kind == "pitch_bend" else 127)

    def test_a_photographed_fist_presses_the_sustain_pedal(self, reference_cases):
        config = load_config(DEFAULT_CONFIG_PATH)
        fist = next(c for c in reference_cases if c["expected_pose"] == "fist")
        # The fixture fist is a right hand; the pedal listens to the left one.
        landmarks = np.asarray(fist["landmarks"])
        events = run_frames([{"left": landmarks}] * 12, config, aspect=fist["aspect"])
        assert 127 in cc_values(events, 64)
