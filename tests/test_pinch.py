"""Thumb-to-finger pinches: four keys under one thumb.

The subtle part is that a *curled* finger sits as close to the thumb as a
pinched one — a clenched fist measures the same distance as a real thumb-to-index
touch — so closeness alone reports a fist as four simultaneous pinches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.gesture_features import (
    INDEX_TIP,
    MIDDLE_TIP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
    build_feature_vector,
    hand_features,
)
from src.midi_mapper import MidiMapper, build_smoother, load_config, parse_config

from .conftest import make_hand
from .test_midi_mapper import config_dict

WIDE = 16 / 9
PINCH_CONFIG = "config/pinch.json"
FINGERS = (("index", INDEX_TIP), ("middle", MIDDLE_TIP), ("ring", RING_TIP), ("pinky", PINKY_TIP))


def pinching(finger_tip, **kwargs):
    """A hand with the thumb tip touching one fingertip."""
    landmarks = make_hand(curl=0.25, thumb=1.0, aspect=WIDE, **kwargs)
    landmarks[THUMB_TIP] = landmarks[finger_tip]
    return landmarks


class TestPinchDetection:
    @pytest.mark.parametrize("name, tip", FINGERS)
    def test_touching_a_fingertip_registers_as_that_pinch(self, name, tip):
        features = hand_features(pinching(tip), "right", aspect=WIDE)
        assert features[f"pinch_{name}"] > 0.9
        assert features["pinch_any"] > 0.9

    @pytest.mark.parametrize("name, tip", FINGERS)
    def test_the_selector_names_the_pinched_finger(self, name, tip):
        expected = [finger for finger, _ in FINGERS].index(name) / 3
        features = hand_features(pinching(tip), "right", aspect=WIDE)
        assert features["pinch_select"] == pytest.approx(expected)

    def test_an_open_hand_is_not_pinching(self):
        features = hand_features(make_hand(curl=0.0, thumb=1.0, aspect=WIDE), "right", aspect=WIDE)
        assert features["pinch_any"] < 0.2

    def test_a_fist_is_not_four_pinches(self):
        """The whole reason extension is part of the measure."""
        features = hand_features(make_hand(curl=1.0, thumb=0.0, aspect=WIDE), "right", aspect=WIDE)
        assert features["pinch_any"] < 0.2

    def test_no_photographed_pose_registers_a_pinch(self, reference_cases):
        """None of the reference photos is pinching, so none may report one."""
        for case in reference_cases:
            features = hand_features(
                np.asarray(case["landmarks"]), case["handedness"], aspect=case["aspect"]
            )
            assert features["pinch_any"] < 0.5, f"{case['image']} ({case['expected_pose']})"

    def test_the_selector_is_absent_when_not_pinching(self):
        """Absent means 'hold', which is what stops a stray note on release."""
        vector = build_feature_vector(
            {"right": make_hand(curl=0.0, thumb=1.0, aspect=WIDE)}, aspect=WIDE
        )
        assert vector["right.pinch_any"] < 0.2
        assert "right.pinch_select" not in vector

    def test_the_selector_is_present_while_pinching(self):
        vector = build_feature_vector({"right": pinching(RING_TIP)}, aspect=WIDE)
        assert vector["right.pinch_select"] == pytest.approx(2 / 3)

    def test_pinch_any_is_the_strongest_of_the_four(self):
        features = hand_features(pinching(MIDDLE_TIP), "right", aspect=WIDE)
        singles = [features[f"pinch_{name}"] for name, _ in FINGERS]
        assert features["pinch_any"] == pytest.approx(max(singles))


def keys(**overrides):
    entry = {
        "name": "Keys", "feature": "right.pinch_select", "type": "scale",
        "root": 57, "scale": "pentatonic_minor", "octaves": 3, "span": 4,
        "gate_feature": "right.pinch_any", "offset_feature": "right.height",
        "offset_span": 3, "offset_step": 4, "hysteresis": 0.15,
    }
    entry.update(overrides)
    return parse_config(config_dict(entry)).mappings[0]


def play(mapping, selector, height, now=0.0):
    return [
        event.number
        for event in mapping.update(
            {"right.pinch_select": selector, "right.pinch_any": 1.0, "right.height": height}, now
        )
        if event.kind == "note_on"
    ]


class TestKeyLayout:
    def test_span_keeps_the_four_pinches_adjacent(self):
        """Without span the four pinches spread across the whole ladder."""
        mapping = keys()
        played = []
        for step, selector in enumerate((0.0, 1 / 3, 2 / 3, 1.0)):
            mapping.reset()
            played += play(mapping, selector, 0.0, step)
        assert played == [57, 60, 62, 64]

    @pytest.mark.parametrize("height, expected", [
        (0.0, [57, 60, 62, 64]),
        (0.5, [67, 69, 72, 74]),
        (1.0, [76, 79, 81, 84]),
    ])
    def test_hand_height_shifts_the_whole_set(self, height, expected):
        mapping = keys()
        played = []
        for step, selector in enumerate((0.0, 1 / 3, 2 / 3, 1.0)):
            mapping.reset()
            played += play(mapping, selector, height, step)
        assert played == expected

    def test_twelve_distinct_notes_are_reachable(self):
        mapping = keys()
        reachable = set()
        for height in (0.0, 0.5, 1.0):
            for selector in (0.0, 1 / 3, 2 / 3, 1.0):
                mapping.reset()
                reachable.update(play(mapping, selector, height))
        assert len(reachable) == 12


class TestRegisterTracking:
    def test_the_register_follows_the_hand_with_nothing_sounding(self):
        """Moving your hand changes octave whether or not a note is playing.

        Otherwise the first note after moving sounds in the register you left.
        """
        mapping = keys()
        for step in range(5):     # hand raised, not pinching
            mapping.update(
                {"right.pinch_select": 0.0, "right.pinch_any": 0.0, "right.height": 1.0}, step
            )
        assert play(mapping, 0.0, 1.0, 10) == [76]

    def test_no_stray_note_when_changing_register_and_pinching_at_once(self):
        mapping = keys()
        played = play(mapping, 0.0, 0.0, 0.0)             # low register
        mapping.update({"right.pinch_any": 0.0, "right.height": 0.0}, 1.0)
        played += play(mapping, 0.0, 1.0, 2.0)            # jump straight to high
        assert played == [57, 76]


class TestSmootherWiring:
    def test_a_mappings_window_also_covers_its_offset_feature(self):
        """Both inputs decide one note, so they must respond together."""
        config = parse_config(config_dict({
            "name": "Keys", "feature": "right.pinch_select", "type": "scale",
            "offset_feature": "right.height", "smoothing": 1,
        }, smoothing={"window": 12}))
        smoother = build_smoother(config)
        assert smoother.alpha_for("right.pinch_select") == 1.0
        assert smoother.alpha_for("right.height") == 1.0
        assert smoother.alpha_for("right.x") < 0.2      # everything else stays smoothed


class TestPinchPreset:
    def test_the_shipped_preset_is_valid(self):
        config = load_config(PINCH_CONFIG)
        lead = config.mappings[0]
        assert lead.feature == "right.pinch_select"
        assert lead.gate_feature == "right.pinch_any"
        assert lead.span == 4
        assert config.synth.drone is None

    def test_the_selector_is_not_smoothed(self):
        """Which finger you are touching is a choice, not a position."""
        assert load_config(PINCH_CONFIG).mappings[0].smoothing == 1

    def test_playing_the_four_keys_gives_four_notes(self):
        config = load_config(PINCH_CONFIG)
        mapper = MidiMapper(config)
        played = []
        for step, selector in enumerate((0.0, 1 / 3, 2 / 3, 1.0)):
            features = {
                "right.pinch_select": selector, "right.pinch_any": 1.0, "right.height": 0.05,
            }
            played += [e.number for e in mapper.update(features, step) if e.kind == "note_on"]
            mapper.update({"right.pinch_any": 0.0, "right.height": 0.05}, step + 0.5)
        assert len(set(played)) == 4
        assert played == sorted(played)
