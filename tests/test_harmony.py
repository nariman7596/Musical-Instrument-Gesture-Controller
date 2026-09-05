"""Chords and arpeggios: the mappings that add harmony and rhythm."""

from __future__ import annotations

import pytest

from src.midi_mapper import (
    ArpeggioMapping,
    ChordMapping,
    ConfigError,
    MidiMapper,
    chord_from_scale,
    load_config,
    parse_config,
)

from .test_midi_mapper import config_dict

ENSEMBLE = "config/ensemble.json"


def build(**overrides):
    entry = {"name": "Chords", "feature": "left.x", "type": "chord",
             "root": 48, "scale": "minor", "span": 5}
    entry.update(overrides)
    return parse_config(config_dict(entry)).mappings[0]


def notes_on(events):
    return [event.number for event in events if event.kind == "note_on"]


def notes_off(events):
    return [event.number for event in events if event.kind == "note_off"]


class TestChordConstruction:
    def test_triads_stack_within_the_scale(self):
        assert chord_from_scale(48, "minor", 0) == [48, 51, 55]     # C minor
        assert chord_from_scale(48, "minor", 1) == [50, 53, 56]     # D diminished

    def test_sevenths_add_a_fourth_note(self):
        assert chord_from_scale(48, "minor", 0, size=4) == [48, 51, 55, 58]

    def test_stacking_stays_in_key(self):
        from src.midi_mapper import SCALES

        pitch_classes = {(48 + degree) % 12 for degree in SCALES["minor"]}
        for degree in range(7):
            for note in chord_from_scale(48, "minor", degree):
                assert note % 12 in pitch_classes

    def test_notes_beyond_midi_range_are_dropped(self):
        assert all(note <= 127 for note in chord_from_scale(120, "major", 6, size=4))

    def test_unknown_scale_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown scale"):
            chord_from_scale(48, "nonsense", 0)


class TestChordMapping:
    def test_a_chord_sounds_as_several_notes_at_once(self):
        played = notes_on(build().update({"left.x": 0.0, "left.present": 1.0}, 0.0))
        assert played == [48, 51, 55]

    def test_moving_the_hand_changes_chord_and_releases_the_old_one(self):
        mapping = build()
        mapping.update({"left.x": 0.0, "left.present": 1.0}, 0.0)
        events = mapping.update({"left.x": 1.0, "left.present": 1.0}, 1.0)
        assert notes_off(events) == [48, 51, 55]
        assert notes_on(events) == [55, 58, 62]

    def test_a_still_hand_holds_the_chord(self):
        mapping = build()
        mapping.update({"left.x": 0.5, "left.present": 1.0}, 0.0)
        assert mapping.update({"left.x": 0.5, "left.present": 1.0}, 1.0) == []

    def test_hysteresis_stops_a_shaky_hand_switching_chords(self):
        mapping = build(hysteresis=0.25)
        mapping.update({"left.x": 0.5, "left.present": 1.0}, 0.0)
        for step in range(30):
            jitter = 0.5 + 0.004 * (-1) ** step
            assert mapping.update({"left.x": jitter, "left.present": 1.0}, step) == []

    def test_losing_the_hand_releases_the_chord(self):
        mapping = build()
        mapping.update({"left.x": 0.0, "left.present": 1.0}, 0.0)
        assert notes_off(mapping.update({"left.x": 0.0, "left.present": 0.0}, 1.0)) == [48, 51, 55]

    def test_panic_releases_everything(self):
        mapping = build()
        mapping.update({"left.x": 0.0, "left.present": 1.0}, 0.0)
        assert len(notes_off(mapping.panic())) == 3
        assert mapping.panic() == []

    def test_size_controls_how_many_notes_sound(self):
        assert len(notes_on(build(size=4).update({"left.x": 0.0, "left.present": 1.0}, 0.0))) == 4


def arpeggio(**overrides):
    entry = {"name": "Arp", "feature": "left.x", "type": "arpeggio",
             "root": 48, "scale": "minor", "span": 5, "octaves": 1,
             "pattern": "up", "rate_range": [4.0, 4.0]}
    entry.update(overrides)
    return parse_config(config_dict(entry)).mappings[0]


def play(mapping, seconds=2.0, fps=30.0, features=None):
    """Run the mapping over simulated frames, returning every event."""
    features = features or {"left.x": 0.0, "left.present": 1.0}
    events = []
    for frame in range(int(seconds * fps)):
        events.extend(mapping.update(features, frame / fps))
    return events


class TestArpeggio:
    def test_it_plays_notes_in_time_rather_than_all_at_once(self):
        events = play(arpeggio(), seconds=2.0)
        played = notes_on(events)
        assert 6 <= len(played) <= 10       # 4 per second, give or take a frame
        assert played[:3] == [48, 51, 55]

    def test_only_one_note_sounds_at_a_time(self):
        sounding = 0
        for event in play(arpeggio(), seconds=3.0):
            sounding += 1 if event.kind == "note_on" else -1
            assert 0 <= sounding <= 1

    def test_the_rate_feature_speeds_it_up(self):
        slow = arpeggio(rate_feature="left.height", rate_range=[1.0, 12.0])
        fast = arpeggio(rate_feature="left.height", rate_range=[1.0, 12.0])
        slow_notes = notes_on(play(slow, 3.0, features={"left.x": 0.0, "left.present": 1.0, "left.height": 0.0}))
        fast_notes = notes_on(play(fast, 3.0, features={"left.x": 0.0, "left.present": 1.0, "left.height": 1.0}))
        assert len(fast_notes) > len(slow_notes) * 3

    def test_patterns_change_the_order(self):
        up = notes_on(play(arpeggio(pattern="up"), 1.5))
        down = notes_on(play(arpeggio(pattern="down"), 1.5))
        assert up[:3] == sorted(up[:3])
        assert down[:3] == sorted(down[:3], reverse=True)

    def test_updown_turns_around_without_repeating_the_top(self):
        mapping = arpeggio(pattern="updown", octaves=1)
        played = notes_on(play(mapping, 3.0))
        assert played[:5] == [48, 51, 55, 51, 48]

    def test_octaves_widen_the_range(self):
        one = set(notes_on(play(arpeggio(octaves=1), 3.0)))
        two = set(notes_on(play(arpeggio(octaves=2), 3.0)))
        assert max(two) > max(one)

    def test_losing_the_hand_stops_it(self):
        mapping = arpeggio()
        play(mapping, 1.0)
        stop = mapping.update({"left.x": 0.0, "left.present": 0.0}, 5.0)
        assert all(event.kind == "note_off" for event in stop)
        assert mapping.update({"left.x": 0.0, "left.present": 0.0}, 6.0) == []

    def test_moving_to_a_new_chord_changes_the_notes(self):
        mapping = arpeggio()
        first = set(notes_on(play(mapping, 2.0, features={"left.x": 0.0, "left.present": 1.0})))
        second = set(notes_on(play(mapping, 2.0, features={"left.x": 1.0, "left.present": 1.0})))
        assert first != second

    def test_notes_are_released_before_the_next_one_starts(self):
        """gate_length shorter than the step is what makes it rhythmic."""
        events = play(arpeggio(gate_length=0.5), seconds=2.0)
        kinds = [event.kind for event in events]
        for index in range(len(kinds) - 1):
            if kinds[index] == "note_on":
                assert "note_off" in kinds[index + 1:index + 4]


class TestValidation:
    @pytest.mark.parametrize("entry, message", [
        ({"feature": "f", "type": "chord", "scale": "klingon"}, "unknown scale"),
        ({"feature": "f", "type": "chord", "root": 999}, "between 0 and 127"),
        ({"feature": "f", "type": "chord", "span": 0}, "at least 1"),
        ({"feature": "f", "type": "chord", "size": 9}, "between 1 and 6"),
        ({"feature": "f", "type": "arpeggio", "pattern": "sideways"}, "unknown pattern"),
        ({"feature": "f", "type": "arpeggio", "rate_range": [0, 5]}, "must be positive"),
        ({"feature": "f", "type": "chord", "nonsense": 1}, "unknown key"),
    ])
    def test_bad_entries_are_rejected(self, entry, message):
        with pytest.raises(ConfigError, match=message):
            parse_config(config_dict(entry))

    def test_aliases(self):
        assert parse_config(config_dict({"feature": "f", "type": "harmony"})).mappings[0].kind == "chord"
        assert parse_config(config_dict({"feature": "f", "type": "arp"})).mappings[0].kind == "arpeggio"


class TestEnsemblePreset:
    def test_the_shipped_preset_is_valid(self):
        config = load_config(ENSEMBLE)
        kinds = {mapping.kind for mapping in config.mappings}
        assert {"scale", "arpeggio", "cc", "gate"} <= kinds
        assert config.synth.drone is None

    def test_both_hands_play_independent_parts(self):
        config = load_config(ENSEMBLE)
        mapper = MidiMapper(config)
        melody, arp = [], []
        for frame in range(120):
            features = {
                "right.x": (frame % 40) / 39, "right.present": 1.0, "right.speed": 0.6,
                "left.x": 0.2, "left.present": 1.0, "left.height": 0.9,
            }
            for event in mapper.update(features, frame / 30):
                if event.kind == "note_on":
                    (melody if event.label == "Melody" else arp).append(event.number)
        assert len(melody) > 5 and len(arp) > 5
        assert min(melody) > max(arp)       # melody sits above the accompaniment
