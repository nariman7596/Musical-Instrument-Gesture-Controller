"""Hand position -> notes of a scale: the mapping that makes it an instrument."""

from __future__ import annotations

import pytest

from src.midi_mapper import (
    SCALES,
    ConfigError,
    MidiMapper,
    ScaleMapping,
    load_config,
    parse_config,
    scale_notes,
)

from .test_midi_mapper import config_dict

PLAY_CONFIG = "config/play.json"


def lead(**overrides):
    entry = {"name": "Lead", "feature": "right.x", "type": "scale",
             "root": 57, "scale": "pentatonic_minor", "octaves": 2}
    entry.update(overrides)
    return parse_config(config_dict(entry)).mappings[0]


def sweep(mapping, values, gate=1.0):
    """Play a sequence of positions, returning the events produced."""
    events = []
    for index, value in enumerate(values):
        events.extend(mapping.update(
            {"right.x": value, "right.present": gate, "right.height": 0.8}, index * 0.03
        ))
    return events


def notes_on(events):
    return [event.number for event in events if event.kind == "note_on"]


class TestScaleNotes:
    def test_pentatonic_minor_from_a3(self):
        assert scale_notes(57, "pentatonic_minor", 2) == [57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81]

    def test_every_named_scale_builds(self):
        for name in SCALES:
            notes = scale_notes(60, name, 1)
            assert notes == sorted(notes)
            assert all(0 <= note <= 127 for note in notes)

    def test_notes_are_clamped_to_the_midi_range(self):
        assert all(note <= 127 for note in scale_notes(120, "major", 4))

    def test_unknown_scale_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown scale"):
            scale_notes(60, "klingon", 1)


class TestPlaying:
    def test_sweeping_the_hand_plays_the_scale_in_order(self):
        played = notes_on(sweep(lead(), [i / 40 for i in range(41)]))
        assert played == [57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81]

    def test_every_note_belongs_to_the_scale(self):
        mapping = lead(scale="blues", root=48, octaves=2)
        played = notes_on(sweep(mapping, [i / 60 for i in range(61)]))
        assert set(played) <= set(mapping.notes)

    def test_it_is_monophonic(self):
        """Each new note must release the previous one, or notes pile up."""
        events = sweep(lead(), [i / 40 for i in range(41)])
        sounding = 0
        for event in events:
            sounding += 1 if event.kind == "note_on" else -1
            assert 0 <= sounding <= 1

    def test_hysteresis_stops_a_hovering_hand_stuttering(self):
        """A hand resting on a note boundary must not machine-gun two pitches."""
        mapping = lead(hysteresis=0.25)
        sweep(mapping, [0.5])
        jitter = [0.5 + 0.004 * (-1) ** i for i in range(40)]   # a shaky hand
        assert notes_on(sweep(mapping, jitter)) == []

    def test_a_deliberate_move_still_changes_the_note(self):
        mapping = lead(hysteresis=0.25)
        sweep(mapping, [0.5])
        assert notes_on(sweep(mapping, [0.6, 0.7])) != []

    def test_invert_reverses_the_keyboard(self):
        played = notes_on(sweep(lead(invert=True), [i / 40 for i in range(41)]))
        assert played == sorted(played, reverse=True)

    def test_input_range_maps_the_usable_part_of_the_frame(self):
        mapping = lead(input_range=[0.25, 0.75])
        assert notes_on(sweep(mapping, [0.0]))[0] == mapping.notes[0]
        assert notes_on(sweep(mapping, [1.0]))[-1] == mapping.notes[-1]

    def test_velocity_can_follow_another_feature(self):
        mapping = lead(velocity=100, velocity_feature="right.height")
        events = mapping.update({"right.x": 0.0, "right.present": 1.0, "right.height": 0.5}, 0.0)
        assert events[0].value == 50


class TestGate:
    def test_nothing_sounds_until_the_gate_opens(self):
        mapping = lead()
        assert mapping.update({"right.x": 0.5, "right.present": 0.0}, 0.0) == []

    def test_losing_the_hand_releases_the_note(self):
        mapping = lead()
        assert notes_on(sweep(mapping, [0.5]))
        released = mapping.update({"right.x": 0.5, "right.present": 0.0}, 1.0)
        assert [event.kind for event in released] == ["note_off"]

    def test_the_gate_defaults_to_the_hands_presence(self):
        assert lead().gate_feature == "right.present"

    def test_an_explicit_gate_feature_is_honoured(self):
        mapping = lead(gate_feature="right.open_palm")
        assert mapping.gate_feature == "right.open_palm"
        assert mapping.update({"right.x": 0.5, "right.open_palm": 0.0}, 0.0) == []
        assert notes_on(mapping.update({"right.x": 0.5, "right.open_palm": 1.0}, 0.1))

    def test_panic_releases_a_sounding_note(self):
        mapping = lead()
        sweep(mapping, [0.5])
        assert [event.kind for event in mapping.panic()] == ["note_off"]
        assert mapping.panic() == []


class TestValidation:
    @pytest.mark.parametrize("entry, message", [
        ({"feature": "f", "type": "scale", "scale": "klingon"}, "unknown scale"),
        ({"feature": "f", "type": "scale", "root": 200}, "between 0 and 127"),
        ({"feature": "f", "type": "scale", "octaves": 0}, "must be positive"),
        ({"feature": "f", "type": "scale", "threshold": 0.2, "release": 0.9}, "must not be above"),
        ({"feature": "f", "type": "scale", "wat": 1}, "unknown key"),
    ])
    def test_bad_entries_are_rejected(self, entry, message):
        with pytest.raises(ConfigError, match=message):
            parse_config(config_dict(entry))

    def test_type_aliases(self):
        for alias in ("scale", "pitch", "lead"):
            assert parse_config(config_dict(
                {"feature": "right.x", "type": alias}
            )).mappings[0].kind == "scale"


class TestSynthBlock:
    def test_a_patch_can_turn_the_drone_off(self):
        config = parse_config(config_dict(
            {"feature": "right.x", "type": "scale"}, synth={"drone": -1, "gain": 0.4}
        ))
        assert config.synth.drone is None
        assert config.synth.gain == 0.4

    def test_the_default_is_a_drone(self):
        """Control-change patches need something for the filter to shape."""
        config = parse_config(config_dict({"feature": "f", "type": "cc", "cc": 1}))
        assert config.synth.drone == 45

    def test_unknown_synth_key_is_rejected(self):
        with pytest.raises(ConfigError, match="config.synth"):
            parse_config(config_dict({"feature": "f", "type": "cc", "cc": 1}, synth={"volume": 1}))


class TestPlayPreset:
    def test_the_shipped_playable_preset_is_valid(self):
        config = load_config(PLAY_CONFIG)
        assert config.synth.drone is None          # it makes its own notes
        lead_mapping = config.mappings[0]
        assert isinstance(lead_mapping, ScaleMapping)
        assert lead_mapping.feature == "right.x"

    def test_playing_it_produces_a_melody_in_key(self):
        config = load_config(PLAY_CONFIG)
        mapper = MidiMapper(config)
        lead_mapping = config.mappings[0]
        played = []
        for index in range(60):
            features = {"right.x": index / 59, "right.present": 1.0, "right.height": 0.7}
            played += [e.number for e in mapper.update(features, index * 0.03) if e.kind == "note_on"]
        assert len(played) > 5
        assert set(played) <= set(lead_mapping.notes)
        assert played == sorted(played)
