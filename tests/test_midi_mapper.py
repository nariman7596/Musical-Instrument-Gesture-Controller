"""Config parsing, validation, value scaling and hot reload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.midi_mapper import (
    ConfigError,
    ConfigWatcher,
    ContinuousMapping,
    GateMapping,
    MidiMapper,
    NoteMapping,
    build_smoother,
    load_config,
    parse_config,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default_mapping.json"


def config_dict(*mappings, **overrides):
    """Minimal valid config document with the given mapping entries."""
    document = {"version": 1, "mappings": list(mappings)}
    document.update(overrides)
    return document


def build(entry, **overrides):
    """Parse a single mapping entry and return it."""
    return parse_config(config_dict(entry, **overrides)).mappings[0]


# -- shipped preset --------------------------------------------------------
class TestDefaultPreset:
    def test_the_shipped_mapping_file_is_valid(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        assert len(config.mappings) >= 10
        assert config.midi.channel == 1

    def test_every_mapping_uses_a_real_feature(self):
        from src.gesture_features import feature_names

        known = set(feature_names())
        for mapping in load_config(DEFAULT_CONFIG_PATH).mappings:
            assert mapping.feature in known, mapping.name
            if getattr(mapping, "velocity_feature", None):
                assert mapping.velocity_feature in known, mapping.name

    def test_preset_covers_the_documented_gesture_design(self):
        by_name = {m.name: m for m in load_config(DEFAULT_CONFIG_PATH).mappings}
        assert by_name["Filter Cutoff"].number == 74
        assert by_name["Master Volume"].number == 7
        assert by_name["Sustain Pedal"].kind == "gate" and by_name["Sustain Pedal"].number == 64
        assert by_name["Pitch Bend"].kind == "pitch_bend"
        assert by_name["Sample Trigger"].kind == "note"

    def test_describe_lists_every_mapping(self):
        text = load_config(DEFAULT_CONFIG_PATH).describe()
        assert "Filter Cutoff" in text and "CC 74" in text


# -- continuous mappings ---------------------------------------------------
class TestContinuousMapping:
    def test_linear_scaling_to_the_full_cc_range(self):
        mapping = build({"name": "cut", "feature": "right.height", "type": "cc", "cc": 74})
        assert mapping.update({"right.height": 0.0}, 0.0)[0].value == 0
        assert mapping.update({"right.height": 1.0}, 1.0)[0].value == 127
        assert mapping.update({"right.height": 0.5}, 2.0)[0].value == 64

    def test_input_range_clamps_and_expands(self):
        mapping = build({"name": "cut", "feature": "f", "type": "cc", "cc": 74,
                         "input_range": [0.25, 0.75]})
        assert mapping.update({"f": 0.1}, 0.0)[0].value == 0
        assert mapping.update({"f": 0.5}, 1.0)[0].value == 64
        assert mapping.update({"f": 0.9}, 2.0)[0].value == 127

    def test_invert(self):
        mapping = build({"name": "vol", "feature": "f", "type": "cc", "cc": 7, "invert": True})
        assert mapping.update({"f": 0.0}, 0.0)[0].value == 127

    def test_output_range_limits_the_travel(self):
        mapping = build({"name": "vol", "feature": "f", "type": "cc", "cc": 7,
                         "output_range": [40, 100]})
        assert mapping.update({"f": 0.0}, 0.0)[0].value == 40
        assert mapping.update({"f": 1.0}, 1.0)[0].value == 100

    def test_curve_bends_the_response(self):
        mapping = build({"name": "cut", "feature": "f", "type": "cc", "cc": 74, "curve": 2.0})
        # gamma > 1 gives finer resolution at the bottom of the range
        assert mapping.update({"f": 0.5}, 0.0)[0].value == 32

    def test_deadband_suppresses_micro_jitter(self):
        mapping = build({"name": "cut", "feature": "f", "type": "cc", "cc": 74, "deadband": 5})
        assert mapping.update({"f": 0.5}, 0.0)  # first value always sent
        assert mapping.update({"f": 0.51}, 1.0) == []       # +1 step: swallowed
        assert mapping.update({"f": 0.6}, 2.0)[0].value == 76

    def test_endpoints_bypass_the_deadband(self):
        """A big deadband must not stop the filter from fully closing."""
        mapping = build({"name": "cut", "feature": "f", "type": "cc", "cc": 74, "deadband": 40})
        mapping.update({"f": 0.5}, 0.0)
        assert mapping.update({"f": 0.0}, 1.0)[0].value == 0

    def test_unchanged_value_is_never_resent(self):
        mapping = build({"name": "cut", "feature": "f", "type": "cc", "cc": 74})
        mapping.update({"f": 0.5}, 0.0)
        assert mapping.update({"f": 0.5}, 1.0) == []

    def test_rate_limit_drops_intermediate_updates(self):
        mapping = build({"name": "cut", "feature": "f", "type": "cc", "cc": 74,
                         "rate_limit_hz": 10})
        mapping.update({"f": 0.30}, 0.0)
        assert mapping.update({"f": 0.40}, 0.02) == []   # 20 ms later: too soon
        assert mapping.update({"f": 0.50}, 0.20)          # 200 ms later: fine

    def test_missing_feature_sends_nothing(self):
        mapping = build({"name": "cut", "feature": "right.height", "type": "cc", "cc": 74})
        assert mapping.update({}, 0.0) == []
        assert mapping.state().tracked is False

    def test_disabled_mapping_is_silent(self):
        mapping = build({"name": "cut", "feature": "f", "type": "cc", "cc": 74, "enabled": False})
        assert mapping.update({"f": 1.0}, 0.0) == []

    def test_pitch_bend_uses_14_bits(self):
        mapping = build({"name": "bend", "feature": "f", "type": "pitch_bend"})
        assert mapping.update({"f": 0.0}, 0.0)[0].value == 0
        assert mapping.update({"f": 0.5}, 1.0)[0].value == 8192  # centre
        assert mapping.update({"f": 1.0}, 2.0)[0].value == 16383

    def test_aftertouch_is_a_7_bit_channel_message(self):
        mapping = build({"name": "at", "feature": "f", "type": "aftertouch"})
        event = mapping.update({"f": 1.0}, 0.0)[0]
        assert event.kind == "aftertouch" and event.value == 127
        assert len(event.to_bytes()) == 2


# -- gates and notes -------------------------------------------------------
class TestGateMapping:
    def test_switches_on_and_off_with_hysteresis(self):
        mapping = build({"name": "sustain", "feature": "left.fist", "type": "gate", "cc": 64,
                         "threshold": 0.6, "release": 0.4})
        assert mapping.update({"left.fist": 0.0}, 0.0)[0].value == 0  # initial state announced
        assert mapping.update({"left.fist": 0.7}, 1.0)[0].value == 127
        assert mapping.update({"left.fist": 0.5}, 2.0) == []          # inside the band
        assert mapping.update({"left.fist": 0.2}, 3.0)[0].value == 0

    def test_custom_on_off_values(self):
        mapping = build({"name": "mod", "feature": "f", "type": "gate", "cc": 1,
                         "value_on": 90, "value_off": 10})
        mapping.update({"f": 0.0}, 0.0)
        assert mapping.update({"f": 1.0}, 1.0)[0].value == 90

    def test_panic_releases_a_held_gate(self):
        mapping = build({"name": "sustain", "feature": "f", "type": "gate", "cc": 64})
        mapping.update({"f": 1.0}, 0.0)
        assert mapping.panic()[0].value == 0
        assert mapping.panic() == []  # nothing held any more


class TestNoteMapping:
    def test_note_on_then_off(self):
        mapping = build({"name": "kick", "feature": "left.point_up", "type": "note",
                         "note": 36, "velocity": 100})
        assert mapping.update({"left.point_up": 0.0}, 0.0) == []
        on = mapping.update({"left.point_up": 1.0}, 1.0)
        assert on[0].kind == "note_on" and on[0].number == 36 and on[0].value == 100
        assert mapping.update({"left.point_up": 0.9}, 2.0) == []  # still held
        off = mapping.update({"left.point_up": 0.0}, 3.0)
        assert off[0].kind == "note_off"

    def test_velocity_can_follow_another_feature(self):
        mapping = build({"name": "kick", "feature": "g", "type": "note", "note": 36,
                         "velocity": 100, "velocity_feature": "right.height"})
        event = mapping.update({"g": 1.0, "right.height": 0.5}, 0.0)[0]
        assert event.value == 50

    def test_velocity_never_reaches_zero(self):
        """Velocity 0 is a note-off in MIDI, so a soft hit must still sound."""
        mapping = build({"name": "kick", "feature": "g", "type": "note",
                         "velocity": 100, "velocity_feature": "v"})
        assert mapping.update({"g": 1.0, "v": 0.0}, 0.0)[0].value == 1

    def test_retrigger_restarts_the_note(self):
        mapping = build({"name": "kick", "feature": "g", "type": "note", "retrigger": True})
        mapping.update({"g": 1.0}, 0.0)
        kinds = [event.kind for event in mapping.update({"g": 1.0}, 1.0)]
        assert kinds == ["note_off", "note_on"]

    def test_panic_releases_a_sounding_note(self):
        mapping = build({"name": "kick", "feature": "g", "type": "note", "note": 40})
        mapping.update({"g": 1.0}, 0.0)
        assert mapping.panic()[0].kind == "note_off"


# -- validation ------------------------------------------------------------
class TestValidation:
    @pytest.mark.parametrize(
        "entry, message",
        [
            ({"feature": "f", "type": "wobble", "cc": 1}, "unknown type"),
            ({"type": "cc", "cc": 1}, "'feature' is required"),
            ({"feature": "f", "type": "cc"}, "needs a 'cc' number"),
            ({"feature": "f", "type": "cc", "cc": 300}, "between 0 and 127"),
            ({"feature": "f", "type": "cc", "cc": 1, "channel": 20}, "between 1 and 16"),
            ({"feature": "f", "type": "cc", "cc": 1, "curve": -1}, "must be positive"),
            ({"feature": "f", "type": "cc", "cc": 1, "typo": 3}, "unknown key"),
            ({"feature": "f", "type": "cc", "cc": 1, "input_range": [0.5, 0.5]}, "must differ"),
            ({"feature": "f", "type": "cc", "cc": "seven"}, "must be an integer"),
            ({"feature": "f", "type": "gate", "cc": 1, "threshold": 0.3, "release": 0.7},
             "must not be above"),
            ({"feature": "f", "type": "gate", "cc": 200}, "between 0 and 127"),
            ({"feature": "f", "type": "note", "note": 200}, "between 0 and 127"),
        ],
    )
    def test_bad_mapping_entries_are_rejected_with_a_useful_message(self, entry, message):
        with pytest.raises(ConfigError, match=message):
            parse_config(config_dict(entry))

    def test_empty_mapping_list_is_rejected(self):
        with pytest.raises(ConfigError, match="non-empty list"):
            parse_config({"version": 1, "mappings": []})

    def test_unsupported_version_is_rejected(self):
        with pytest.raises(ConfigError, match="unsupported config version"):
            parse_config({"version": 99, "mappings": [{"feature": "f", "type": "cc", "cc": 1}]})

    def test_unknown_top_level_key_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown key"):
            parse_config(config_dict({"feature": "f", "type": "cc", "cc": 1}, wat=1))

    def test_invalid_calibration_is_reported_with_context(self):
        with pytest.raises(ConfigError, match="config.calibration"):
            parse_config(config_dict({"feature": "f", "type": "cc", "cc": 1},
                                     calibration={"bogus": [0, 1]}))

    def test_unknown_feature_only_warns(self, caplog):
        """A typo in a feature name should not stop the show mid-set."""
        config = parse_config(config_dict({"feature": "right.nonsense", "type": "cc", "cc": 1}))
        assert config.mappings[0].feature == "right.nonsense"
        assert "unknown feature" in caplog.text

    def test_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(ConfigError, match="could not read"):
            load_config(tmp_path / "nope.json")

    def test_invalid_json_is_reported_clearly(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid JSON"):
            load_config(path)

    def test_type_aliases_are_accepted(self):
        assert build({"feature": "f", "type": "pitchbend"}).kind == "pitch_bend"
        assert build({"feature": "f", "type": "toggle", "cc": 64}).kind == "gate"
        assert build({"feature": "f", "type": "trigger"}).kind == "note"


# -- mapper --------------------------------------------------------------
class TestMidiMapper:
    def test_channel_defaults_come_from_the_midi_block(self):
        config = parse_config(config_dict(
            {"feature": "f", "type": "cc", "cc": 1},
            {"feature": "g", "type": "cc", "cc": 2, "channel": 5},
            midi={"channel": 3},
        ))
        assert [m.channel for m in config.mappings] == [3, 5]

    def test_update_collects_events_from_every_mapping(self):
        mapper = MidiMapper(parse_config(config_dict(
            {"name": "a", "feature": "f", "type": "cc", "cc": 1},
            {"name": "b", "feature": "g", "type": "cc", "cc": 2},
        )))
        events = mapper.update({"f": 1.0, "g": 0.0}, now=0.0)
        assert [(e.number, e.value) for e in events] == [(1, 127), (2, 0)]

    def test_panic_sends_all_notes_off_per_channel(self):
        mapper = MidiMapper(parse_config(config_dict(
            {"feature": "f", "type": "note", "note": 36, "channel": 1},
            {"feature": "g", "type": "cc", "cc": 1, "channel": 10},
        )))
        mapper.update({"f": 1.0, "g": 0.5}, now=0.0)
        events = mapper.panic()
        assert any(e.kind == "note_off" for e in events)
        assert sorted(e.channel for e in events if e.number == 123) == [1, 10]

    def test_states_expose_bars_for_the_overlay(self):
        mapper = MidiMapper(parse_config(config_dict(
            {"name": "cut", "feature": "f", "type": "cc", "cc": 74},
            {"name": "off", "feature": "g", "type": "cc", "cc": 75, "enabled": False},
        )))
        mapper.update({"f": 0.5}, now=0.0)
        states = mapper.states()
        assert [s.name for s in states] == ["cut"]  # disabled mappings are hidden
        assert states[0].label == "CC 74"
        assert states[0].normalised == pytest.approx(0.5, abs=0.01)

    def test_reset_forgets_sent_values(self):
        mapper = MidiMapper(parse_config(config_dict({"feature": "f", "type": "cc", "cc": 1})))
        mapper.update({"f": 0.5}, now=0.0)
        assert mapper.update({"f": 0.5}, now=1.0) == []
        mapper.reset()
        assert mapper.update({"f": 0.5}, now=2.0) != []


class TestBuildSmoother:
    def test_global_window_and_per_mapping_override(self):
        smoother = build_smoother(parse_config(config_dict(
            {"feature": "slow", "type": "cc", "cc": 1},
            {"feature": "fast", "type": "cc", "cc": 2, "smoothing": 1},
            smoothing={"window": 9},
        )))
        assert smoother.alpha == pytest.approx(0.2)
        assert smoother.alpha_for("fast") == 1.0
        assert smoother.alpha_for("slow") == pytest.approx(0.2)


# -- hot reload ------------------------------------------------------------
class TestConfigWatcher:
    @staticmethod
    def write(path: Path, cc: int) -> None:
        path.write_text(json.dumps(config_dict(
            {"name": "cut", "feature": "right.height", "type": "cc", "cc": cc}
        )), encoding="utf-8")

    def test_reloads_after_the_file_changes(self, tmp_path):
        path = tmp_path / "mapping.json"
        self.write(path, 74)
        watcher = ConfigWatcher(path, min_interval=0.0)
        assert watcher.poll(now=1.0) is None       # nothing changed yet

        self.write(path, 71)
        config = watcher.poll(now=2.0)
        assert config is not None and config.mappings[0].number == 71
        assert watcher.poll(now=3.0) is None       # only once per change

    def test_keeps_running_when_the_new_file_is_invalid(self, tmp_path, caplog):
        path = tmp_path / "mapping.json"
        self.write(path, 74)
        watcher = ConfigWatcher(path, min_interval=0.0)
        path.write_text("{oops", encoding="utf-8")
        assert watcher.poll(now=2.0) is None
        assert "keeping previous mapping" in caplog.text

    def test_respects_the_polling_interval(self, tmp_path):
        path = tmp_path / "mapping.json"
        self.write(path, 74)
        watcher = ConfigWatcher(path, min_interval=10.0)
        self.write(path, 71)
        assert watcher.poll(now=0.1) is None   # too soon to even stat the file
        assert watcher.poll(now=20.0) is not None

    def test_missing_file_is_ignored(self, tmp_path):
        path = tmp_path / "mapping.json"
        self.write(path, 74)
        watcher = ConfigWatcher(path, min_interval=0.0)
        path.unlink()
        assert watcher.poll(now=5.0) is None
