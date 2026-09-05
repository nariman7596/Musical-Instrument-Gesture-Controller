"""MIDI encoding, port resolution and the dry-run output."""

from __future__ import annotations

import pytest

from src.midi_output import (
    MidiEvent,
    MidiOutput,
    NullOutput,
    clamp7,
    clamp14,
    open_output,
)


class TestEncoding:
    def test_control_change(self):
        assert MidiEvent("cc", 1, 74, 100).to_bytes() == [0xB0, 74, 100]

    def test_channel_is_one_based_on_the_outside(self):
        assert MidiEvent("cc", 16, 1, 1).to_bytes()[0] == 0xBF
        assert MidiEvent("cc", 1, 1, 1).to_bytes()[0] == 0xB0

    def test_channel_is_clamped_rather_than_wrapping(self):
        """Wrapping would silently send on a channel nobody asked for."""
        assert MidiEvent("cc", 99, 1, 1).to_bytes()[0] == 0xBF
        assert MidiEvent("cc", 0, 1, 1).to_bytes()[0] == 0xB0

    def test_note_on_and_off(self):
        assert MidiEvent("note_on", 10, 36, 110).to_bytes() == [0x99, 36, 110]
        assert MidiEvent("note_off", 10, 36, 0).to_bytes() == [0x89, 36, 0]

    def test_aftertouch_is_two_bytes(self):
        assert MidiEvent("aftertouch", 1, value=64).to_bytes() == [0xD0, 64]

    @pytest.mark.parametrize("value, lsb, msb", [(0, 0, 0), (8192, 0, 64), (16383, 127, 127)])
    def test_pitch_bend_splits_into_two_7_bit_bytes(self, value, lsb, msb):
        assert MidiEvent("pitch_bend", 1, value=value).to_bytes() == [0xE0, lsb, msb]

    def test_out_of_range_values_are_clamped(self):
        assert MidiEvent("cc", 1, 74, 999).to_bytes()[2] == 127
        assert MidiEvent("cc", 1, 74, -5).to_bytes()[2] == 0
        assert MidiEvent("pitch_bend", 1, value=99999).to_bytes() == [0xE0, 127, 127]

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown MIDI event kind"):
            MidiEvent("sysex", 1).to_bytes()

    def test_clamp_helpers(self):
        assert clamp7(200) == 127 and clamp7(-1) == 0 and clamp7(63.6) == 64
        assert clamp14(-1) == 0 and clamp14(99999) == 16383

    def test_str_is_readable_for_logs(self):
        assert "CC 74" in str(MidiEvent("cc", 1, 74, 100, "cutoff"))
        assert "cutoff" in str(MidiEvent("cc", 1, 74, 100, "cutoff"))


class TestNullOutput:
    def test_records_events_without_touching_hardware(self):
        output = NullOutput()
        output.send_all([MidiEvent("cc", 1, 74, 10), MidiEvent("cc", 1, 74, 20)])
        assert [event.value for event in output.events] == [10, 20]

    def test_open_output_dry_run_returns_null(self):
        assert isinstance(open_output(dry_run=True), NullOutput)

    def test_echo_prints_each_message(self, capsys):
        NullOutput(echo=True).send(MidiEvent("cc", 1, 74, 100, "cutoff"))
        assert "cutoff" in capsys.readouterr().out

    def test_usable_as_a_context_manager(self):
        with NullOutput() as output:
            output.send(MidiEvent("cc", 1, 1, 1))
        assert len(output.events) == 1


class TestPortResolution:
    PORTS = ["IAC Driver Bus 1", "DDJ-400 MIDI 1", "UM-ONE"]

    def test_matches_a_case_insensitive_name_fragment(self):
        assert MidiOutput._resolve_port("iac", self.PORTS) == 0
        assert MidiOutput._resolve_port("DDJ", self.PORTS) == 1

    def test_accepts_an_index(self):
        assert MidiOutput._resolve_port(2, self.PORTS) == 2
        assert MidiOutput._resolve_port("2", self.PORTS) == 2

    def test_ambiguous_match_takes_the_first_and_warns(self, caplog):
        assert MidiOutput._resolve_port("MIDI", ["a MIDI x", "b MIDI y"]) == 0
        assert "matches several ports" in caplog.text

    def test_unknown_name_lists_what_is_available(self):
        with pytest.raises(ValueError, match="DDJ-400 MIDI 1"):
            MidiOutput._resolve_port("nope", self.PORTS)

    def test_index_out_of_range_is_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            MidiOutput._resolve_port(9, self.PORTS)

    def test_no_ports_at_all_suggests_the_iac_driver(self):
        with pytest.raises(RuntimeError, match="IAC Driver"):
            MidiOutput._resolve_port("anything", [])
