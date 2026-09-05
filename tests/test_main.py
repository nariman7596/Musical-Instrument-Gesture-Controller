"""Command line surface and the controller's wiring."""

from __future__ import annotations

import numpy as np
import pytest

import cv2

import main as app
from src.camera import Frame
from src.hand_tracker import TrackingResult
from src.midi_mapper import ConfigError
from src.midi_output import NullOutput


def parse(argv):
    return app.build_parser().parse_args(argv)


class TestArgumentParsing:
    def test_defaults(self):
        args = parse([])
        assert args.camera == "0"
        assert args.mirror is True
        assert args.dry_run is False
        assert args.hot_reload is True
        assert args.config == app.DEFAULT_CONFIG

    def test_camera_accepts_an_index_or_a_url(self):
        assert parse(["--camera", "1"]).camera == "1"
        url = "rtsp://admin:@192.168.1.15:554/stream1"
        assert parse(["--camera", url]).camera == url

    def test_switches(self):
        args = parse(["--no-mirror", "--no-hot-reload", "--show", "--dry-run"])
        assert (args.mirror, args.hot_reload, args.show, args.dry_run) == (False, False, True, True)

    def test_rejects_an_unknown_rtsp_transport(self):
        with pytest.raises(SystemExit):
            parse(["--rtsp-transport", "carrier-pigeon"])


class TestListings:
    def test_list_features_prints_documented_names(self, capsys):
        assert app.main(["--list-features"]) == 0
        output = capsys.readouterr().out
        assert "<hand>.pinch" in output          # the table uses the <hand> template
        assert "both.hands_distance" in output
        assert "left' or 'right'" in output      # ... and explains how to expand it

    def test_list_ports_handles_a_machine_without_midi(self, capsys, monkeypatch):
        from src import midi_output

        monkeypatch.setattr(app, "list_output_ports", lambda: [])
        assert app.main(["--list-ports"]) == 0
        assert "IAC Driver" in capsys.readouterr().out

    def test_list_ports_numbers_the_ports(self, capsys, monkeypatch):
        monkeypatch.setattr(app, "list_output_ports", lambda: ["IAC Driver Bus 1", "DDJ-400"])
        assert app.main(["--list-ports"]) == 0
        out = capsys.readouterr().out
        assert "0: IAC Driver Bus 1" in out and "1: DDJ-400" in out


class TestController:
    def test_channel_override_applies_to_every_mapping(self):
        controller = app.GestureController(parse(["--dry-run", "--channel", "9"]))
        assert {mapping.channel for mapping in controller.config.mappings} == {9}

    def test_invalid_channel_is_rejected(self):
        with pytest.raises(ConfigError, match="between 1 and 16"):
            app.GestureController(parse(["--dry-run", "--channel", "99"]))

    def test_bad_config_exits_with_an_error_code(self, tmp_path, caplog):
        broken = tmp_path / "broken.json"
        broken.write_text("{", encoding="utf-8")
        assert app.main(["--config", str(broken), "--dry-run"]) == 2
        assert "not valid JSON" in caplog.text

    def test_unopenable_camera_exits_with_an_error_code(self, tmp_path, caplog):
        assert app.main(["--camera", str(tmp_path / "nope.avi"), "--dry-run"]) == 2
        assert "could not open camera source" in caplog.text

    def test_hot_reload_disabled_means_no_watcher(self):
        assert app.GestureController(parse(["--dry-run", "--no-hot-reload"])).watcher is None


class FakeCamera:
    """Hands out prepared frames, then reports the end of the source."""

    source = "fake"

    def __init__(self, frames):
        self._frames = list(frames)

    def read(self):
        return self._frames.pop(0) if self._frames else None

    def close(self):
        pass


class FakeTracker:
    """Tracker stand-in: records what it was given, finds no hands."""

    def __init__(self):
        self.calls = []

    def process(self, rgb, timestamp_ms):
        self.calls.append((rgb.shape, timestamp_ms))
        return TrackingResult([])

    def close(self):
        pass


class TestLoopWiring:
    """Drives the real loop with stand-ins, no camera and no MediaPipe."""

    @staticmethod
    def frames(count, width, height):
        import time

        return [
            Frame(np.zeros((height, width, 3), np.uint8), index + 1, time.monotonic(), index + 1)
            for index in range(count)
        ]

    def test_frame_aspect_ratio_reaches_the_feature_extractor(self, monkeypatch):
        """A 4:3 frame and a 16:9 frame must not produce the same geometry."""
        seen = []
        original = app.build_feature_vector

        def spy(hands, calibration, aspect=1.0):
            seen.append(aspect)
            return original(hands, calibration, aspect)

        monkeypatch.setattr(app, "build_feature_vector", spy)
        controller = app.GestureController(parse(["--dry-run", "--no-hot-reload"]))
        controller.midi_target = "test"
        controller._loop(FakeCamera(self.frames(3, 640, 480)), FakeTracker(), NullOutput(), None)
        assert seen == [pytest.approx(4 / 3)] * 3

    def test_loop_ends_when_the_source_runs_out(self):
        controller = app.GestureController(parse(["--dry-run", "--no-hot-reload"]))
        controller.midi_target = "test"
        tracker = FakeTracker()
        controller._loop(FakeCamera(self.frames(5, 320, 240)), tracker, NullOutput(), None)
        assert len(tracker.calls) == 5

    def test_frames_are_converted_to_rgb_for_the_tracker(self):
        controller = app.GestureController(parse(["--dry-run", "--no-hot-reload"]))
        controller.midi_target = "test"
        tracker = FakeTracker()
        controller._loop(FakeCamera(self.frames(1, 160, 120)), tracker, NullOutput(), None)
        assert tracker.calls[0][0] == (120, 160, 3)

    def test_muting_stops_output_but_not_tracking(self):
        controller = app.GestureController(parse(["--dry-run", "--no-hot-reload"]))
        controller.midi_target = "test"
        controller.muted = True
        output = NullOutput()
        tracker = FakeTracker()
        controller._loop(FakeCamera(self.frames(4, 160, 120)), tracker, output, None)
        assert output.events == [] and len(tracker.calls) == 4


class TestEndToEnd:
    """Runs the real loop over a video file, with MIDI going to the dry-run sink."""

    @pytest.fixture
    def clip(self, tmp_path):
        path = tmp_path / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
        assert writer.isOpened()
        for _ in range(12):
            writer.write(np.full((120, 160, 3), 90, dtype=np.uint8))
        writer.release()
        return path

    def test_runs_a_clip_to_completion(self, clip, capsys):
        pytest.importorskip("mediapipe", reason="tracking backend not installed")
        try:
            exit_code = app.main([
                "--camera", str(clip), "--dry-run", "--no-hot-reload", "--stats-interval", "0",
            ])
        except RuntimeError as exc:  # model download unavailable offline
            pytest.skip(f"hand landmarker model unavailable: {exc}")
        assert exit_code == 0
