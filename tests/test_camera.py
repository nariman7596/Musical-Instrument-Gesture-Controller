"""Camera source: argument parsing, frame delivery, mirroring and timestamps."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.camera import CameraOpenError, CameraSource, Frame, is_stream, parse_source


@pytest.fixture
def video_file(tmp_path):
    """A short video whose frames carry a recognisable left/right asymmetry."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    assert writer.isOpened(), "no MJPG encoder available in this environment"
    for index in range(10):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :10] = 255                # bright stripe on the left
        frame[0, 0] = (index, index, index)
        writer.write(frame)
    writer.release()
    return path


class TestSourceParsing:
    @pytest.mark.parametrize("text, expected", [("0", 0), ("2", 2), (" 1 ", 1)])
    def test_digits_become_a_device_index(self, text, expected):
        assert parse_source(text) == expected

    def test_urls_and_paths_stay_strings(self):
        url = "rtsp://admin:@192.168.1.15:554/stream1"
        assert parse_source(url) == url
        assert parse_source("/videos/clip.mp4") == "/videos/clip.mp4"

    def test_is_stream_only_covers_live_network_sources(self):
        """A file is not a stream: EOF must end the run, not trigger reconnects."""
        assert is_stream("rtsp://host/stream") is True
        assert is_stream("http://host/mjpeg") is True
        assert is_stream(0) is False
        assert is_stream("/videos/clip.mp4") is False


class TestFrame:
    def test_age_grows_from_the_capture_time(self):
        import time

        frame = Frame(np.zeros((2, 2, 3), np.uint8), 1, time.monotonic() - 0.5, 500)
        assert frame.age == pytest.approx(0.5, abs=0.1)


class TestCameraSource:
    def test_reads_frames_from_a_file(self, video_file):
        with CameraSource(str(video_file), mirror=False, threaded=False) as camera:
            frame = camera.read()
        assert frame is not None
        assert frame.image.shape == (48, 64, 3)
        assert frame.index == 1

    def test_returns_none_at_the_end_of_a_file(self, video_file):
        with CameraSource(str(video_file), mirror=False, threaded=False) as camera:
            count = 0
            while camera.read() is not None:
                count += 1
                assert count < 100, "read() never reported the end of the clip"
        assert count == 10

    def test_mirroring_flips_the_image(self, video_file):
        with CameraSource(str(video_file), mirror=False, threaded=False) as camera:
            straight = camera.read().image
        with CameraSource(str(video_file), mirror=True, threaded=False) as camera:
            mirrored = camera.read().image
        assert straight[:, :10].mean() > 200      # stripe starts on the left ...
        assert mirrored[:, -10:].mean() > 200     # ... and ends up on the right
        assert np.array_equal(mirrored, cv2.flip(straight, 1))

    def test_timestamps_increase_strictly(self, video_file):
        """MediaPipe's VIDEO mode rejects a timestamp that does not move forward."""
        with CameraSource(str(video_file), mirror=False, threaded=False) as camera:
            stamps = [camera.read().timestamp_ms for _ in range(10)]
        assert stamps == sorted(stamps)
        assert len(set(stamps)) == len(stamps)

    def test_frame_index_counts_up(self, video_file):
        with CameraSource(str(video_file), mirror=False, threaded=False) as camera:
            assert [camera.read().index for _ in range(5)] == [1, 2, 3, 4, 5]

    def test_threaded_mode_delivers_frames(self, video_file):
        with CameraSource(str(video_file), mirror=False, threaded=True) as camera:
            frame = camera.read()
        assert frame is not None and frame.image.shape == (48, 64, 3)

    def test_threading_defaults_to_on_for_streams_only(self):
        assert CameraSource("rtsp://host/stream").threaded is True
        assert CameraSource(0).threaded is False
        assert CameraSource("/videos/clip.mp4").threaded is False
        assert CameraSource("rtsp://host/stream", threaded=False).threaded is False

    def test_opening_a_missing_source_explains_what_to_try(self, tmp_path):
        with pytest.raises(CameraOpenError, match="webcam try another index"):
            CameraSource(str(tmp_path / "missing.avi")).open()

    def test_reading_before_open_is_an_error(self):
        with pytest.raises(RuntimeError, match="not open"):
            CameraSource(0).read()

    def test_close_is_safe_to_call_twice(self, video_file):
        camera = CameraSource(str(video_file), threaded=False).open()
        camera.close()
        camera.close()
