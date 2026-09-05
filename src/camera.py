"""Camera-agnostic frame capture.

The controller must work equally well with a USB webcam sitting on the desk
(``--camera 0``) and with a ceiling-mounted RTSP camera
(``--camera rtsp://user:pass@host:554/stream1``).  Those two have opposite
failure modes:

* a USB camera delivers frames on demand and never disconnects, but OpenCV's
  internal queue adds latency if we read slower than the sensor produces;
* an RTSP stream buffers aggressively, stalls, and drops out entirely when the
  WiFi hiccups.

``CameraSource`` hides both behind one interface: an optional background grabber
thread that always keeps only the newest frame (so latency stays bounded no
matter how slow inference is), and automatic reconnection for network streams.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)

Source = Union[int, str]


def parse_source(value: str) -> Source:
    """``"0"`` -> ``0`` (device index), anything else stays a string (URL/path)."""
    text = str(value).strip()
    if text.lstrip("+-").isdigit():
        return int(text)
    return text


#: URL schemes that mean "a camera on the network", i.e. something worth
#: reconnecting to and worth reading in a frame-dropping thread.
STREAM_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://", "rtmp://")


def is_stream(source: Source) -> bool:
    """True for live network sources only.

    A local video file is deliberately *not* a stream: running off the end of a
    clip means the clip finished, and reconnecting to it forever would hang the
    controller instead of shutting it down.
    """
    if isinstance(source, int):
        return False
    return str(source).lower().startswith(STREAM_SCHEMES)


@dataclass
class Frame:
    """One captured frame plus the bookkeeping the pipeline needs."""

    image: np.ndarray          # BGR, as OpenCV delivers it
    index: int                 # monotonically increasing frame counter
    timestamp: float           # time.monotonic() at grab time, seconds
    timestamp_ms: int          # milliseconds since capture start (MediaPipe clock)

    @property
    def age(self) -> float:
        """Seconds elapsed since this frame was grabbed."""
        return time.monotonic() - self.timestamp


class CameraOpenError(RuntimeError):
    """Raised when the capture device or stream cannot be opened."""


class CameraSource:
    """Capture frames from a webcam index or an RTSP/file URL.

    Args:
        source: device index (``0``) or URL/path.
        width, height, fps: requested capture format; ignored by most streams.
        mirror: horizontally flip frames (selfie view). Recommended for a
            webcam pointed at the player, since it makes on-screen motion match
            physical motion.
        threaded: keep only the newest frame using a grabber thread. ``None``
            enables it automatically for streams, where buffering is the main
            source of latency.
        rtsp_transport: ``"tcp"`` (reliable) or ``"udp"`` (lower latency).
        reconnect_delay: seconds to wait before re-opening a dropped stream.
    """

    def __init__(
        self,
        source: Source,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        mirror: bool = True,
        threaded: Optional[bool] = None,
        rtsp_transport: str = "tcp",
        reconnect_delay: float = 2.0,
        read_timeout: float = 5.0,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self.threaded = is_stream(source) if threaded is None else bool(threaded)
        self.rtsp_transport = rtsp_transport
        self.reconnect_delay = reconnect_delay
        self.read_timeout = read_timeout

        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._new_frame = threading.Event()
        self._frame_index = 0
        self._start_time = 0.0
        self._last_ms = -1

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> "CameraSource":
        """Open the device/stream, raising :class:`CameraOpenError` on failure."""
        if is_stream(self.source) and str(self.source).startswith("rtsp"):
            # Must be set before the capture is constructed; FFmpeg reads it once.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                f"rtsp_transport;{self.rtsp_transport}",
            )

        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise CameraOpenError(
                f"could not open camera source {self.source!r}. "
                "For a webcam try another index (0/1/2); for RTSP check the URL, "
                "credentials and that the camera is reachable on the network."
            )

        # A one-frame buffer is the single most effective latency fix.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            capture.set(cv2.CAP_PROP_FPS, self.fps)

        self._capture = capture
        self._start_time = time.monotonic()
        self._stop.clear()

        if self.threaded:
            self._thread = threading.Thread(
                target=self._grab_loop, name="camera-grabber", daemon=True
            )
            self._thread.start()
        log.info(
            "camera %r opened (%dx%d, threaded=%s)",
            self.source,
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.threaded,
        )
        return self

    def close(self) -> None:
        """Stop the grabber thread and release the device."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "CameraSource":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- reading -----------------------------------------------------------
    def read(self) -> Optional[Frame]:
        """Return the newest frame, or ``None`` if the source is exhausted.

        In threaded mode this blocks until a frame newer than the previous one
        is available (bounded by ``read_timeout``); in direct mode it performs a
        blocking ``VideoCapture.read``.
        """
        if self._capture is None:
            raise RuntimeError("camera is not open; call open() first")

        if self.threaded:
            if not self._new_frame.wait(self.read_timeout):
                log.warning("no frame within %.1fs", self.read_timeout)
                return None
            with self._lock:
                self._new_frame.clear()
                return self._latest

        ok, image = self._capture.read()
        if not ok or image is None:
            if is_stream(self.source) and self._reopen():
                return self.read()
            return None
        return self._wrap(image)

    def _wrap(self, image: np.ndarray) -> Frame:
        if self.mirror:
            image = cv2.flip(image, 1)
        self._frame_index += 1
        now = time.monotonic()
        # MediaPipe's VIDEO mode requires strictly increasing timestamps.
        stamp_ms = int((now - self._start_time) * 1000.0)
        if stamp_ms <= self._last_ms:
            stamp_ms = self._last_ms + 1
        self._last_ms = stamp_ms
        return Frame(image=image, index=self._frame_index, timestamp=now, timestamp_ms=stamp_ms)

    def _grab_loop(self) -> None:
        """Background thread: always hold the most recent frame, drop the rest."""
        while not self._stop.is_set():
            capture = self._capture
            if capture is None:
                break
            ok, image = capture.read()
            if not ok or image is None:
                if is_stream(self.source) and not self._stop.is_set():
                    if self._reopen():
                        continue
                log.warning("camera %r stopped delivering frames", self.source)
                break
            frame = self._wrap(image)
            with self._lock:
                self._latest = frame
                self._new_frame.set()

    def _reopen(self) -> bool:
        """Re-open a dropped stream. Returns True when capture resumed."""
        log.warning("stream %r dropped, reconnecting in %.1fs", self.source, self.reconnect_delay)
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._stop.wait(self.reconnect_delay):
            return False
        try:
            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                capture.release()
                return False
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture = capture
            log.info("stream %r reconnected", self.source)
            return True
        except cv2.error as exc:  # pragma: no cover - depends on backend
            log.error("reconnect failed: %s", exc)
            return False
