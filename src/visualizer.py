"""On-screen feedback: hand skeleton + live CC meters.

Playing an invisible instrument is hard without feedback — you cannot tell
whether the filter stopped moving because you stopped moving or because
tracking dropped.  The overlay answers that at a glance: the skeleton shows
what the tracker sees, and every mapping gets a bar showing the value that was
last sent, dimmed when its feature is not currently tracked.

Rendering is plain OpenCV drawing so it costs ~1 ms per frame and can be turned
off entirely with ``--no-show`` for the lowest possible latency.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from .gesture_features import HAND_CONNECTIONS

log = logging.getLogger(__name__)

# BGR colours.
COLOR_RIGHT = (120, 220, 255)   # amber
COLOR_LEFT = (255, 190, 120)    # blue
COLOR_BONE = (230, 230, 230)
COLOR_PANEL = (24, 24, 28)
COLOR_TEXT = (240, 240, 240)
COLOR_DIM = (120, 120, 128)
COLOR_BAR = (110, 230, 140)
COLOR_BAR_ACTIVE = (90, 170, 255)
COLOR_WARN = (80, 120, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def gui_available() -> bool:
    """Whether OpenCV can actually open a window here.

    Checked up front because a headless Linux box does not raise a catchable
    error: the Qt backend calls ``abort()`` and takes the process with it.
    ``opencv-python-headless`` has no ``imshow`` at all, and macOS (Cocoa) needs
    no display variable.
    """
    if not hasattr(cv2, "imshow"):
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


@dataclass
class HudStats:
    """Numbers shown in the header of the overlay."""

    fps: float = 0.0
    latency_ms: float = 0.0
    hands: int = 0
    midi_target: str = ""
    config_name: str = ""
    muted: bool = False
    messages_per_second: float = 0.0


class Visualizer:
    """Draws the overlay and (optionally) owns the preview window."""

    def __init__(
        self,
        window_name: str = "Gesture MIDI Controller",
        show_landmarks: bool = True,
        panel_width: int = 300,
    ) -> None:
        self.window_name = window_name
        self.show_landmarks = show_landmarks
        self.panel_width = panel_width
        self._window_ready = False
        self._gui_available = gui_available()
        if not self._gui_available:
            log.info("no display available; running without the preview window")

    # -- drawing -----------------------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        hands: Dict[str, Optional[np.ndarray]],
        states: Sequence,
        stats: HudStats,
    ) -> np.ndarray:
        """Return a copy of ``frame`` with skeleton, meters and header drawn."""
        canvas = frame.copy()
        if self.show_landmarks:
            for side, landmarks in hands.items():
                if landmarks is not None:
                    self._draw_hand(canvas, landmarks, side)
        self._draw_panel(canvas, states)
        self._draw_header(canvas, stats)
        return canvas

    def _draw_hand(self, canvas: np.ndarray, landmarks: np.ndarray, side: str) -> None:
        height, width = canvas.shape[:2]
        points = [
            (int(x * width), int(y * height)) for x, y in landmarks[:, :2]
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(canvas, points[start], points[end], COLOR_BONE, 2, cv2.LINE_AA)
        colour = COLOR_LEFT if side == "left" else COLOR_RIGHT
        for index, point in enumerate(points):
            radius = 6 if index in (0, 4, 8, 12, 16, 20) else 3
            cv2.circle(canvas, point, radius, colour, -1, cv2.LINE_AA)
        cv2.putText(canvas, side.upper(), (points[0][0] - 20, points[0][1] + 28),
                    FONT, 0.5, colour, 1, cv2.LINE_AA)

    def _draw_panel(self, canvas: np.ndarray, states: Sequence) -> None:
        if not len(states):
            return
        height, width = canvas.shape[:2]
        panel_width = min(self.panel_width, width // 2)
        row_height = 26
        top = 56
        panel_height = min(height - top - 8, row_height * len(states) + 16)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (width - panel_width - 8, top),
                      (width - 8, top + panel_height), COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

        x0 = width - panel_width
        for row, state in enumerate(states):
            y = top + 24 + row * row_height
            if y > top + panel_height - 4:
                break
            text_colour = COLOR_TEXT if state.tracked else COLOR_DIM
            cv2.putText(canvas, state.name[:18], (x0, y - 10), FONT, 0.42,
                        text_colour, 1, cv2.LINE_AA)
            value_text = f"{state.label}: {state.value}"
            cv2.putText(canvas, value_text, (x0 + 168, y - 10), FONT, 0.42,
                        text_colour, 1, cv2.LINE_AA)

            bar_width = panel_width - 24
            cv2.rectangle(canvas, (x0, y - 6), (x0 + bar_width, y), (60, 60, 66), -1)
            filled = int(bar_width * max(0.0, min(1.0, state.normalised)))
            if filled > 0:
                colour = COLOR_BAR_ACTIVE if state.active else COLOR_BAR
                if not state.tracked:
                    colour = COLOR_DIM
                cv2.rectangle(canvas, (x0, y - 6), (x0 + filled, y), colour, -1)

    def _draw_header(self, canvas: np.ndarray, stats: HudStats) -> None:
        width = canvas.shape[1]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, 48), COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

        left = (
            f"{stats.fps:5.1f} fps   {stats.latency_ms:5.1f} ms   "
            f"hands {stats.hands}   {stats.messages_per_second:4.0f} msg/s"
        )
        cv2.putText(canvas, left, (12, 20), FONT, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        right = f"{stats.midi_target}   [{stats.config_name}]"
        cv2.putText(canvas, right, (12, 40), FONT, 0.45, COLOR_DIM, 1, cv2.LINE_AA)
        if stats.muted:
            cv2.putText(canvas, "MUTED", (width - 90, 30), FONT, 0.7, COLOR_WARN, 2, cv2.LINE_AA)

    # -- window ------------------------------------------------------------
    def show(self, image: np.ndarray) -> int:
        """Display a frame and return the key pressed (-1 for none).

        Returns ``-1`` and disables itself if OpenCV has no GUI support (e.g.
        ``opencv-python-headless`` on a server), rather than crashing the run.
        """
        if not self._gui_available:
            return -1
        try:
            if not self._window_ready:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._window_ready = True
            cv2.imshow(self.window_name, image)
            return cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            log.warning("preview window unavailable (%s); continuing without it", exc)
            self._gui_available = False
            return -1

    def close(self) -> None:
        if self._window_ready:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:  # pragma: no cover - platform dependent
                pass
            self._window_ready = False
