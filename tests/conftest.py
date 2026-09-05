"""Shared test fixtures.

The synthetic hand below lets the geometry be tested without a camera or a
MediaPipe install: it builds landmarks in a canonical hand space (wrist at the
origin, palm one unit long, fingers pointing "up") and then places them into
image coordinates the way MediaPipe would report them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"

# Canonical geometry in palm units: MCP position and the three bone lengths.
_FINGERS = {
    "index": ((-0.30, 0.95), (0.42, 0.26, 0.20)),
    "middle": ((0.00, 1.00), (0.46, 0.28, 0.20)),
    "ring": ((0.26, 0.95), (0.42, 0.26, 0.19)),
    "pinky": ((0.48, 0.84), (0.32, 0.20, 0.17)),
}
_FINGER_ORDER = ("index", "middle", "ring", "pinky")


def _rotate(point, degrees: float):
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return (point[0] * cos - point[1] * sin, point[0] * sin + point[1] * cos)


def make_hand(
    curl: float = 0.0,
    curls=None,
    thumb: float = 1.0,
    center=(0.5, 0.5),
    scale: float = 0.25,
    rotation: float = 0.0,
    aspect: float = 1.0,
) -> np.ndarray:
    """Build a ``(21, 3)`` landmark array for a synthetic hand.

    Args:
        curl: 0.0 = every finger straight, 1.0 = fully curled (90 deg per joint).
        curls: per-finger curls as ``{"index": 0.0, ...}``, overriding ``curl``.
        thumb: 1.0 = thumb held out, 0.0 = tucked across the palm.
        center: wrist position in normalised image coordinates.
        scale: palm length as a fraction of the frame height.
        rotation: palm rotation in degrees, counter-clockwise on screen.
        aspect: frame ``width / height``; x is divided by it so that
            ``metric_landmarks(..., aspect)`` recovers the canonical shape.
    """
    per_finger = dict.fromkeys(_FINGER_ORDER, curl)
    per_finger.update(curls or {})
    points = [(0.0, 0.0)]  # 0: wrist

    # Thumb: the tip travels between "held out beside the palm" and "tucked
    # across the palm". The thumb sits on the index side, i.e. negative x.
    # CMC/MCP/IP are placed along the wrist-to-tip line with a slight outward bow.
    out_tip, tucked_tip = (-0.95, 0.95), (0.20, 0.55)
    tip = (
        tucked_tip[0] + (out_tip[0] - tucked_tip[0]) * thumb,
        tucked_tip[1] + (out_tip[1] - tucked_tip[1]) * thumb,
    )
    bow = (-tip[1] * 0.08, tip[0] * 0.08)  # perpendicular offset, so it is not a straight line
    for fraction in (0.25, 0.55, 0.80, 1.0):
        bend = math.sin(math.pi * fraction) * (1.0 - fraction * 0.5)
        points.append((tip[0] * fraction + bow[0] * bend, tip[1] * fraction + bow[1] * bend))

    for name in _FINGER_ORDER:
        (mcp_x, mcp_y), lengths = _FINGERS[name]
        finger_curl = per_finger[name]
        position = (mcp_x, mcp_y)
        chain = [position]
        heading = 0.0
        for index, length in enumerate(lengths):
            if index > 0:  # bend happens at PIP and DIP, not at the MCP
                heading -= 90.0 * finger_curl
            direction = _rotate((0.0, 1.0), heading)
            position = (position[0] + direction[0] * length, position[1] + direction[1] * length)
            chain.append(position)
        points.extend(chain)  # MCP, PIP, DIP, TIP

    landmarks = np.zeros((21, 3), dtype=np.float64)
    for index, point in enumerate(points):
        x, y = _rotate(point, rotation)
        # Canonical space has y pointing up; image coordinates have it pointing
        # down. x is pre-divided by the aspect ratio so the metric view is round.
        landmarks[index, 0] = center[0] + (x * scale) / aspect
        landmarks[index, 1] = center[1] - y * scale
    return landmarks


@pytest.fixture
def hand_factory():
    return make_hand


@pytest.fixture(scope="session")
def reference_cases():
    """Landmarks captured from real photographs, with the pose they show."""
    path = DATA_DIR / "reference_landmarks.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]
