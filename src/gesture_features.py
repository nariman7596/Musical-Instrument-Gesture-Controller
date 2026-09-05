"""Landmark -> feature vector conversion.

MediaPipe Hands returns 21 landmarks per hand in *normalised image coordinates*
(``x`` and ``y`` in ``[0, 1]``, ``y`` growing downwards, ``z`` a relative depth
estimate).  Raw landmarks are useless as MIDI sources: they depend on where the
hand is, how big it is and how far from the camera it sits.

This module turns them into a flat dictionary of **named, scale-invariant
features, each normalised to ``[0, 1]``** so that ``midi_mapper`` only has to
deal with plain floats.  Every distance is divided by the *palm size*
(wrist -> middle-finger MCP), which makes the features largely independent of
the distance between hand and camera.

Only ``x``/``y`` are used for the geometric features: MediaPipe's ``z`` is too
noisy frame-to-frame to drive a filter cutoff without audible zipper noise.
``z`` is exposed indirectly through ``depth``, which is derived from the
apparent palm size instead.

Two subtleties worth knowing:

* **Aspect ratio.** MediaPipe normalises ``x`` and ``y`` independently, so on a
  16:9 frame one unit of ``x`` is 1.78x longer than one unit of ``y``.  Every
  distance and angle here is therefore computed on landmarks whose ``x`` has
  been multiplied by the frame aspect ratio; pass it as ``aspect`` or circles
  come out as ellipses and a pinch reads differently across the frame.
* **Finger curl is measured as a joint angle**, not as a tip-to-knuckle
  distance.  Distances shrink dramatically when a finger points at the camera,
  which made an outstretched hand read as a fist; the summed bend of the PIP and
  DIP joints stays near 0 deg for a straight finger and passes 150 deg for a
  curled one no matter which way the hand faces.

Feature keys are namespaced by hand: ``right.pinch``, ``left.fist``,
``both.hands_distance`` — exactly the strings used in the JSON mapping file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Landmark indices (MediaPipe Hands topology)
# --------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

LANDMARK_COUNT = 21
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

#: Bone connections, used by the visualiser to draw the skeleton.
HAND_CONNECTIONS: Sequence[tuple] = (
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                  # palm base
)


@dataclass(frozen=True)
class FeatureCalibration:
    """Input ranges used to normalise raw geometry into ``[0, 1]``.

    The defaults were chosen for a hand roughly 40-70 cm from a webcam.  Every
    field can be overridden from the ``"calibration"`` block of the JSON config,
    so a user with unusually small hands or a wide-angle lens can re-scale the
    controller without touching the code.

    Each field is ``(low, high)`` in *palm units* (multiples of the
    wrist-to-middle-MCP distance) unless documented otherwise.
    """

    #: Apparent palm size in normalised image units -> ``depth`` feature.
    palm_size: tuple = (0.08, 0.38)
    #: Thumb-tip to index-tip distance. Small = pinched.
    pinch: tuple = (0.15, 1.20)
    #: Mean fingertip-to-wrist distance. Small = fist.
    openness: tuple = (0.80, 2.05)
    #: Mean distance between neighbouring fingertips.
    finger_spread: tuple = (0.20, 0.60)
    #: Palm roll in degrees, 0 deg = fingers up / palm level.
    roll_degrees: tuple = (-90.0, 90.0)
    #: Summed PIP + DIP bend in degrees -> extension. Inverted on purpose:
    #: a straight finger bends ~0-30 deg, a curled one 140-240 deg.
    finger_bend_degrees: tuple = (175.0, 20.0)
    #: The thumb has no useful curl angle, so it is measured as the
    #: thumb-tip-to-pinky-MCP distance: it collapses when tucked into the palm.
    thumb_extension: tuple = (0.45, 1.15)
    #: Wrist-to-wrist distance in aspect-corrected image units (two-hand).
    hands_distance: tuple = (0.15, 1.00)
    #: Signed vertical wrist offset in normalised image units (two-hand).
    hands_vertical_delta: tuple = (-0.45, 0.45)
    #: Horizontal wrist spread in aspect-corrected image units (two-hand).
    hands_spread: tuple = (0.05, 0.95)
    #: Wrist speed in image units per second -> the ``speed`` feature.
    hand_speed: tuple = (0.05, 1.60)

    #: A finger counts as "extended" above this normalised value ...
    extended_threshold: float = 0.65
    #: ... and as "curled" below this one.
    curled_threshold: float = 0.35

    @classmethod
    def from_dict(cls, data: Optional[Mapping]) -> "FeatureCalibration":
        """Build a calibration from a (partial) mapping, ignoring unknown keys."""
        if not data:
            return cls()
        known = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key not in known:
                raise ValueError(
                    f"unknown calibration key {key!r}; "
                    f"expected one of {sorted(known)}"
                )
            if isinstance(value, (list, tuple)):
                if len(value) != 2:
                    raise ValueError(f"calibration {key!r} must be [low, high]")
                kwargs[key] = (float(value[0]), float(value[1]))
            else:
                kwargs[key] = float(value)
        return cls(**kwargs)


DEFAULT_CALIBRATION = FeatureCalibration()


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------
def clamp01(value: float) -> float:
    """Clamp to ``[0, 1]`` (NaN becomes ``0.0``)."""
    if value != value:  # NaN
        return 0.0
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


def normalise(value: float, low: float, high: float) -> float:
    """Map ``value`` from ``[low, high]`` onto ``[0, 1]``, clamped.

    Works with an inverted range (``low > high``) as well, which is handy for
    features where "more" should mean "less".
    """
    if high == low:
        return 0.0
    return clamp01((value - low) / (high - low))


def metric_landmarks(landmarks, aspect: float = 1.0) -> np.ndarray:
    """``(21, 2)`` xy coordinates corrected for the frame aspect ratio.

    MediaPipe maps the frame width to ``x`` in ``[0, 1]`` and the height to
    ``y`` in ``[0, 1]`` independently, so distances computed on the raw values
    are stretched horizontally.  Multiplying ``x`` by ``width / height`` restores
    a common unit (the frame height) and makes every distance below isotropic.
    """
    xy = np.asarray(landmarks, dtype=np.float64)[:, :2].copy()
    xy[:, 0] *= float(aspect)
    return xy


def _dist(xy: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(xy[a] - xy[b]))


def _joint_angle(xy: np.ndarray, a: int, b: int, c: int) -> float:
    """Direction change at ``b`` along the path ``a -> b -> c``, in degrees.

    ``0`` means the three points are collinear (a straight finger segment).
    """
    first = xy[b] - xy[a]
    second = xy[c] - xy[b]
    norms = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norms < 1e-12:
        return 0.0
    cosine = float(np.dot(first, second) / norms)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def as_landmark_array(landmarks) -> np.ndarray:
    """Coerce ``landmarks`` to a validated ``(21, 3)`` float array."""
    array = np.asarray(landmarks, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != LANDMARK_COUNT or array.shape[1] < 2:
        raise ValueError(
            f"expected landmarks of shape ({LANDMARK_COUNT}, 3), got {array.shape}"
        )
    if array.shape[1] == 2:  # tolerate 2-D input by padding z with zeros
        array = np.hstack([array, np.zeros((LANDMARK_COUNT, 1))])
    return array[:, :3]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def palm_size(landmarks, aspect: float = 1.0) -> float:
    """Wrist -> middle-MCP distance: the scale reference for the whole module.

    Falls back to a tiny epsilon so a degenerate hand can never divide by zero.
    """
    xy = metric_landmarks(landmarks, aspect)
    return max(_dist(xy, WRIST, MIDDLE_MCP), 1e-6)


def palm_roll_degrees(landmarks, handedness: str = "right", aspect: float = 1.0) -> float:
    """Roll of the palm around the camera axis, in degrees.

    ``0`` means the knuckle line is horizontal (fingers pointing up), positive
    means the hand is rotated counter-clockwise *as seen on screen*.  The
    knuckle vector is flipped for the left hand so that both hands report the
    same sign for the same physical gesture.
    """
    xy = metric_landmarks(landmarks, aspect)
    start, end = INDEX_MCP, PINKY_MCP
    if handedness.lower().startswith("l"):
        start, end = PINKY_MCP, INDEX_MCP
    vector = xy[end] - xy[start]
    # Screen y grows downwards; negate it so that positive == counter-clockwise.
    angle = math.degrees(math.atan2(-float(vector[1]), float(vector[0])))
    # Wrap into [-180, 180] and then fold to [-90, 90]: a palm rolled past
    # vertical is reported as saturated rather than wrapping around.
    if angle > 90.0:
        angle = 180.0 - angle
    elif angle < -90.0:
        angle = -180.0 - angle
    return angle


#: MCP, PIP, DIP, TIP of each curling finger (the thumb is handled separately).
CURL_CHAINS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


def finger_bend_degrees(landmarks, aspect: float = 1.0) -> Dict[str, float]:
    """Summed PIP + DIP bend per finger, in degrees.

    Roughly 0-30 deg for a straight finger and 140-240 deg for a curled one,
    measured on real hands across a range of poses.
    """
    xy = metric_landmarks(landmarks, aspect)
    return {
        name: _joint_angle(xy, mcp, pip, dip) + _joint_angle(xy, pip, dip, tip)
        for name, (mcp, pip, dip, tip) in CURL_CHAINS.items()
    }


def finger_extensions(
    landmarks, calib: FeatureCalibration = DEFAULT_CALIBRATION, aspect: float = 1.0
) -> Dict[str, float]:
    """Per-finger extension in ``[0, 1]`` (0 = fully curled, 1 = straight).

    The four fingers use their joint bend, which survives foreshortening: a hand
    pointing at the lens still reads as open.  The thumb barely bends at all when
    it tucks, so it is measured as its distance to the pinky knuckle instead.
    """
    xy = metric_landmarks(landmarks, aspect)
    scale = max(_dist(xy, WRIST, MIDDLE_MCP), 1e-6)
    bends = finger_bend_degrees(landmarks, aspect)

    extensions = {
        name: normalise(bend, *calib.finger_bend_degrees) for name, bend in bends.items()
    }
    extensions["thumb"] = normalise(
        _dist(xy, THUMB_TIP, PINKY_MCP) / scale, *calib.thumb_extension
    )
    return {name: extensions[name] for name in FINGER_NAMES}


# --------------------------------------------------------------------------
# Per-hand feature vector
# --------------------------------------------------------------------------
def hand_features(
    landmarks,
    handedness: str = "right",
    calib: FeatureCalibration = DEFAULT_CALIBRATION,
    aspect: float = 1.0,
) -> Dict[str, float]:
    """Continuous + gate features for a single hand, all in ``[0, 1]``.

    ``handedness`` is only used to keep ``roll`` consistent between hands;
    ``aspect`` is the frame's ``width / height``.
    """
    lm = as_landmark_array(landmarks)
    xy = metric_landmarks(lm, aspect)
    scale = max(_dist(xy, WRIST, MIDDLE_MCP), 1e-6)

    # Position features stay in raw normalised coordinates: they describe where
    # the hand is in the picture, not how big it is.
    wrist_x, wrist_y = float(lm[WRIST, 0]), float(lm[WRIST, 1])
    fingers = finger_extensions(lm, calib, aspect)

    # Mean fingertip distance to the wrist -> how "big" the hand looks.
    tips = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    openness_raw = float(np.mean([_dist(xy, WRIST, tip) for tip in tips])) / scale

    # Lateral gaps between neighbouring fingertips -> how spread the fan is.
    neighbours = ((INDEX_TIP, MIDDLE_TIP), (MIDDLE_TIP, RING_TIP), (RING_TIP, PINKY_TIP))
    spread_raw = float(np.mean([_dist(xy, a, b) for a, b in neighbours])) / scale

    pinch_raw = _dist(xy, THUMB_TIP, INDEX_TIP) / scale

    four = [fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]]
    extended = [value >= calib.extended_threshold for value in four]
    curled = [value <= calib.curled_threshold for value in four]
    index_points_up = float(lm[INDEX_TIP, 1]) < float(lm[INDEX_MCP, 1])

    features: Dict[str, float] = {
        "present": 1.0,
        # --- position in frame -------------------------------------------
        # 1.0 = top of frame, so "raise your hand" always means "more".
        "height": clamp01(1.0 - wrist_y),
        # 0.0 = left edge of the *displayed* image, 1.0 = right edge.
        "x": clamp01(wrist_x),
        # Apparent palm size: 1.0 = hand close to the lens.
        "depth": normalise(scale, *calib.palm_size),
        # --- shape --------------------------------------------------------
        # 1.0 = thumb and index touching.
        "pinch": 1.0 - normalise(pinch_raw, *calib.pinch),
        "openness": normalise(openness_raw, *calib.openness),
        "finger_spread": normalise(spread_raw, *calib.finger_spread),
        # 0.5 = level palm, 0.0/1.0 = rolled +-90 deg.
        "roll": normalise(palm_roll_degrees(lm, handedness, aspect), *calib.roll_degrees),
        # --- discrete poses (0.0 / 1.0, smoothed into ramps downstream) ----
        "fist": float(all(curled)),
        "open_palm": float(all(extended) and fingers["thumb"] >= 0.5),
        "point_up": float(extended[0] and all(curled[1:]) and index_points_up),
        "peace": float(extended[0] and extended[1] and curled[2] and curled[3]),
        # How many fingers are up, as 0.0-1.0 in fifths. Discrete enough to
        # choose between things (a chord, an octave, a pattern) with one hand.
        "finger_count": sum(
            1 for value in fingers.values() if value >= calib.extended_threshold
        ) / 5.0,
    }
    features.update({f"{name}_extension": value for name, value in fingers.items()})
    return features


def two_hand_features(
    left,
    right,
    calib: FeatureCalibration = DEFAULT_CALIBRATION,
    aspect: float = 1.0,
) -> Dict[str, float]:
    """Relative features that only exist while *both* hands are visible."""
    left_wrist = metric_landmarks(as_landmark_array(left), aspect)[WRIST]
    right_wrist = metric_landmarks(as_landmark_array(right), aspect)[WRIST]

    distance = float(np.linalg.norm(left_wrist - right_wrist))
    # Screen y grows downwards: left hand higher than right -> positive.
    vertical = float(right_wrist[1] - left_wrist[1])
    horizontal = float(abs(right_wrist[0] - left_wrist[0]))

    return {
        "present": 1.0,
        # Theremin-style: pull the hands apart for "more".
        "hands_distance": normalise(distance, *calib.hands_distance),
        # 0.5 = wrists level; used for pitch bend, hence the centred range.
        "hands_vertical_delta": normalise(vertical, *calib.hands_vertical_delta),
        "hands_spread": normalise(horizontal, *calib.hands_spread),
    }


def build_feature_vector(
    hands: Mapping[str, Optional[np.ndarray]],
    calib: FeatureCalibration = DEFAULT_CALIBRATION,
    aspect: float = 1.0,
) -> Dict[str, float]:
    """Namespaced feature dictionary for a frame.

    ``hands`` maps ``"left"``/``"right"`` to a ``(21, 3)`` landmark array or
    ``None``.  Features of a hand that is not currently visible are **omitted**
    rather than zeroed — the mapper then holds the last MIDI value instead of
    slamming every controller to 0 when tracking blinks for a frame.  Only
    ``<hand>.present`` is always reported, so a mapping can react to a hand
    leaving the frame on purpose.
    """
    vector: Dict[str, float] = {}
    for side in ("left", "right"):
        landmarks = hands.get(side)
        if landmarks is None:
            vector[f"{side}.present"] = 0.0
            continue
        for key, value in hand_features(landmarks, side, calib, aspect).items():
            vector[f"{side}.{key}"] = value

    left, right = hands.get("left"), hands.get("right")
    if left is None or right is None:
        vector["both.present"] = 0.0
    else:
        for key, value in two_hand_features(left, right, calib, aspect).items():
            vector[f"both.{key}"] = value
    return vector


#: Human-readable description of every feature, used by ``main.py --list-features``
#: and by the README table.  ``<hand>`` is ``left`` or ``right``.
FEATURE_DOCS = {
    "<hand>.present": "1.0 while the hand is tracked, 0.0 otherwise",
    "<hand>.height": "wrist height in frame (1.0 = top)",
    "<hand>.x": "wrist horizontal position (0.0 = left edge)",
    "<hand>.depth": "apparent palm size (1.0 = closest to camera)",
    "<hand>.pinch": "thumb-index pinch (1.0 = fingers touching)",
    "<hand>.openness": "fist (0.0) to open hand (1.0)",
    "<hand>.finger_spread": "gap between neighbouring fingertips",
    "<hand>.roll": "palm roll, 0.5 = level, 0.0/1.0 = +-90 deg",
    "<hand>.fist": "gate: all four fingers curled",
    "<hand>.open_palm": "gate: all fingers extended",
    "<hand>.point_up": "gate: index finger pointing up, others curled",
    "<hand>.peace": "gate: index + middle extended, ring + pinky curled",
    "<hand>.finger_count": "extended fingers, in fifths (0.0, 0.2 ... 1.0)",
    "<hand>.speed": "how fast the hand is moving (needs MotionTracker)",
    "<hand>.thumb_extension": "thumb extension, 0.0 = tucked",
    "<hand>.index_extension": "index extension, 0.0 = curled",
    "<hand>.middle_extension": "middle extension, 0.0 = curled",
    "<hand>.ring_extension": "ring extension, 0.0 = curled",
    "<hand>.pinky_extension": "pinky extension, 0.0 = curled",
    "both.present": "1.0 while both hands are tracked",
    "both.hands_distance": "wrist-to-wrist distance (theremin axis)",
    "both.hands_vertical_delta": "vertical wrist offset, 0.5 = level",
    "both.hands_spread": "horizontal wrist spread (stereo width)",
}


class MotionTracker:
    """Adds features that need memory of where the hands were a frame ago.

    Everything else in this module is a pure function of one frame, which keeps
    it testable — but *speed* is what separates a hand placed on a note from a
    hand thrown at it, and that needs the previous frame. Keeping the state in
    one small object leaves the rest of the module pure.
    """

    def __init__(self, calib: FeatureCalibration = DEFAULT_CALIBRATION, smoothing: float = 0.4):
        self.calib = calib
        self.smoothing = smoothing
        self._previous: Dict[str, tuple] = {}
        self._speed: Dict[str, float] = {}

    def update(
        self,
        hands: Mapping[str, Optional[np.ndarray]],
        vector: Dict[str, float],
        timestamp: float,
        aspect: float = 1.0,
    ) -> Dict[str, float]:
        """Add ``<hand>.speed`` to ``vector`` in place, and return it."""
        for side in ("left", "right"):
            landmarks = hands.get(side)
            if landmarks is None:
                self._previous.pop(side, None)
                self._speed.pop(side, None)
                continue

            wrist = metric_landmarks(as_landmark_array(landmarks), aspect)[WRIST]
            previous = self._previous.get(side)
            self._previous[side] = (wrist, timestamp)
            if previous is None:
                continue

            last_wrist, last_time = previous
            elapsed = timestamp - last_time
            if not 1e-4 < elapsed < 1.0:      # a stale or duplicated frame
                continue

            raw = float(np.linalg.norm(wrist - last_wrist)) / elapsed
            # Speed is spiky by nature; smooth it or every mapping fed from it
            # fires on single-frame noise.
            smoothed = self._speed.get(side)
            value = raw if smoothed is None else smoothed + (raw - smoothed) * self.smoothing
            self._speed[side] = value
            vector[f"{side}.speed"] = normalise(value, *self.calib.hand_speed)
        return vector

    def reset(self) -> None:
        self._previous.clear()
        self._speed.clear()


#: Features supplied by :class:`MotionTracker` rather than by a single frame.
MOTION_FEATURES = ("left.speed", "right.speed")


def feature_names() -> Sequence[str]:
    """All feature keys that can appear in a frame vector."""
    names = []
    for template in FEATURE_DOCS:
        if template.startswith("<hand>"):
            names.extend(template.replace("<hand>", side) for side in ("left", "right"))
        else:
            names.append(template)
    return sorted(names)
