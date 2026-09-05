"""MediaPipe Hands wrapper.

MediaPipe ships two generations of the hand API and which one you get depends on
the installed version:

* **Tasks API** (``mediapipe.tasks.python.vision.HandLandmarker``) — the current
  one, and the *only* one in MediaPipe >= 1.0.  It needs a ``.task`` model file,
  which this module downloads and caches on first run.
* **Legacy solutions API** (``mediapipe.solutions.hands``) — present up to
  MediaPipe 0.10.x, bundles its own model.

``HandTracker`` auto-detects whichever is available so the project keeps working
across both, and normalises the output into plain ``(21, 3)`` NumPy arrays keyed
by ``"left"``/``"right"``.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .gesture_features import LANDMARK_COUNT

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"


def default_model_dir() -> Path:
    """Cache directory for downloaded model assets (override with ``GMC_MODEL_DIR``)."""
    env = os.environ.get("GMC_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "gesture-midi-controller"


def ensure_model(path: Optional[Path] = None) -> Path:
    """Return a local ``hand_landmarker.task``, downloading it once if needed."""
    target = Path(path).expanduser() if path else default_model_dir() / MODEL_FILENAME
    if target.is_file() and target.stat().st_size > 0:
        return target
    if path is not None and not target.parent.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {target.parent}")

    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading hand landmarker model to %s (~7.5 MB, once)", target)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download the MediaPipe hand model from {MODEL_URL}: {exc}. "
            "Download it manually and pass --model /path/to/hand_landmarker.task"
        ) from exc
    partial.replace(target)
    return target


@dataclass
class HandObservation:
    """One detected hand."""

    handedness: str            # "left" or "right", from the player's point of view
    score: float               # handedness classification confidence
    landmarks: np.ndarray      # (21, 3) normalised image coordinates


@dataclass
class TrackingResult:
    """Per-frame tracking output."""

    observations: List[HandObservation] = field(default_factory=list)

    @property
    def hands(self) -> Dict[str, Optional[np.ndarray]]:
        """``{"left": (21,3) | None, "right": (21,3) | None}``.

        If the same label is reported twice (it happens when two hands overlap),
        the higher-scoring observation wins.
        """
        best: Dict[str, HandObservation] = {}
        for observation in self.observations:
            current = best.get(observation.handedness)
            if current is None or observation.score > current.score:
                best[observation.handedness] = observation
        return {side: (best[side].landmarks if side in best else None) for side in ("left", "right")}

    def __len__(self) -> int:
        return len(self.observations)


#: When to flip MediaPipe's handedness label, per backend.
#:
#: The two APIs disagree, and silently: feed both the same photograph and they
#: name opposite hands.  The legacy API documents that it assumes a *mirrored*
#: (selfie) input, so an un-mirrored feed needs the swap.  The Tasks API labels
#: the anatomically correct hand for an *un-mirrored* frame, so it needs the
#: opposite rule.  Verified against MediaPipe's own ``left_hands.jpg`` /
#: ``right_hands.jpg`` samples in both mirror modes; see
#: ``tests/test_hand_tracker.py``.
#:
#: Getting this wrong is quiet and infuriating: every right-hand mapping
#: responds to the left hand and nothing looks broken.
_SWAP_HANDEDNESS = {
    "tasks": lambda mirrored: mirrored,
    "legacy": lambda mirrored: not mirrored,
}


def resolve_handedness(label: str, backend: str, mirrored: bool) -> str:
    """Normalise a raw MediaPipe handedness label to the player's real hand.

    Args:
        label: what MediaPipe reported (``"Left"``/``"Right"``, any case).
        backend: ``"tasks"`` or ``"legacy"`` — they disagree, see
            :data:`_SWAP_HANDEDNESS`.
        mirrored: whether the frame was flipped horizontally before tracking.

    Returns:
        ``"left"`` or ``"right"``, from the player's point of view.
    """
    if backend not in _SWAP_HANDEDNESS:
        raise ValueError(f"unknown backend {backend!r}")
    side = "left" if str(label).lower().startswith("l") else "right"
    if _SWAP_HANDEDNESS[backend](mirrored):
        side = "right" if side == "left" else "left"
    return side


class HandTracker:
    """Detect hands in RGB frames and return normalised landmarks.

    Args:
        max_hands: how many hands to track (2 enables the two-hand features).
        min_detection_confidence / min_tracking_confidence: MediaPipe thresholds.
        model_path: explicit ``.task`` file (Tasks backend only).
        mirrored: set to ``True`` when the frames handed to :meth:`process` have
            already been flipped horizontally (selfie view).  Together with the
            backend this decides whether the handedness labels need swapping —
            see :data:`_SWAP_HANDEDNESS`.
        backend: ``"auto"``, ``"tasks"`` or ``"legacy"``.
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[Path] = None,
        mirrored: bool = True,
        backend: str = "auto",
    ) -> None:
        self.max_hands = max_hands
        self.mirrored = mirrored
        self._closed = False

        import mediapipe as mp  # imported lazily: heavy, and optional for tests

        self._mp = mp
        chosen = backend
        if chosen == "auto":
            chosen = "legacy" if hasattr(mp, "solutions") else "tasks"

        if chosen == "tasks":
            self._init_tasks(mp, model_path, min_detection_confidence, min_tracking_confidence)
        elif chosen == "legacy":
            self._init_legacy(mp, min_detection_confidence, min_tracking_confidence)
        else:
            raise ValueError(f"unknown backend {backend!r}; use auto, tasks or legacy")
        self.backend = chosen
        log.info(
            "hand tracker ready (backend=%s, max_hands=%d, mirrored=%s)",
            chosen, max_hands, mirrored,
        )

    # -- backends ----------------------------------------------------------
    def _init_tasks(self, mp, model_path, detection_conf, tracking_conf) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._vision = vision
        model = ensure_model(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def _init_legacy(self, mp, detection_conf, tracking_conf) -> None:
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "the legacy mediapipe.solutions API is not available in "
                f"mediapipe {getattr(mp, '__version__', '?')}; use --tracker-backend tasks"
            )
        self._landmarker = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            model_complexity=0,  # fastest variant; plenty for gesture control
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    # -- inference ---------------------------------------------------------
    def process(self, rgb_frame: np.ndarray, timestamp_ms: int = 0) -> TrackingResult:
        """Run inference on one **RGB** frame (``timestamp_ms`` must increase)."""
        if self._closed:
            raise RuntimeError("tracker is closed")
        if self.backend == "tasks":
            return self._process_tasks(rgb_frame, timestamp_ms)
        return self._process_legacy(rgb_frame)

    def _process_tasks(self, rgb_frame: np.ndarray, timestamp_ms: int) -> TrackingResult:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        observations: List[HandObservation] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            category = handedness[0]
            observations.append(
                self._make_observation(
                    [(p.x, p.y, p.z) for p in landmarks],
                    category.category_name,
                    float(category.score),
                )
            )
        return TrackingResult(observations)

    def _process_legacy(self, rgb_frame: np.ndarray) -> TrackingResult:
        result = self._landmarker.process(rgb_frame)
        observations: List[HandObservation] = []
        landmark_sets = result.multi_hand_landmarks or []
        handedness_sets = result.multi_handedness or []
        for landmarks, handedness in zip(landmark_sets, handedness_sets):
            category = handedness.classification[0]
            observations.append(
                self._make_observation(
                    [(p.x, p.y, p.z) for p in landmarks.landmark],
                    category.label,
                    float(category.score),
                )
            )
        return TrackingResult(observations)

    def _make_observation(self, points, label: str, score: float) -> HandObservation:
        array = np.asarray(points, dtype=np.float64)
        if array.shape != (LANDMARK_COUNT, 3):
            raise ValueError(f"unexpected landmark shape {array.shape}")
        side = resolve_handedness(label, self.backend, self.mirrored)
        return HandObservation(handedness=side, score=score, landmarks=array)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._landmarker, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
