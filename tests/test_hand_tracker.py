"""Tracker plumbing: handedness conventions, result shaping, model resolution.

None of this needs MediaPipe — the parts that do are exercised by the
end-to-end test in ``test_main.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.hand_tracker import (
    HandObservation,
    TrackingResult,
    ensure_model,
    resolve_handedness,
)


class TestHandedness:
    """The two MediaPipe APIs label hands oppositely; this is the adapter.

    Ground truth comes from MediaPipe's own ``left_hands.jpg`` /
    ``right_hands.jpg`` samples, run through both backends in both mirror
    modes. Getting this wrong is silent: every right-hand mapping would follow
    the player's left hand and nothing would look broken.
    """

    @pytest.mark.parametrize("mirrored", [True, False])
    @pytest.mark.parametrize("backend", ["tasks", "legacy"])
    def test_the_players_real_hand_is_reported(self, backend, mirrored):
        # What each backend reports for a physically right hand, as measured.
        raw = {
            ("tasks", True): "Left",     # frame was flipped before tracking
            ("tasks", False): "Right",
            ("legacy", True): "Right",
            ("legacy", False): "Left",
        }[(backend, mirrored)]
        assert resolve_handedness(raw, backend, mirrored) == "right"

    def test_the_two_backends_need_opposite_swaps(self):
        """A regression guard: one rule for both backends is wrong for one."""
        assert resolve_handedness("Right", "tasks", True) != resolve_handedness(
            "Right", "legacy", True
        )

    def test_labels_are_case_insensitive_and_normalised(self):
        assert resolve_handedness("LEFT", "tasks", False) == "left"
        assert resolve_handedness("right", "tasks", False) == "right"

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="unknown backend"):
            resolve_handedness("Left", "guesswork", True)


class TestTrackingResult:
    @staticmethod
    def observation(side, score=0.9, seed=0):
        return HandObservation(side, score, np.full((21, 3), float(seed)))

    def test_hands_are_keyed_by_side(self):
        result = TrackingResult([self.observation("left"), self.observation("right")])
        hands = result.hands
        assert hands["left"] is not None and hands["right"] is not None
        assert len(result) == 2

    def test_a_missing_hand_is_none(self):
        hands = TrackingResult([self.observation("right")]).hands
        assert hands["left"] is None
        assert hands["right"] is not None

    def test_empty_result(self):
        assert TrackingResult().hands == {"left": None, "right": None}
        assert len(TrackingResult()) == 0

    def test_duplicate_labels_keep_the_confident_one(self):
        """Overlapping hands sometimes both come back with the same label."""
        result = TrackingResult([
            self.observation("right", score=0.55, seed=1),
            self.observation("right", score=0.95, seed=2),
        ])
        assert result.hands["right"][0, 0] == 2.0
        assert result.hands["left"] is None


class TestModelResolution:
    def test_an_existing_file_is_used_as_is(self, tmp_path):
        model = tmp_path / "hand_landmarker.task"
        model.write_bytes(b"not really a model, but non-empty")
        assert ensure_model(model) == model

    def test_a_missing_directory_is_reported_rather_than_downloaded_into(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="model directory does not exist"):
            ensure_model(tmp_path / "nowhere" / "hand_landmarker.task")

    def test_default_location_honours_the_environment(self, tmp_path, monkeypatch):
        from src import hand_tracker

        monkeypatch.setenv("GMC_MODEL_DIR", str(tmp_path))
        assert hand_tracker.default_model_dir() == tmp_path
