"""Feature extraction: normalisation, geometry, poses, and real-hand regression."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.gesture_features import (
    FeatureCalibration,
    INDEX_TIP,
    THUMB_TIP,
    build_feature_vector,
    clamp01,
    feature_names,
    finger_bend_degrees,
    finger_extensions,
    hand_features,
    metric_landmarks,
    normalise,
    palm_roll_degrees,
    palm_size,
)

from .conftest import make_hand

WIDE = 16 / 9


# -- helpers ---------------------------------------------------------------
class TestNormalisation:
    def test_clamp01_bounds_and_nan(self):
        assert clamp01(-2.0) == 0.0
        assert clamp01(2.0) == 1.0
        assert clamp01(0.25) == 0.25
        assert clamp01(float("nan")) == 0.0

    def test_normalise_maps_range(self):
        assert normalise(5.0, 0.0, 10.0) == 0.5
        assert normalise(-1.0, 0.0, 10.0) == 0.0
        assert normalise(11.0, 0.0, 10.0) == 1.0

    def test_normalise_supports_inverted_range(self):
        # Used for finger bend, where a *smaller* angle means more extension.
        assert normalise(20.0, 175.0, 20.0) == 1.0
        assert normalise(175.0, 175.0, 20.0) == 0.0
        assert normalise(97.5, 175.0, 20.0) == pytest.approx(0.5, abs=0.01)

    def test_normalise_degenerate_range(self):
        assert normalise(1.0, 2.0, 2.0) == 0.0


# -- geometry --------------------------------------------------------------
class TestGeometry:
    def test_metric_landmarks_undo_aspect_squash(self):
        """A hand is the same shape whether the frame is square or 16:9."""
        square = make_hand(aspect=1.0)
        wide = make_hand(aspect=WIDE)
        assert not np.allclose(square[:, 0], wide[:, 0])
        # Compare shapes relative to the wrist: the corrected views must be congruent.
        square_shape = metric_landmarks(square, 1.0) - metric_landmarks(square, 1.0)[0]
        wide_shape = metric_landmarks(wide, WIDE) - metric_landmarks(wide, WIDE)[0]
        assert np.allclose(square_shape, wide_shape)

    def test_palm_size_tracks_distance_to_camera(self):
        near = palm_size(make_hand(scale=0.35, aspect=WIDE), WIDE)
        far = palm_size(make_hand(scale=0.12, aspect=WIDE), WIDE)
        assert near > far
        assert near == pytest.approx(0.35, abs=0.02)

    def test_features_are_scale_invariant(self):
        """Moving the hand towards the lens must not change its shape features."""
        near = hand_features(make_hand(scale=0.34, aspect=WIDE), aspect=WIDE)
        far = hand_features(make_hand(scale=0.13, aspect=WIDE), aspect=WIDE)
        for key in ("openness", "finger_spread", "index_extension", "pinch"):
            assert near[key] == pytest.approx(far[key], abs=0.02), key
        assert near["depth"] > far["depth"]  # ... but depth must change

    def test_roll_is_signed_and_folds_at_90_degrees(self):
        level = palm_roll_degrees(make_hand(aspect=WIDE), "right", WIDE)
        tilted = palm_roll_degrees(make_hand(rotation=30.0, aspect=WIDE), "right", WIDE)
        assert abs(level) < 15.0
        assert tilted == pytest.approx(level + 30.0, abs=1.0)
        assert -90.0 <= palm_roll_degrees(make_hand(rotation=170.0, aspect=WIDE), "right", WIDE) <= 90.0

    def test_roll_has_the_same_sign_for_both_hands(self):
        """Rolling both wrists the same way on screen must report the same sign.

        The left hand's knuckles run the other way round, so the vector is
        flipped in the implementation; without that, mirrored hands would report
        opposite pans for the same movement.
        """
        def mirrored(landmarks):
            flipped = landmarks.copy()
            flipped[:, 0] = 1.0 - flipped[:, 0]
            return flipped

        # A mirrored -25 deg hand is a hand rolled +25 deg on screen.
        left_level = palm_roll_degrees(mirrored(make_hand(aspect=WIDE)), "left", WIDE)
        left_rolled = palm_roll_degrees(mirrored(make_hand(rotation=-25.0, aspect=WIDE)), "left", WIDE)
        right_level = palm_roll_degrees(make_hand(aspect=WIDE), "right", WIDE)
        right_rolled = palm_roll_degrees(make_hand(rotation=25.0, aspect=WIDE), "right", WIDE)

        assert abs(left_level) < 15.0 and abs(right_level) < 15.0
        assert left_rolled - left_level == pytest.approx(25.0, abs=1.0)
        assert right_rolled - right_level == pytest.approx(25.0, abs=1.0)

    def test_rotation_does_not_change_finger_extension(self):
        upright = finger_extensions(make_hand(aspect=WIDE), aspect=WIDE)
        sideways = finger_extensions(make_hand(rotation=75.0, aspect=WIDE), aspect=WIDE)
        for finger, value in upright.items():
            assert sideways[finger] == pytest.approx(value, abs=0.02), finger


# -- per-finger curl -------------------------------------------------------
class TestFingerExtension:
    def test_straight_and_curled_extremes(self):
        straight = finger_extensions(make_hand(curl=0.0, aspect=WIDE), aspect=WIDE)
        curled = finger_extensions(make_hand(curl=1.0, thumb=0.0, aspect=WIDE), aspect=WIDE)
        for finger in ("index", "middle", "ring", "pinky"):
            assert straight[finger] > 0.95, finger
            assert curled[finger] < 0.05, finger
        assert straight["thumb"] > curled["thumb"]

    def test_extension_is_monotonic_in_curl(self):
        values = [
            finger_extensions(make_hand(curl=c, aspect=WIDE), aspect=WIDE)["index"]
            for c in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert values == sorted(values, reverse=True)

    def test_bend_angle_matches_the_geometry(self):
        # Two joints bent by 45 deg each at curl=0.5.
        bends = finger_bend_degrees(make_hand(curl=0.5, aspect=WIDE), WIDE)
        assert bends["index"] == pytest.approx(90.0, abs=1.0)


# -- poses -----------------------------------------------------------------
class TestPoses:
    @pytest.mark.parametrize(
        "pose, kwargs",
        [
            ("open_palm", dict(curl=0.0, thumb=1.0)),
            ("fist", dict(curl=1.0, thumb=0.0)),
            ("point_up", dict(curls={"index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}, thumb=0.0)),
            ("peace", dict(curls={"index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0}, thumb=0.0)),
        ],
    )
    def test_only_the_intended_gate_fires(self, pose, kwargs):
        features = hand_features(make_hand(aspect=WIDE, **kwargs), "right", aspect=WIDE)
        assert features[pose] == 1.0
        for other in ("fist", "open_palm", "point_up", "peace"):
            if other != pose:
                assert features[other] == 0.0, f"{other} should not fire for {pose}"

    def test_point_up_requires_the_finger_to_point_up(self):
        curls = {"index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}
        upside_down = make_hand(curls=curls, thumb=0.0, rotation=180.0, aspect=WIDE)
        assert hand_features(upside_down, "right", aspect=WIDE)["point_up"] == 0.0

    def test_pinch_rises_as_thumb_meets_index(self):
        hand = make_hand(aspect=WIDE)
        apart = hand_features(hand, aspect=WIDE)["pinch"]
        pinched = hand.copy()
        pinched[THUMB_TIP] = pinched[INDEX_TIP]  # thumb tip touching index tip
        assert hand_features(pinched, aspect=WIDE)["pinch"] > 0.95
        assert apart < 0.2

    def test_openness_separates_fist_from_open_hand(self):
        open_hand = hand_features(make_hand(curl=0.0, aspect=WIDE), aspect=WIDE)
        fist = hand_features(make_hand(curl=1.0, thumb=0.0, aspect=WIDE), aspect=WIDE)
        assert open_hand["openness"] > fist["openness"] + 0.3

    def test_height_is_inverted_screen_y(self):
        high = hand_features(make_hand(center=(0.5, 0.2), aspect=WIDE), aspect=WIDE)
        low = hand_features(make_hand(center=(0.5, 0.8), aspect=WIDE), aspect=WIDE)
        assert high["height"] == pytest.approx(0.8, abs=0.01)
        assert low["height"] == pytest.approx(0.2, abs=0.01)


# -- calibration -----------------------------------------------------------
class TestCalibration:
    def test_overrides_from_dict(self):
        calib = FeatureCalibration.from_dict({"pinch": [0.2, 1.0], "extended_threshold": 0.8})
        assert calib.pinch == (0.2, 1.0)
        assert calib.extended_threshold == 0.8
        assert calib.openness == FeatureCalibration().openness  # untouched

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError, match="unknown calibration key"):
            FeatureCalibration.from_dict({"nope": [0, 1]})

    def test_malformed_range_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[low, high\]"):
            FeatureCalibration.from_dict({"pinch": [0.2]})

    def test_stricter_threshold_changes_pose_detection(self):
        half_curled = make_hand(curl=0.5, thumb=0.0, aspect=WIDE)
        lenient = FeatureCalibration(curled_threshold=0.6)
        assert hand_features(half_curled, "right", lenient, WIDE)["fist"] == 1.0
        assert hand_features(half_curled, "right", FeatureCalibration(), WIDE)["fist"] == 0.0


# -- frame vector ----------------------------------------------------------
class TestFeatureVector:
    def test_namespacing_and_two_hand_features(self):
        vector = build_feature_vector(
            {"left": make_hand(center=(0.3, 0.5), aspect=WIDE),
             "right": make_hand(center=(0.7, 0.5), aspect=WIDE)},
            aspect=WIDE,
        )
        assert vector["left.present"] == 1.0
        assert vector["right.present"] == 1.0
        assert vector["both.present"] == 1.0
        assert 0.0 < vector["both.hands_distance"] <= 1.0
        assert vector["both.hands_vertical_delta"] == pytest.approx(0.5, abs=0.01)

    def test_missing_hand_omits_its_features(self):
        """A hand that is not tracked must not report zeroed controllers."""
        vector = build_feature_vector({"left": None, "right": make_hand(aspect=WIDE)}, aspect=WIDE)
        assert vector["left.present"] == 0.0
        assert vector["both.present"] == 0.0
        assert not [key for key in vector if key.startswith("left.") and key != "left.present"]
        assert "right.pinch" in vector

    def test_hands_distance_grows_as_hands_separate(self):
        def distance(gap):
            return build_feature_vector(
                {"left": make_hand(center=(0.5 - gap, 0.5), aspect=WIDE),
                 "right": make_hand(center=(0.5 + gap, 0.5), aspect=WIDE)},
                aspect=WIDE,
            )["both.hands_distance"]

        assert distance(0.05) < distance(0.2) < distance(0.4)

    def test_every_documented_feature_is_produced(self):
        """Documentation and reality must not drift apart.

        Speed is the one feature a single frame cannot supply, so it comes from
        MotionTracker — but it is still documented, and still valid in a mapping.
        """
        from src.gesture_features import (
            CONDITIONAL_FEATURES,
            MOTION_FEATURES,
            THUMB_TIP,
            INDEX_TIP,
            MotionTracker,
        )

        hands = {"left": make_hand(aspect=WIDE), "right": make_hand(aspect=WIDE)}
        vector = build_feature_vector(hands, aspect=WIDE)
        always = set(feature_names()) - set(MOTION_FEATURES) - set(CONDITIONAL_FEATURES)
        assert always == set(vector)

        # Speed needs the previous frame ...
        tracker = MotionTracker()
        tracker.update(hands, vector, 0.0, WIDE)
        tracker.update(hands, vector, 0.05, WIDE)

        # ... and the pinch selector only exists while a pinch is happening.
        pinched = {}
        for side in ("left", "right"):
            landmarks = make_hand(curl=0.25, thumb=1.0, aspect=WIDE)
            landmarks[THUMB_TIP] = landmarks[INDEX_TIP]
            pinched[side] = landmarks
        vector.update(build_feature_vector(pinched, aspect=WIDE))
        assert set(feature_names()) == set(vector)

    def test_rejects_wrong_landmark_shape(self):
        with pytest.raises(ValueError, match="expected landmarks of shape"):
            build_feature_vector({"right": np.zeros((5, 3))})


# -- regression against real photographs -----------------------------------
class TestRealHands:
    """Landmarks captured from real photos (see tests/data/reference_landmarks.json).

    These are the cases synthetic geometry cannot prove: real fingers are not
    straight lines, and real hands are photographed at odd angles.
    """

    def test_poses_are_classified_correctly(self, reference_cases):
        assert len(reference_cases) >= 8
        for case in reference_cases:
            features = hand_features(
                np.asarray(case["landmarks"]), case["handedness"], aspect=case["aspect"]
            )
            expected = case["expected_pose"]
            assert features[expected] == 1.0, f"{case['image']} should read as {expected}"
            for other in ("fist", "open_palm", "point_up", "peace"):
                if other != expected:
                    assert features[other] == 0.0, f"{case['image']} wrongly read as {other}"

    def test_open_hands_score_higher_openness_than_closed_ones(self, reference_cases):
        by_pose = {}
        for case in reference_cases:
            features = hand_features(
                np.asarray(case["landmarks"]), case["handedness"], aspect=case["aspect"]
            )
            by_pose.setdefault(case["expected_pose"], []).append(features["openness"])
        assert min(by_pose["open_palm"]) > max(by_pose["fist"])

    def test_all_features_stay_within_range(self, reference_cases):
        for case in reference_cases:
            features = hand_features(
                np.asarray(case["landmarks"]), case["handedness"], aspect=case["aspect"]
            )
            for name, value in features.items():
                assert 0.0 <= value <= 1.0, f"{name}={value} out of range in {case['image']}"


# -- motion and finger counting --------------------------------------------
class TestFingerCount:
    @pytest.mark.parametrize("kwargs, expected", [
        (dict(curl=1.0, thumb=0.0), 0.0),
        (dict(curls={"index": 0, "middle": 1, "ring": 1, "pinky": 1}, thumb=0.0), 0.2),
        (dict(curls={"index": 0, "middle": 0, "ring": 1, "pinky": 1}, thumb=0.0), 0.4),
        (dict(curl=0.0, thumb=1.0), 1.0),
    ])
    def test_counts_extended_fingers_in_fifths(self, kwargs, expected):
        features = hand_features(make_hand(aspect=WIDE, **kwargs), "right", aspect=WIDE)
        assert features["finger_count"] == pytest.approx(expected)

    def test_it_is_a_usable_selector(self):
        """Distinct poses must land on distinct values, or it cannot pick anything."""
        counts = {
            hand_features(make_hand(aspect=WIDE, **kwargs), "right", aspect=WIDE)["finger_count"]
            for kwargs in (
                dict(curl=1.0, thumb=0.0),
                dict(curls={"index": 0, "middle": 1, "ring": 1, "pinky": 1}, thumb=0.0),
                dict(curls={"index": 0, "middle": 0, "ring": 1, "pinky": 1}, thumb=0.0),
                dict(curl=0.0, thumb=1.0),
            )
        }
        assert len(counts) == 4


class TestMotionTracker:
    from src.gesture_features import MotionTracker

    def track(self, positions, dt=1 / 30):
        tracker = self.MotionTracker()
        values = []
        for index, x in enumerate(positions):
            hands = {"right": make_hand(center=(x, 0.5), aspect=WIDE), "left": None}
            vector = build_feature_vector(hands, aspect=WIDE)
            tracker.update(hands, vector, index * dt, WIDE)
            values.append(vector.get("right.speed"))
        return values

    def test_the_first_frame_has_no_speed_yet(self):
        assert self.track([0.5])[0] is None

    def test_a_still_hand_reads_as_slow(self):
        assert self.track([0.5] * 10)[-1] == pytest.approx(0.0, abs=0.01)

    def test_a_moving_hand_reads_as_fast(self):
        assert self.track([0.2 + 0.06 * step for step in range(10)])[-1] > 0.5

    def test_faster_movement_reads_higher(self):
        slow = self.track([0.4 + 0.004 * step for step in range(10)])[-1]
        fast = self.track([0.4 + 0.04 * step for step in range(10)])[-1]
        assert fast > slow

    def test_landmark_jitter_does_not_read_as_movement(self):
        """A still hand jitters by a pixel or two; that must not read as a strike."""
        positions = [0.5 + 0.004 * (-1) ** step for step in range(16)]
        peak = max(value for value in self.track(positions) if value is not None)
        assert peak < 0.35

    def test_a_real_strike_still_reads_high(self):
        """Smoothing must not blunt a genuine fast move — that is the point of it."""
        positions = [0.3] * 4 + [0.3 + 0.09 * step for step in range(5)]
        assert max(value for value in self.track(positions) if value is not None) > 0.8

    def test_losing_the_hand_clears_its_state(self):
        tracker = self.MotionTracker()
        hands = {"right": make_hand(aspect=WIDE), "left": None}
        tracker.update(hands, build_feature_vector(hands, aspect=WIDE), 0.0, WIDE)
        tracker.update({"right": None, "left": None}, {}, 0.1, WIDE)
        # Coming back must not report a huge jump from the stale position.
        vector = build_feature_vector(hands, aspect=WIDE)
        tracker.update(hands, vector, 5.0, WIDE)
        assert "right.speed" not in vector

    def test_a_stale_timestamp_is_ignored(self):
        tracker = self.MotionTracker()
        hands = {"right": make_hand(aspect=WIDE), "left": None}
        tracker.update(hands, build_feature_vector(hands, aspect=WIDE), 0.0, WIDE)
        vector = build_feature_vector(hands, aspect=WIDE)
        tracker.update(hands, vector, 0.0, WIDE)     # same timestamp, no elapsed time
        assert "right.speed" not in vector
