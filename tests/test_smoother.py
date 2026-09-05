"""EMA smoothing and the Schmitt trigger used for gesture gates."""

from __future__ import annotations

import pytest

from src.smoother import EMASmoother, SchmittTrigger, window_to_alpha


class TestWindowToAlpha:
    def test_matches_the_moving_average_convention(self):
        assert window_to_alpha(5) == pytest.approx(2 / 6)
        assert window_to_alpha(9) == pytest.approx(0.2)

    def test_window_of_one_or_less_disables_smoothing(self):
        assert window_to_alpha(1) == 1.0
        assert window_to_alpha(0) == 1.0

    def test_larger_window_smooths_harder(self):
        assert window_to_alpha(20) < window_to_alpha(5)


class TestEMASmoother:
    def test_first_sample_passes_through(self):
        """Seeding with the first value avoids a slow ramp from zero on start-up."""
        assert EMASmoother(window=10).update({"a": 0.8})["a"] == 0.8

    def test_converges_towards_the_input(self):
        smoother = EMASmoother(window=5)
        smoother.update({"a": 0.0})
        values = [smoother.update({"a": 1.0})["a"] for _ in range(20)]
        assert values == sorted(values)
        assert values[0] == pytest.approx(1 / 3, abs=0.01)
        assert values[-1] == pytest.approx(1.0, abs=0.01)

    def test_attenuates_a_single_frame_spike(self):
        smoother = EMASmoother(window=5)
        for _ in range(10):
            smoother.update({"a": 0.5})
        spiked = smoother.update({"a": 1.0})["a"]
        assert 0.5 < spiked < 0.7  # a one-frame glitch moves the output only a little

    def test_missing_feature_holds_its_value_and_is_not_reported(self):
        smoother = EMASmoother(window=5)
        smoother.update({"a": 0.9, "b": 0.1})
        result = smoother.update({"b": 0.1})
        assert "a" not in result           # nothing to send for a lost hand ...
        assert smoother.value("a") == 0.9  # ... but the value is remembered

    def test_state_expires_after_a_long_absence(self):
        smoother = EMASmoother(window=5, reset_after=3)
        smoother.update({"a": 1.0})
        for _ in range(3):
            smoother.update({"b": 0.0})
        assert smoother.value("a") is None
        # A hand returning later starts from its new value, not from a stale one.
        assert smoother.update({"a": 0.2})["a"] == 0.2

    def test_reset_after_none_keeps_state_forever(self):
        smoother = EMASmoother(window=5, reset_after=None)
        smoother.update({"a": 1.0})
        for _ in range(50):
            smoother.update({})
        assert smoother.value("a") == 1.0

    def test_per_feature_window_override(self):
        smoother = EMASmoother(window=20)
        smoother.set_window("snappy", 1)
        smoother.update({"snappy": 0.0, "slow": 0.0})
        result = smoother.update({"snappy": 1.0, "slow": 1.0})
        assert result["snappy"] == 1.0
        assert result["slow"] < 0.2

    def test_set_alpha_is_clamped(self):
        smoother = EMASmoother()
        smoother.set_alpha("a", 5.0)
        assert smoother.alpha_for("a") == 1.0

    def test_reset_clears_everything(self):
        smoother = EMASmoother()
        smoother.update({"a": 1.0, "b": 2.0})
        assert len(smoother) == 2
        smoother.reset()
        assert len(smoother) == 0


class TestSchmittTrigger:
    def test_hysteresis_band_prevents_chatter(self):
        trigger = SchmittTrigger(on=0.6, off=0.4)
        assert trigger.update(0.5) is False   # below the trigger point
        assert trigger.update(0.65) is True
        assert trigger.update(0.5) is True    # inside the band: stays latched
        assert trigger.update(0.35) is False

    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError, match="release threshold"):
            SchmittTrigger(on=0.3, off=0.7)

    def test_equal_thresholds_are_allowed(self):
        trigger = SchmittTrigger(on=0.5, off=0.5)
        assert trigger.update(0.5) is True

    def test_smoothed_gate_debounces_a_single_bad_frame(self):
        """One misdetected frame must not flip a sustain pedal."""
        smoother = EMASmoother(window=5)
        trigger = SchmittTrigger(on=0.6, off=0.4)
        for _ in range(10):
            trigger.update(smoother.update({"fist": 1.0})["fist"])
        assert trigger.state is True
        assert trigger.update(smoother.update({"fist": 0.0})["fist"]) is True
        # A sustained change does get through.
        for _ in range(5):
            trigger.update(smoother.update({"fist": 0.0})["fist"])
        assert trigger.state is False
