"""Exponential moving average smoothing for gesture features.

Hand landmarks jitter by a pixel or two every frame even when the hand is
perfectly still.  Fed straight into a MIDI CC that becomes a stream of
``74 -> 75 -> 74`` messages: audible zipper noise on a filter, and needless
traffic on the MIDI bus.

An EMA is the right trade-off here — one multiply per feature, no added latency
beyond the filter's own lag, and a single intuitive knob (``window``).  With
``alpha = 2 / (window + 1)`` the "window" matches the familiar moving-average
span, so ``window=5`` at 30 fps costs roughly 60 ms of lag: smooth enough for a
filter sweep, still tight enough to feel like an instrument.

Discrete pose gates (``left.fist`` and friends) benefit too: smoothing turns
0/1 flips into ramps, and the Schmitt trigger in :mod:`midi_mapper` then reads
those ramps as debounce, so a single misdetected frame cannot flip the sustain
pedal.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional


def window_to_alpha(window: float) -> float:
    """Convert a moving-average span to an EMA coefficient, clamped to ``(0, 1]``.

    ``window <= 1`` disables smoothing (``alpha = 1.0``, pass-through).
    """
    if window <= 1:
        return 1.0
    return max(min(2.0 / (float(window) + 1.0), 1.0), 1e-6)


class EMASmoother:
    """Per-feature exponential moving average with optional per-key windows.

    Features missing from an update keep their stored value but are **not**
    returned, so a hand that leaves the frame freezes its controllers instead of
    snapping them to zero.

    Args:
        window: default moving-average span in frames.
        alpha: explicit coefficient; overrides ``window`` when given.
        reset_after: drop a feature's state after this many consecutive frames
            without it, so a hand returning after a long absence starts clean
            instead of gliding from a stale value. ``None`` keeps state forever.
    """

    def __init__(
        self,
        window: float = 5.0,
        alpha: Optional[float] = None,
        reset_after: Optional[int] = 30,
    ) -> None:
        self.alpha = float(alpha) if alpha is not None else window_to_alpha(window)
        self.reset_after = reset_after
        self._values: Dict[str, float] = {}
        self._alphas: Dict[str, float] = {}
        self._missing: Dict[str, int] = {}

    # -- configuration -----------------------------------------------------
    def set_window(self, feature: str, window: float) -> None:
        """Give one feature its own smoothing span (e.g. snappier triggers)."""
        self._alphas[feature] = window_to_alpha(window)

    def set_alpha(self, feature: str, alpha: float) -> None:
        """Give one feature an explicit EMA coefficient."""
        self._alphas[feature] = max(min(float(alpha), 1.0), 1e-6)

    def alpha_for(self, feature: str) -> float:
        return self._alphas.get(feature, self.alpha)

    # -- runtime -----------------------------------------------------------
    def update(self, features: Mapping[str, float]) -> Dict[str, float]:
        """Smooth one frame of features and return the smoothed values."""
        smoothed: Dict[str, float] = {}
        for key, raw in features.items():
            value = float(raw)
            alpha = self.alpha_for(key)
            previous = self._values.get(key)
            current = value if previous is None else previous + alpha * (value - previous)
            self._values[key] = current
            self._missing[key] = 0
            smoothed[key] = current

        if self.reset_after is not None:
            self._expire(features.keys())
        return smoothed

    def _expire(self, present: Iterable[str]) -> None:
        seen = set(present)
        for key in list(self._values):
            if key in seen:
                continue
            missed = self._missing.get(key, 0) + 1
            if missed >= self.reset_after:
                self._values.pop(key, None)
                self._missing.pop(key, None)
            else:
                self._missing[key] = missed

    def value(self, feature: str, default: Optional[float] = None) -> Optional[float]:
        """Last smoothed value of ``feature`` (``default`` if never seen)."""
        return self._values.get(feature, default)

    def reset(self) -> None:
        """Forget all state (used when the config is hot-reloaded)."""
        self._values.clear()
        self._missing.clear()

    def __len__(self) -> int:
        return len(self._values)


class SchmittTrigger:
    """Two-threshold gate: fires above ``on``, releases below ``off``.

    A single threshold on a noisy signal chatters; the gap between the two
    thresholds is what keeps a sustain pedal from machine-gunning while a hand
    hovers between "fist" and "not quite fist".
    """

    def __init__(self, on: float = 0.6, off: float = 0.4, initial: bool = False) -> None:
        if off > on:
            raise ValueError(f"release threshold {off} must not exceed trigger threshold {on}")
        self.on = float(on)
        self.off = float(off)
        self.state = bool(initial)

    def update(self, value: float) -> bool:
        """Feed a value, return the (possibly unchanged) gate state."""
        if self.state:
            if value <= self.off:
                self.state = False
        elif value >= self.on:
            self.state = True
        return self.state

    def reset(self, state: bool = False) -> None:
        self.state = bool(state)
