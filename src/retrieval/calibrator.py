"""Finite-safe calibration helpers for opt-in Top-20 cross-modal rescoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

CalibrationMethod = Literal["rank", "minmax", "zscore"]


@dataclass(frozen=True, slots=True)
class CalibrationWeights:
    base: float = 0.60
    global_image: float = 0.25
    crop: float = 0.15

    def __post_init__(self) -> None:
        values = (self.base, self.global_image, self.crop)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("calibration weights must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("calibration weights must sum to one")

    def to_dict(self) -> dict[str, float]:
        return {
            "base": self.base,
            "global_image": self.global_image,
            "crop": self.crop,
        }


def _finite_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values)


def calibrate_scores(
    values: Sequence[float],
    *,
    method: CalibrationMethod = "rank",
    neutral: float = 0.5,
) -> np.ndarray:
    """Map one comparable score pool to [0, 1], keeping invalid values neutral."""

    if method not in {"rank", "minmax", "zscore"}:
        raise ValueError("calibration method must be rank, minmax, or zscore")
    if not 0.0 <= neutral <= 1.0:
        raise ValueError("neutral calibration value must be in [0, 1]")
    raw = np.asarray(tuple(values), dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("calibration values must be one-dimensional")
    output = np.full(raw.shape, float(neutral), dtype=np.float64)
    mask = _finite_mask(raw)
    finite = raw[mask]
    if finite.size < 2 or float(np.ptp(finite)) <= 1e-12:
        return output.astype(np.float32)

    if method == "minmax":
        calibrated = (finite - float(np.min(finite))) / float(np.ptp(finite))
    elif method == "zscore":
        deviation = float(np.std(finite))
        if deviation <= 1e-12:
            return output.astype(np.float32)
        zscore = np.clip((finite - float(np.mean(finite))) / deviation, -3.0, 3.0)
        calibrated = (zscore + 3.0) / 6.0
    else:
        order = np.argsort(finite, kind="stable")
        calibrated = np.empty(finite.shape, dtype=np.float64)
        position = 0
        denominator = max(1, finite.size - 1)
        while position < finite.size:
            end = position + 1
            value = finite[order[position]]
            while end < finite.size and finite[order[end]] == value:
                end += 1
            average_rank = (position + end - 1) / 2.0
            percentile = average_rank / denominator
            calibrated[order[position:end]] = percentile
            position = end

    output[mask] = calibrated
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def blend_components(
    components: Mapping[str, Sequence[float]],
    *,
    weights: CalibrationWeights = CalibrationWeights(),
    method: CalibrationMethod = "rank",
) -> np.ndarray:
    """Calibrate comparable Top-N components independently and blend them."""

    required = {
        "base": weights.base,
        "global_image": weights.global_image,
        "crop": weights.crop,
    }
    if set(components) != set(required):
        raise ValueError("components must contain base, global_image, and crop")
    lengths = {len(tuple(values)) for values in components.values()}
    if len(lengths) != 1:
        raise ValueError("calibration component lengths must match")
    length = next(iter(lengths), 0)
    if length == 0:
        return np.empty((0,), dtype=np.float32)
    output = np.zeros((length,), dtype=np.float32)
    for name, weight in required.items():
        output += float(weight) * calibrate_scores(components[name], method=method)
    return output


__all__ = [
    "CalibrationMethod",
    "CalibrationWeights",
    "blend_components",
    "calibrate_scores",
]
