"""Fixed strict-past Ridge residual and causal EMA controls.

This is a classical VAR/FIR Ridge predictor plus known EMA accumulation,
not a new neural architecture. Feature order matches the archived control:
channel-major, each channel oldest to newest (lags 512 down to 1), intercept.
Prior error accumulation reference: official Telemanom errors.py uses absolute
prediction error and EWM (https://raw.githubusercontent.com/khundman/telemanom/master/telemanom/errors.py).
We do not reuse its future-dependent startup replacement or test-window-fitted
thresholds: initialization is g[0]=r[0], calibration is fixed normal CAL only.
"""
from __future__ import annotations

import numpy as np

LAG = 512
EMA_DECAY = 63.0 / 64.0
EPSILON = 1e-6


def feature_view(values: np.ndarray) -> np.ndarray:
    """View X[t] contains only values[t-512:t], never values[t]."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) <= LAG or not np.isfinite(values).all():
        raise ValueError("finite [time, channel] with more than 512 points required")
    return np.lib.stride_tricks.sliding_window_view(values, LAG, axis=0)[:-1]


def feature_block(view: np.ndarray, begin: int, end: int) -> np.ndarray:
    x = np.empty((end - begin, view.shape[1] * LAG + 1), np.float64)
    x[:, :-1] = view[begin:end].reshape(end - begin, -1)
    x[:, -1] = 1.0
    return x


def predict_numpy(values: np.ndarray, weight: np.ndarray, block: int = 2048) -> np.ndarray:
    view = feature_view(values)
    channels = values.shape[1]
    if weight.shape != (channels * LAG + 1, channels):
        raise ValueError("Ridge weight/channel dimension mismatch")
    prediction = np.empty((len(view), channels), np.float64)
    for begin in range(0, len(view), block):
        end = min(begin + block, len(view))
        prediction[begin:end] = feature_block(view, begin, end) @ weight
    return prediction


def accumulate(residual_score: np.ndarray) -> np.ndarray:
    """One independent run; g[0]=r[0], no invented preceding residuals."""
    score = np.asarray(residual_score, dtype=np.float64)
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all():
        raise ValueError("finite nonempty score vector required")
    result = np.empty_like(score)
    result[0] = score[0]
    for t in range(1, len(score)):
        result[t] = EMA_DECAY * result[t - 1] + (1.0 - EMA_DECAY) * score[t]
    return result


def residual_score(residual: np.ndarray, scale: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if residual.ndim != 2 or scale.shape != (residual.shape[1],):
        raise ValueError("residual/channel scale mismatch")
    if not np.isfinite(residual).all() or not np.isfinite(scale).all() or (scale < EPSILON).any():
        raise ValueError("invalid residual/scale")
    return (np.abs(residual) / scale).mean(axis=1)


def robust_parameters(calibration: np.ndarray) -> dict[str, float]:
    values = np.asarray(calibration, dtype=np.float64)
    if values.shape != (512,) or not np.isfinite(values).all():
        raise ValueError("exactly 512 finite fixed calibration points required")
    median = float(np.median(values))
    return {"median": median, "MAD": float(max(np.median(np.abs(values - median)), EPSILON))}


def normal_excess(values: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    return np.maximum(0.0, (np.asarray(values, np.float64) - parameters["median"]) / parameters["MAD"])


def calibrated_scores(left: np.ndarray, point: np.ndarray, accumulated: np.ndarray, parameters: dict) -> dict[str, np.ndarray]:
    if not (left.shape == point.shape == accumulated.shape):
        raise ValueError("all components must share the identical scoring axis")
    z_left = normal_excess(left, parameters["LEFT"])
    z_point = normal_excess(point, parameters["point"])
    z_accumulated = normal_excess(accumulated, parameters["accumulated"])
    # Single-model canonical scores remain unclipped, avoiding artificial ties.
    return {
        "LEFT_native": np.asarray(left),
        "ARX_point": np.asarray(point),
        "ARX_accumulated": np.asarray(accumulated),
        "LEFT_ARX_point": 0.5 * z_left + 0.5 * z_point,
        "Hybrid_LEFT_accumulated": 0.5 * z_left + 0.5 * z_accumulated,
    }
