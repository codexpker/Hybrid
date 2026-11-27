"""An auditable lag-change readout alongside a frozen native LEFT score."""

from dataclasses import dataclass

import numpy as np

HISTORY, WINDOW, BLOCK, MAX_LAG = 256, 192, 64, 24
EPS = 1e-12
MIN_HISTORY_R2 = 0.20
MAX_TEMPLATE_CORRELATION = 0.95


@dataclass(frozen=True)
class Link:
    driver: int
    receiver: int
    delay: int
    sign: float
    history_r2: float


@dataclass(frozen=True)
class Reference:
    width: int
    rho: float
    links: tuple[Link, ...]


@dataclass(frozen=True)
class BlockScore:
    delay: float
    residual: float
    available_links: int
    confirming_links: int
    best_receiver: int | None
    best_delay: int | None
    unconfirmed: float = 0.0
    testable_links: int = 0


def _array(x, length):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) != length or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError(
            f"Expected finite observations of shape [{length}, channels>=2]"
        )
    return x


def _center(x, axis=0):
    return x - x.mean(axis=axis, keepdims=True)


def _filtered(x, rho):
    z = np.zeros_like(x)
    z[1:] = x[1:] - rho * x[:-1]
    return z


def _history_filter(h):
    previous = _center(h[:127])
    following = _center(h[1:128])
    energy = np.mean(previous**2, axis=0)
    valid = energy > EPS
    if not valid.any():
        return 0.0
    slopes = np.mean(previous * following, axis=0)[valid] / energy[valid]
    return float(np.clip(np.median(slopes), -0.95, 0.95))


def fit_reference(history):
    """Select on the first half and verify on a disjoint second footprint."""
    h = _array(history, HISTORY)
    rho = _history_filter(h)
    z = _filtered(h, rho)
    selection = np.arange(MAX_LAG + 1, HISTORY // 2)
    validation = np.arange(HISTORY // 2 + MAX_LAG + 1, HISTORY)
    y = z[selection]
    y_mean = y.mean(0)
    yc = y - y_mean
    y_energy = np.mean(yc**2, axis=0)
    best = np.full(z.shape[1], -np.inf)
    proposals = [None] * z.shape[1]
    for lag in range(1, MAX_LAG + 1):
        x = z[selection - lag]
        x_mean = x.mean(0)
        xc = x - x_mean
        x_energy = np.mean(xc**2, axis=0)
        cov = xc.T @ yc / len(selection)
        valid = (x_energy[:, None] > EPS) & (y_energy[None, :] > EPS)
        r2 = np.full_like(cov, -np.inf)
        np.divide(cov**2, x_energy[:, None] * y_energy[None, :], out=r2, where=valid)
        np.fill_diagonal(r2, -np.inf)
        drivers = np.argmax(r2, axis=0)
        for j, i in enumerate(drivers):
            if r2[i, j] > best[j]:
                best[j] = r2[i, j]
                b = cov[i, j] / x_energy[i]
                proposals[j] = (int(i), lag, b, y_mean[j] - b * x_mean[i])
    links = []
    for j, proposal in enumerate(proposals):
        if proposal is None:
            continue
        i, lag, slope, intercept = proposal
        observed = z[validation, j]
        baseline = np.mean((observed - y_mean[j]) ** 2)
        if slope == 0 or baseline <= EPS:
            continue
        predicted = intercept + slope * z[validation - lag, i]
        gain = 1.0 - np.mean((observed - predicted) ** 2) / baseline
        if gain >= MIN_HISTORY_R2:
            links.append(Link(i, j, lag, float(np.sign(slope)), float(gain)))
    return Reference(z.shape[1], rho, tuple(links))


def _profile(x_fit, y_fit, x_test, y_test):
    """Profile the same intercept and nonnegative gain at every signed delay."""
    mx, my = x_fit.mean(1, keepdims=True), y_fit.mean()
    xc = x_fit - mx
    denom = np.mean(xc**2, axis=1)
    gain = np.zeros(len(x_fit))
    np.divide(np.mean(xc * (y_fit - my), axis=1), denom, out=gain, where=denom > EPS)
    gain = np.maximum(gain, 0.0)[:, None]
    train = np.mean((y_fit - my - gain * xc) ** 2, axis=1)
    test = np.mean((y_test - my - gain * (x_test - mx)) ** 2, axis=1)
    return train, test


def _distinguishable(templates, reference):
    x = _center(templates, axis=1)
    scale = np.sqrt(np.mean(x**2, axis=1))
    correlation = np.ones_like(scale)
    usable = (scale**2 > EPS) & (scale[reference] ** 2 > EPS)
    np.divide(
        np.abs(np.mean(x * x[reference], axis=1)),
        scale * scale[reference],
        out=correlation,
        where=usable,
    )
    return usable & (correlation < MAX_TEMPLATE_CORRELATION)


def _block(z, indices, links):
    delay_score = residual_score = 0.0
    unconfirmed = 0.0
    best_receiver = best_delay = None
    available = confirming = testable = 0
    half = len(indices) // 2
    lag = np.arange(1, MAX_LAG + 1)
    for link in links:
        y = z[indices, link.receiver]
        variance = np.var(y)
        if variance <= EPS:
            continue
        x = link.sign * z[indices[None, :] - lag[:, None], link.driver]
        null = link.delay - 1
        xa, xb = x[:, :half], x[:, half:]
        ya, yb = y[:half], y[half:]
        if min(np.var(xa[null]), np.var(xb[null])) <= EPS:
            continue
        available += 1
        train_a, test_b = _profile(xa, ya, xb, yb)
        train_b, test_a = _profile(xb, yb, xa, ya)
        residual_score = max(
            residual_score, (test_a[null] + test_b[null]) / (2 * variance)
        )
        legal = _distinguishable(xa, null) & _distinguishable(xb, null)
        if not legal.any():
            continue
        testable += 1
        a = int(np.argmin(np.where(legal, train_a, np.inf)))
        b = int(np.argmin(np.where(legal, train_b, np.inf)))
        # C uses the same two proposals and strictly out-of-half errors.
        # Each direction keeps its original fit-side improvement requirement.
        if train_a[a] < train_a[null]:
            unconfirmed = max(unconfirmed, (test_b[null] - test_b[a]) / variance)
        if train_b[b] < train_b[null]:
            unconfirmed = max(unconfirmed, (test_a[null] - test_a[b]) / variance)
        if a != b or train_a[a] >= train_a[null] or train_b[b] >= train_b[null]:
            continue
        gain = min(test_a[null] - test_a[a], test_b[null] - test_b[b]) / variance
        if gain > 0:
            confirming += 1
        if gain > delay_score:
            delay_score = float(gain)
            best_receiver, best_delay = link.receiver, a + 1
    return BlockScore(
        delay_score,
        float(residual_score),
        available,
        confirming,
        best_receiver,
        best_delay,
        float(unconfirmed),
        testable,
    )


def score_region(reference, history, current, start, stop):
    """Score one full 64-point region. All footprints are inside H256+W192."""
    h, x = _array(history, HISTORY), _array(current, WINDOW)
    if x.shape[1] != h.shape[1] or h.shape[1] != reference.width:
        raise ValueError(
            "Current and historical channels must have the same order and width"
        )
    if not (0 <= start < stop <= WINDOW and stop - start == BLOCK):
        raise ValueError("A score region must contain exactly 64 current points")
    z = _filtered(np.concatenate((h, x)), reference.rho)
    return _block(z, HISTORY + np.arange(start, stop), reference.links)


def score_window(reference, history, current):
    return tuple(
        score_region(reference, history, current, s, s + BLOCK)
        for s in range(0, WINDOW, BLOCK)
    )


def _tail(normal, observed):
    normal = np.sort(np.asarray(normal, dtype=np.float64))
    observed = np.asarray(observed, dtype=np.float64)
    if (
        normal.ndim != 1
        or not len(normal)
        or not (np.isfinite(normal).all() and np.isfinite(observed).all())
    ):
        raise ValueError(
            "Finite nonempty calibration reference and finite scores required"
        )
    rank = len(normal) - np.searchsorted(normal, observed, side="left")
    return (1 + rank) / (1 + len(normal))


def combine(native, blocks, native_normal, delay_normal, channel="delay"):
    """Return point scores using the same calibration for the full and control readouts."""
    native = np.asarray(native, dtype=np.float64)
    if native.shape != (WINDOW,) or len(blocks) != WINDOW // BLOCK:
        raise ValueError("Expected W192 point scores and three ordered block scores")
    if channel not in {"delay", "residual", "unconfirmed"}:
        raise ValueError(channel)
    reference = np.asarray(delay_normal)
    if np.any(reference < 0):
        raise ValueError("Calibrate the nonnegative block statistic")
    values = np.array([getattr(b, channel) for b in blocks])
    return np.maximum(
        -np.log(_tail(native_normal, native)),
        np.repeat(-np.log(_tail(reference, values)), BLOCK),
    )
