"""Deterministic block decomposition in standardized units (CPU float64)."""
import numpy as np
from scipy.fft import dct
from scipy.linalg import block_diag

W, L, RANK = 96, 32, 2
SHIFTS = W - L + 1
F = dct(np.eye(L), axis=0, norm="ortho")
TAU, EPS, ETA = 0.05, 1e-4, 0.1
TOL, LIMIT = 1e-5, 500


def hankel(y):
    return np.lib.stride_tricks.sliding_window_view(y, L, axis=0).transpose(2, 0, 1)


def adjoint(h):
    y = np.zeros((W, h.shape[-1]))
    for j in range(L):
        y[j:j + SHIFTS] += h[j]
    return y


def covariance(blocks):
    """Mean of per-block, per-record normalized band covariances."""
    total = np.zeros((4, 8, 8))
    for x in blocks:
        lag = hankel(x).reshape(L, -1)
        bands = (F @ lag).reshape(4, 8, -1)
        total += bands @ bands.transpose(0, 2, 1) / lag.size
    return total / len(blocks)


def planes(cov, anchor=None):
    matrix = cov if anchor is None else cov + 2 * ETA * anchor
    result = []
    for band in matrix:
        values, vectors = np.linalg.eigh(band)
        cut = values[-RANK]
        tol = 16 * np.finfo(float).eps * max(1, abs(values).max())
        above = vectors[:, values > cut + tol]
        tied = vectors[:, abs(values - cut) <= tol]
        selected = list(above.T)
        for v in (tied @ tied.T).T:
            if len(selected) == RANK:
                break
            v = v.copy()
            for q in selected:
                v -= q.dot(v) * q
            if np.linalg.norm(v) > 1e-12:
                selected.append(v / np.linalg.norm(v))
        q = np.asarray(selected)
        result.append(q.T @ q)
    return np.stack(result)


def curvature(p):
    """H*F*(I-P)FH, scaled for the objective multiplied by W*C."""
    kernel = F.T @ (np.eye(L) - block_diag(*p)) @ F
    result = np.zeros((W, W))
    for start in range(SHIFTS):
        result[start:start+L, start:start+L] += kernel
    return W / (L * SHIFTS) * result + EPS * np.eye(W)


def soft(r, threshold=1):
    return np.sign(r) * np.maximum(abs(r)-threshold, 0)


def svt(y, threshold):
    # Equivalent to thin SVD shrinkage, avoiding left vectors in the null space.
    values, v = np.linalg.eigh(y.T @ y)
    singular = np.sqrt(np.maximum(values, 0))
    factors = np.maximum(singular-threshold, 0) / np.maximum(singular, 1e-300)
    return (y @ (v * factors)) @ v.T, float(np.maximum(singular-threshold, 0).sum())


def smooth(x, y, a, robust):
    r = y-x
    g = np.clip(r, -1, 1) if robust else r
    obs = np.sum(g*(r - 0.5*g))
    ay = a @ y
    return obs + 0.5*np.sum(y*ay), g + ay


class Solver:
    def __init__(self, p, c, robust=True):
        self.a = curvature(p)
        self.step = 1 / (1 + np.linalg.eigvalsh(self.a)[-1])
        self.penalty = TAU*np.sqrt(W*c)
        self.robust = robust

    def solve(self, x):
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite observation")
        y = x.copy()
        alpha = self.step
        initial, grad = smooth(x, y, self.a, self.robust)
        objective = initial + self.penalty*np.linalg.svd(y, compute_uv=False).sum()
        first = objective
        value = initial
        for it in range(1, LIMIT+1):
            while True:
                new, norm = svt(y-alpha*grad, alpha*self.penalty)
                delta = new-y
                squared = np.sum(delta*delta)
                newval, newgrad = smooth(x, new, self.a, self.robust)
                # Global Lipschitz step first; guard majorization in finite precision.
                bound = value + np.sum(grad*delta) + squared/(2*alpha)
                if newval <= bound + 1e-12*max(1, abs(bound)):
                    break
                alpha *= 0.5
                if alpha < 1e-12:
                    raise ArithmeticError("Backtracking failed")
            residual = np.sqrt(squared)/(alpha*max(1., np.linalg.norm(y)))
            y, value, grad = new, newval, newgrad
            objective = value + self.penalty*norm
            if residual < TOL:
                break
        if not np.isfinite(objective) or objective > first + 1e-10*max(1,first):
            raise ArithmeticError("Invalid final objective")
        return y, (it, residual, first/x.size, objective/x.size)
