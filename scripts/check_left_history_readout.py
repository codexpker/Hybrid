"""Independent CPU reconstruction of the fixed public readout on sealed banks."""

import argparse
from pathlib import Path

import numpy as np

from hybrid_tsad import left_history_public as public
from hybrid_tsad.left_swat_author import dump, read, sha


def numpy_coefficients(x, start):
    lags = (0, 1, 2, 4, 8, 16, 32, 64)
    series = x[start : start + 32]
    centered = series - series.mean(axis=0)
    values, masks = [], []
    for lag in lags:
        y = x[start - lag : start + 32 - lag]
        centered_y = y - y.mean(axis=0)
        # CPU float64 Pearson coefficient, with a separate validity indicator.
        sx = np.sqrt(np.mean(centered**2, axis=0))
        sy = np.sqrt(np.mean(centered_y**2, axis=0))
        ok = (sx[:, None] > 1e-5) & (sy[None, :] > 1e-5)
        cov = centered.T @ centered_y / 32
        values.append(
            np.divide(cov, sx[:, None] * sy[None, :], out=np.zeros_like(cov), where=ok)
        )
        masks.append(ok)
    return np.array(values), np.array(masks)


def check(w):
    root = w / public.OUT
    review = {}
    for domain in ("PSM", "SWAT", "SMD"):
        folder = root / domain
        matched, joint = [
            read(folder / a / "receipt.json") for a in ("continue", "joint")
        ]
        for key in (
            "parent",
            "steps",
            "training_windows",
            "training_plan_sha256",
            "source_select_plan_sha256",
            "native_rng_start",
            "native_rng_end",
        ):
            assert matched[key] == joint[key], (domain, key)
        assert joint["steps"] == 2000 and joint["added_parameters"] == 54954
        members = public.members(w, domain)
        errors = []
        mixes = []
        for entity in members:
            path = root / "public" / domain / entity / "joint.npz"
            rb = read(path.with_suffix(".json"))
            assert sha(path) == rb["sha256"] and joint["sha256"] == rb["checkpoint"]
            with np.load(path) as z:
                idx, available = z["index"], z["available"]
                np.testing.assert_array_equal(
                    z["history_start"] + 256, z["current_start"]
                )
                np.testing.assert_array_equal(available, z["current_start"] + 191)
                assert np.all((idx <= available) & (idx > available - 192))
                for name in ("self", "cross", "relation", "simple"):
                    assert np.all(z[name].reshape(-1, 32) == z[name][::32, None])
                scales = joint["scales"]
                full = (
                    z["native"] / scales["native"] + z["relation"] / scales["relation"]
                ) / 2
                fallback = z["native"] / scales["native"]
                np.testing.assert_allclose(
                    z["joint"],
                    np.where(
                        (z["relation"] == 0) & (z["joint"] == fallback), fallback, full
                    ),
                    rtol=2e-6,
                )
                if entity not in (members[0], members[-1]):
                    continue
                x, off = public.test_values(w, domain, entity)
                # Count groups and reconstruct the persistence score independently for
                # the first, middle and final complete input footprint on this record.
                simple = z["simple"]
                for window in (0, len(idx) // 384, len(idx) // 192 - 1):
                    i = window * 192
                    start = int(z["history_start"][i]) - off
                    block = x[start : start + 448].astype(np.float64)
                    previous, old_ok = numpy_coefficients(block, 224)
                    diag = np.eye(block.shape[-1], dtype=bool)
                    for b in range(6):
                        actual, valid = numpy_coefficients(block, 256 + 32 * b)
                        both = valid & old_ok
                        delta = (actual - previous) ** 2
                        own = delta[1:][:, diag][both[1:][:, diag]]
                        cross = delta[:, ~diag][both[:, ~diag]]
                        groups = [v.mean() for v in (own, cross) if v.size]
                        expected = np.mean(groups) if groups else 0.0
                        q = i + b * 32
                        errors.append(abs(expected - simple[q]))
                        has_value = valid[1:][:, diag].any() or valid[:, ~diag].any()
                        fused = full[q] if has_value else fallback[q]
                        mixes.append(abs(fused - z["joint"][q]))
        assert max(errors) < 1e-4 and max(mixes) < 1e-4, (
            domain,
            max(errors),
            max(mixes),
        )
        review[domain] = dict(
            entities=len(members),
            checked_real_blocks=len(errors),
            max_simple_abs_error=float(max(errors)),
            max_joint_abs_error=float(max(mixes)),
            sampled_entities=sorted({members[0], members[-1]}),
            matching_exposure_and_rng=True,
        )
    dump(root / "readout_review.json", review)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    check(parser.parse_args().workspace)
