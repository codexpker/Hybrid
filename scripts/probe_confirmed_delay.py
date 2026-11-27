"""CPU probe on the already sealed, synthetic coupled-AR banks."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from hybrid_tsad.confirmed_delay import (
    HISTORY,
    WINDOW,
    combine,
    fit_reference,
    score_region,
    score_window,
)

CORE = slice(96, 160)
EDITS = ("clean", "weak", "weak_control", "lag", "lag_amplitude", "synchronous")


def _sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def planted_oracle(x):
    """Known AR mechanism. This statistic is unavailable on public targets."""
    drivers = 2 * np.arange(x.shape[-1] // 4)
    t = HISTORY + np.arange(CORE.start, CORE.stop)
    y = x[:, t][:, :, drivers + 1] - 0.85 * x[:, t - 1][:, :, drivers + 1]
    errors = []
    for lag in (4, 16):
        d = x[:, t - lag][:, :, drivers] - 0.85 * x[:, t - lag - 1][:, :, drivers]
        e = y - 0.8 * d
        # Both hypotheses profile the same constant nuisance on the core.
        e -= e.mean(axis=1, keepdims=True)
        errors.append(np.mean(e**2, axis=1))
    return np.mean(errors[0] - errors[1], axis=1) / (2 * 0.6**2 * (1 - 0.85**2))


def compare(series, domain, method, panel):
    for kind, comparisons in (
        ("detection_vs_clean", [(edit, "clean") for edit in EDITS[1:]]),
        (
            "edit_preference",
            [
                ("weak", "weak_control"),
                ("lag", "lag_amplitude"),
                ("lag", "synchronous"),
            ],
        ),
    ):
        for edit, control in comparisons:
            a, b = series[edit], series[control]
            scores, labels = np.r_[a, b], np.repeat((1, 0), len(a))
            d = a - b
            yield dict(
                domain=domain,
                panel=panel,
                method=method,
                comparison=kind,
                edit=edit,
                reference=control,
                records=len(a),
                AP=float(average_precision_score(labels, scores)),
                ROC_AUC=float(roc_auc_score(labels, scores)),
                mean_delta=float(d.mean()),
                median_delta=float(np.median(d)),
                wins=int((d > 0).sum()),
                ties=int((d == 0).sum()),
                losses=int((d < 0).sum()),
            )


def probe(root, domain):
    catalog = root / "results/left_history_relation_20261126_v1" / domain / "controlled"
    path = catalog / "coupled_AR.npz"
    identity = json.loads((catalog / "input_receipt.json").read_text())
    assert _sha(path) == identity["arrays"][path.name]
    with np.load(path) as npz:
        data = {k: npz[k].astype(np.float64) for k in EDITS}
    clean = data["clean"]
    assert clean.shape == (64, HISTORY + WINDOW, {"PSM": 25, "SWAT": 31}[domain])
    assert all(
        np.array_equal(x[:, :HISTORY], clean[:, :HISTORY]) for x in data.values()
    )
    references = tuple(fit_reference(record[:HISTORY]) for record in clean)
    planted = {(i, i + 1, 4) for i in 2 * np.arange(clean.shape[-1] // 4)}
    n_correct = sum(
        (link.driver, link.receiver, link.delay) in planted
        for reference in references
        for link in reference.links
    )
    n_fit = sum(len(r.links) for r in references)
    oracle = {k: planted_oracle(v) for k, v in data.items()}
    rows = list(compare(oracle, domain, "planted_oracle", "diagnostic_core_64_records"))
    core = {
        k: [
            score_region(r, x[:HISTORY], x[HISTORY:], CORE.start, CORE.stop)
            for r, x in zip(references, records, strict=True)
        ]
        for k, records in data.items()
    }
    for method in ("delay", "residual"):
        statistics = {
            k: np.array([getattr(b, method) for b in v]) for k, v in core.items()
        }
        rows.extend(compare(statistics, domain, method, "diagnostic_core_64_records"))

    # Cache-integration exercise. Reserve the first 16 independent simulation
    # records for synthetic calibration. The planned benchmark uses SOURCE SELECT.
    score_path = (
        root
        / "results/left_timesfm_hidden_20261126_v1/controlled"
        / domain
        / "A_coupled_AR.npz"
    )
    score_receipt = json.loads(score_path.with_suffix(".json").read_text())
    assert _sha(score_path) == score_receipt["sha256"]
    assert score_receipt["input_archive_sha256"] == _sha(path)
    with np.load(score_path) as cache:
        native = {k: cache[k] for k in EDITS}
    assert all(
        native[k].shape == (64, WINDOW) and np.isfinite(native[k]).all() for k in EDITS
    )
    fit = np.arange(16)
    test = np.arange(16, 64)
    normal_native = native["clean"][fit].ravel()
    block_scores = {
        k: [
            score_window(r, x[:HISTORY], x[HISTORY:])
            for r, x in zip(references, v, strict=True)
        ]
        for k, v in data.items()
    }
    fixture = {"native": {k: s[test, CORE].mean(1) for k, s in native.items()}}
    for channel in ("delay", "residual"):
        normal = np.array(
            [getattr(t, channel) for i in fit for t in block_scores["clean"][i]]
        )
        fixture[f"native_plus_{channel}"] = {
            k: np.array(
                [
                    combine(
                        native[k][i], block_scores[k][i], normal_native, normal, channel
                    )[CORE].mean()
                    for i in test
                ]
            )
            for k in EDITS
        }
    # A calibration-only control separates the effect of the empirical rank
    # transform from the effect of the additional evidence.
    zero = type(block_scores["clean"][0][0])(0.0, 0.0, 0, 0, None, None)
    fixture["native_tail_only"] = {
        k: np.array(
            [
                combine(v[i], (zero,) * 3, normal_native, [0.0])[CORE].mean()
                for i in test
            ]
        )
        for k, v in native.items()
    }
    for method, stat in fixture.items():
        rows.extend(compare(stat, domain, method, "cache_integration_48_records"))

    metadata = dict(
        domain=domain,
        input_sha256=_sha(path),
        native_bank_sha256=_sha(score_path),
        learned_links=n_fit,
        planted_links_recovered=n_correct,
        planted_links_possible=64 * len(planted),
        native_feature_forwards=0,
        source_training_updates=0,
        synthetic_reference_records=fit.tolist(),
        synthetic_evaluation_records=test.tolist(),
    )
    return rows, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workspace", type=Path, help="Contains the original results directories"
    )
    parser.add_argument(
        "output", type=Path, help="New directory for the CPU probe results"
    )
    args = parser.parse_args()
    rows, metadata = [], []
    for domain in ("PSM", "SWAT"):
        a, b = probe(args.workspace, domain)
        rows.extend(a)
        metadata.append(b)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "receipt.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Checked {len(rows)} comparisons across the two synthetic widths")


if __name__ == "__main__":
    main()
