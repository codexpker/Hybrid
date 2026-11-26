"""Paired, evaluation-only lag interventions on independent AR records and public normal blocks."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path("results/left_history_relation_20261126_v1")
DOMAINS = {"PSM": 25, "SWAT": 31}
GROUPS = {"coupled_AR": 64, "normal_SELECT": 16}
ARMS = ("source", "continue", "joint")
EDITS = ("clean", "weak", "weak_control", "lag", "lag_amplitude", "synchronous")
REGIONS = {
    "core": np.arange(96, 160),
    "transition": np.r_[0:32, 160:192],
    "full": np.arange(192),
}
SEED = 20261126


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def arrays(w, domain, group):
    width = DOMAINS[domain]
    n = GROUPS[group]
    if group == "normal_SELECT":
        from .left_multisource import target_inputs

        normal, cut, _, data_hash = target_inputs(w, domain)
        starts = np.linspace(cut, len(normal) - 448, n, dtype=int)
        assert min(np.diff(starts)) >= 448 and min(starts) >= cut
        return normal[starts[:, None] + np.arange(448)], starts, data_hash

    # First half in disjoint driver/receiver pairs. Remaining channels stay AR(1).
    drivers = 2 * np.arange(width // 4)
    samples = []
    for i in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, width, i]))
        z = rng.standard_normal((640, width))
        for t in range(1, len(z)):
            z[t] = 0.85 * z[t - 1] + np.sqrt(1 - 0.85**2) * z[t]
        raw = z[-448:].copy()
        raw[:, drivers + 1] = 0.8 * z[-452:-4, drivers] + 0.6 * raw[:, drivers + 1]
        samples.append(raw)
    return (
        np.stack(samples),
        np.zeros(n, dtype=int),
        "SeedSequence([20261126,width,record])",
    )


def interventions(clean):
    n, _, width = clean.shape
    drivers = 2 * np.arange(width // 4)
    receivers = drivers + 1
    # Zero at the outer endpoints. Exactly one at samples 31..160.
    rise = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 32)))
    env = np.r_[rise, np.ones(128), rise[::-1]][None, :, None]
    sign = np.where(
        (np.arange(n)[:, None, None] + np.arange(len(receivers))) % 2, -1.0, 1.0
    )
    current = clean[:, 256:, :]
    lag_delta = env * (clean[:, 244:436, :] - current)
    weak_delta = 0.25 * env * sign
    bipolar = env * np.cos(np.pi * (np.arange(192)[None, :, None] + 0.5) / 192)

    def energy_match(shape, delta):
        scale = np.sqrt(
            np.sum(delta**2, axis=1, keepdims=True)
            / np.sum(shape**2, axis=1, keepdims=True)
        )
        return shape * scale * sign

    def add(delta, channels):
        x = clean.copy()
        x[:, 256:, channels] += delta
        return x.astype(np.float32)

    return {
        "clean": clean.astype(np.float32),
        "weak": add(weak_delta, receivers),
        "weak_control": add(energy_match(bipolar, weak_delta), receivers),
        "lag": add(lag_delta[:, :, receivers], receivers),
        "lag_amplitude": add(energy_match(env, lag_delta[:, :, receivers]), receivers),
        "synchronous": add(
            lag_delta[:, :, : 2 * len(drivers)], np.arange(2 * len(drivers))
        ),
    }


def known_pair_error(x, delay):
    """Descriptive signed Pearson correlation of the manipulated pairs on the core."""
    d = 2 * np.arange(x.shape[-1] // 4)
    t = 256 + REGIONS["core"]
    driver = x[:, t - delay][:, :, d].astype(np.float64)
    receiver = x[:, t][:, :, d + 1].astype(np.float64)
    driver -= driver.mean(1, keepdims=True)
    receiver -= receiver.mean(1, keepdims=True)
    sd = np.sqrt(np.mean(driver**2, 1) * np.mean(receiver**2, 1))
    prod = np.mean(driver * receiver, 1)
    corr = np.divide(prod, sd, out=np.zeros_like(prod), where=sd > 1e-10)
    valid = (driver.std(1) > 1e-5) & (receiver.std(1) > 1e-5)
    return np.divide(
        (corr * valid).sum(1),
        valid.sum(1),
        out=np.full(len(x), np.nan),
        where=valid.sum(1) > 0,
    ), valid.sum(1)


def prepare(w, domain):
    dest = w / OUT / domain / "controlled"
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest / "input_receipt.json"
    if receipt.exists():
        for name, digest_value in json.loads(receipt.read_text())["arrays"].items():
            assert digest(dest / name) == digest_value
        return
    meta = {
        "arrays": {},
        "origin": {},
        "seed": SEED,
        "positions": list(EDITS),
        "producer_sha256": digest(Path(__file__)),
        "width": DOMAINS[domain],
    }
    rows = []
    for group in GROUPS:
        x, starts, origin = arrays(w, domain, group)
        edits = interventions(np.asarray(x, dtype=np.float64))
        baseline = edits["clean"]
        rec = 2 * np.arange(DOMAINS[domain] // 4) + 1
        mask = np.ones(x.shape[-1], bool)
        mask[rec] = False
        energies = {}
        roundoff = {}
        for name, edited in edits.items():
            assert np.isfinite(edited).all()
            np.testing.assert_array_equal(edited[:, :256], baseline[:, :256])
            np.testing.assert_array_equal(
                edited[:, [256, 447]], baseline[:, [256, 447]]
            )
            if name != "synchronous":
                np.testing.assert_array_equal(edited[:, :, mask], baseline[:, :, mask])
            change = edited[:, 256:].astype(float) - baseline[:, 256:]
            energies[name] = np.sum(change[:, :, rec] ** 2, 1)
            # Bound the effect of casting both endpoints of a difference to f32.
            magnitude = np.abs(edited[:, 256:, rec].astype(float)) + np.abs(
                baseline[:, 256:, rec].astype(float)
            )
            radius = np.finfo(np.float32).eps * np.linalg.norm(magnitude, axis=1)
            roundoff[name] = 2 * np.sqrt(energies[name]) * radius + radius**2
            raw = edited[:, 256 + REGIONS["core"]][:, :, rec].astype(float)
            old = baseline[:, 256 + REGIONS["core"]][:, :, rec].astype(float)
            corr4, pairs4 = known_pair_error(edited, 4)
            corr16, pairs16 = known_pair_error(edited, 16)
            for i in range(len(x)):
                rows.append(
                    dict(
                        domain=domain,
                        group=group,
                        record=i,
                        start=int(starts[i]),
                        edit=name,
                        energy=float(np.sum(change[i] ** 2)),
                        rms=float(np.sqrt(np.mean(change[i] ** 2))),
                        peak=float(np.abs(change[i]).max()),
                        pairs_lag4=pairs4[i],
                        pairs_lag16=pairs16[i],
                        correlation_lag4=corr4[i],
                        correlation_lag16=corr16[i],
                        receiver_core_abs_mean_shift=float(
                            np.abs(raw[i].mean(0) - old[i].mean(0)).mean()
                        ),
                        receiver_core_abs_sd_shift=float(
                            np.abs(raw[i].std(0) - old[i].std(0)).mean()
                        ),
                    )
                )
        for target, control in (("weak", "weak_control"), ("lag", "lag_amplitude")):
            err = np.abs(energies[target] - energies[control])
            assert np.all(err <= roundoff[target] + roundoff[control] + 1e-12)
        p = dest / f"{group}.npz"
        np.savez_compressed(p, starts=starts, **edits)
        meta["arrays"][p.name] = digest(p)
        meta["origin"][group] = origin
    pd.DataFrame(rows).to_csv(dest / "input_diagnostics.csv", index=False)
    meta["input_diagnostics_sha256"] = digest(dest / "input_diagnostics.csv")
    receipt.write_text(json.dumps(meta, indent=2))


def score(w, domain):
    import torch
    from .left_history_relation import load_model, score_batch

    dest = w / OUT / domain / "controlled"
    prep = json.loads((dest / "input_receipt.json").read_text())
    times = []
    for arm in ARMS:
        model, scales = load_model(w, domain, arm)
        model.eval()
        for group in GROUPS:
            out = dest / f"{arm}_{group}_scores.npz"
            if out.exists():
                raise FileExistsError(out)
            path = dest / f"{group}.npz"
            assert digest(path) == prep["arrays"][path.name]
            torch.cuda.synchronize()
            tic = time.monotonic()
            bank = {}
            with np.load(path) as data, torch.no_grad():
                for edit in EDITS:
                    series = {}
                    x = data[edit]
                    for j in range(0, len(x), 8):
                        batch = torch.from_numpy(x[j : j + 8]).cuda()
                        scores = score_batch(model, batch, scales)
                        for key, value in scores.items():
                            a = torch.as_tensor(value).detach().cpu().numpy()
                            assert a.shape == (len(batch), 192) and np.isfinite(a).all()
                            series.setdefault(key, []).append(a)
                    for key, values in series.items():
                        bank[f"{edit}__{key}"] = np.concatenate(values)
            torch.cuda.synchronize()
            seconds = time.monotonic() - tic
            np.savez_compressed(out, **bank)
            times.append(
                dict(
                    domain=domain,
                    arm=arm,
                    group=group,
                    seconds=seconds,
                    windows=GROUPS[group] * len(EDITS),
                    sha256=digest(out),
                )
            )
    pd.DataFrame(times).to_csv(dest / "inference.csv", index=False)


def summarize(w, folder):
    pairs = []
    per_record = []
    for domain in DOMAINS:
        dest = w / OUT / domain / "controlled"
        for group in GROUPS:
            for arm in ARMS:
                with np.load(dest / f"{arm}_{group}_scores.npz") as bank:
                    for key in sorted({k.split("__")[1] for k in bank.files}):
                        for region, t in REGIONS.items():
                            vals = {
                                e: bank[f"{e}__{key}"][:, t].mean(1, dtype=np.float64)
                                for e in EDITS
                            }
                            n = len(vals["clean"])
                            common = dict(
                                domain=domain,
                                group=group,
                                arm=arm,
                                readout=key,
                                region=region,
                            )
                            for i in range(n):
                                per_record.append(
                                    dict(
                                        **common,
                                        record=i,
                                        **{e: s[i] for e, s in vals.items()},
                                    )
                                )
                            for test, ref in (
                                ("weak", "clean"),
                                ("weak", "weak_control"),
                                ("lag", "clean"),
                                ("lag", "lag_amplitude"),
                                ("lag", "synchronous"),
                                ("synchronous", "clean"),
                                ("weak_control", "clean"),
                                ("lag_amplitude", "clean"),
                            ):
                                delta = vals[test] - vals[ref]
                                y = np.r_[np.zeros(n), np.ones(n)]
                                scores = np.r_[vals[ref], vals[test]]
                                pairs.append(
                                    dict(
                                        **common,
                                        target=test,
                                        reference=ref,
                                        n=n,
                                        mean_delta=delta.mean(),
                                        median_delta=np.median(delta),
                                        wins=int(np.sum(delta > 0)),
                                        ties=int(np.sum(delta == 0)),
                                        losses=int(np.sum(delta < 0)),
                                        ROC_AUC=roc_auc_score(y, scores),
                                        AP=average_precision_score(y, scores),
                                    )
                                )
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_csv(folder / "controlled_pairs.csv", index=False)
    pd.DataFrame(per_record).to_csv(
        folder / "controlled_records.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("task", choices=("prepare", "score", "summarize"))
    p.add_argument("--domain", choices=DOMAINS)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if args.task == "summarize":
        summarize(args.workspace, args.report)
    else:
        {"prepare": prepare, "score": score}[args.task](args.workspace, args.domain)
