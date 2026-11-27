"""Seal holdout scores before reading labels and evaluate the fixed panels."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from . import left_history_control as controlled
from . import left_history_public as archive
from . import left_multisource as base
from . import left_timesfm_evaluation as prior
from . import left_holdout_training as train
from . import left_timesfm_training as prior_train
from . import timesfm_hidden
from .left_holdout import MASKS
from .left_timesfm_condition import HISTORY, WINDOW
from .trajectory_evaluation import save_rows

READOUT_BATCH = 32
DONOR_DISTANCE = math.ceil(HISTORY / WINDOW)


def historical_cache(w, domain, entity, descriptor, values, offset, starts, folder):
    caches, bindings = {}, {}
    for arm, name in (("B", "pretrained"), ("C", "random")):
        old = w / prior.OUT / "features" / domain / entity / "TEST" / f"{name}.npy"
        known_state = {
            "B": "b20ef55e020503c3fd059d624e06eb0bb0cf38108124eae4ad7f7f233fb7bf67",
            "C": "c9ca6c52b87bb5187298a876db5c6048de32df8e9de24f297b94d801bbe61680",
        }[arm]
        if old.is_file():
            metadata = base.read(old.with_suffix(".json"))
            assert metadata["inputs"] == descriptor
            assert metadata["encoder_state_sha256"] == known_state
            assert base.sha(old) == metadata["sha256"]
            history_hashes = metadata["history_sha256"]
            for s, digest in zip(starts, history_hashes, strict=True):
                assert (
                    prior.array_sha(values[s - offset - HISTORY : s - offset]) == digest
                )
            caches[arm] = np.load(old, mmap_mode="r")
            bindings[arm] = dict(
                sha256=metadata["sha256"],
                state_sha256=known_state,
                path=str(old.relative_to(w)),
                calls=0,
                seconds=0.0,
            )
            continue
        encoder = timesfm_hidden.load_encoder(w, random=arm == "C")
        assert timesfm_hidden.model_sha(encoder) == known_state
        began = time.monotonic()
        blocks = []
        for i in range(0, len(starts), READOUT_BATCH):
            rows = starts[i : i + READOUT_BATCH] - offset
            past = torch.from_numpy(
                values[rows[:, None] + np.arange(-HISTORY, 0)]
            ).cuda()
            lengths = torch.full((len(rows),), HISTORY, device="cuda", dtype=torch.long)
            f, valid = prior_train.feature_call(encoder, past, lengths)
            assert bool(valid.all())
            blocks.append(f.cpu())
        del encoder
        caches[arm] = torch.cat(blocks).numpy()
        bindings[arm] = dict(
            state_sha256=known_state,
            sha256=prior.array_sha(caches[arm]),
            calls=len(blocks),
            seconds=time.monotonic() - began,
            cache_storage="entity-local RAM",
        )
    base.dump(folder / "INPUTS.json", dict(**descriptor, historical_features=bindings))
    return caches


@torch.no_grad()
def score_model(model, arm, values, offset, starts, cache, donors=None):
    core = train.core(model, arm)
    fingerprint, step = base.fingerprint(model.state_dict()), core.step
    scores = []
    began = time.monotonic()
    for i in range(0, len(starts), READOUT_BATCH):
        rows = starts[i : i + READOUT_BATCH] - offset
        x = torch.from_numpy(values[rows[:, None] + np.arange(WINDOW)]).cuda()
        if arm == "native":
            score = model.infer(x, None, None, None)[0].mean(-1)
        else:
            z = valid = None
            if arm != "A":
                loc = (
                    slice(i, i + len(rows))
                    if donors is None
                    else donors[i : i + len(rows)]
                )
                z = torch.from_numpy(np.array(cache[loc])).cuda()
                valid = torch.ones(z.shape[:3], device="cuda", dtype=torch.bool)
            past = x.new_empty((len(x), 0, x.shape[-1]))
            lengths = torch.full((len(x),), HISTORY, device="cuda", dtype=torch.long)
            score = model.score(x, past, lengths, z, valid)
        assert torch.isfinite(score).all()
        scores.append(score.cpu().numpy().reshape(-1))
    assert step == core.step and fingerprint == base.fingerprint(model.state_dict())
    return np.concatenate(scores), time.monotonic() - began


def seal(
    folder, name, stage, model, arm, receipt, values, offset, starts, cache, donors=None
):
    path = folder / f"{stage}_{name}.npz"
    if path.with_suffix(".json").is_file():
        meta = base.read(path.with_suffix(".json"))
        assert (
            base.sha(path) == meta["sha256"] and meta["checkpoint"] == receipt["sha256"]
        )
        assert meta["input_receipt_sha256"] == base.sha(folder / "INPUTS.json")
        return
    selected = starts if donors is None else starts[DONOR_DISTANCE:]
    history_end = selected if donors is None else starts[donors]
    if donors is not None:
        assert np.all(history_end <= selected - HISTORY)
    scores, seconds = score_model(model, arm, values, offset, selected, cache, donors)
    np.savez_compressed(
        path,
        score=scores,
        index=(selected[:, None] + np.arange(WINDOW)).reshape(-1),
        current_start=np.repeat(selected, WINDOW),
        history_start=np.repeat(history_end - HISTORY, WINDOW),
        history_end_exclusive=np.repeat(history_end, WINDOW),
        available=np.repeat(selected + WINDOW - 1, WINDOW),
    )
    forwards = math.ceil(len(selected) / READOUT_BATCH) * (
        1 if arm == "native" else MASKS
    )
    base.dump(
        path.with_suffix(".json"),
        dict(
            sha256=base.sha(path),
            checkpoint=receipt["sha256"],
            input_receipt_sha256=base.sha(folder / "INPUTS.json"),
            stage=stage,
            arm=arm,
            nonoverlap_replacement=donors is not None,
            native_forwards=forwards,
            window_forwards=len(selected) * (1 if arm == "native" else MASKS),
            seconds=seconds,
            labels_read=False,
        ),
    )
    print("sealed", folder.name, stage, name, len(scores), seconds, flush=True)


def public(w, domain, entity, stage):
    folder = w / train.OUT / "public" / domain / entity
    folder.mkdir(parents=True, exist_ok=True)
    values, offset, starts, descriptor = prior.role_input(w, domain, entity, "TEST")
    caches = historical_cache(
        w, domain, entity, descriptor, values, offset, starts, folder
    )
    for arm in (*train.ARMS, "native") if stage == "N" else train.ARMS:
        model, ck = train.load(w, domain, stage, arm, entity)
        seal(
            folder, arm, stage, model, arm, ck, values, offset, starts, caches.get(arm)
        )
        if arm == "B":
            donors = np.arange(len(starts) - DONOR_DISTANCE)
            seal(
                folder,
                "B_nonoverlap",
                stage,
                model,
                arm,
                ck,
                values,
                offset,
                starts,
                caches[arm],
                donors,
            )
        del model


def controlled_bank(w, domain, stage):
    root = w / controlled.OUT / domain / "controlled"
    binding = base.read(root / "input_receipt.json")
    folder = w / train.OUT / "controlled" / domain
    folder.mkdir(parents=True, exist_ok=True)
    models = {arm: train.load(w, domain, stage, arm, domain) for arm in train.ARMS}
    for group in controlled.GROUPS:
        source = root / f"{group}.npz"
        source_hash = base.sha(source)
        assert source_hash == binding["arrays"][source.name]
        old = w / prior.OUT / "controlled" / domain
        with np.load(source) as z:
            raw = z["clean"]
            history = np.zeros((len(raw), HISTORY, raw.shape[-1]), np.float32)
            history[:, -256:] = raw[:, :256]
            past = torch.from_numpy(history).cuda()
            lengths = torch.full((len(raw),), 256, device="cuda", dtype=torch.long)
            caches, receipts = {}, {}
            for arm, name in (("B", "pretrained"), ("C", "random")):
                cache = old / f"{group}_{name}_features.npz"
                receipt = base.read(cache.with_suffix(".json"))
                assert receipt["sha256"] == base.sha(cache)
                assert receipt["input_archive_sha256"] == source_hash
                assert receipt["historical_input_sha256"] == prior.array_sha(history)
                assert receipt["encoder_state_sha256"] == (
                    "b20ef55e020503c3fd059d624e06eb0bb0cf38108124eae4ad7f7f233fb7bf67"
                    if arm == "B"
                    else "c9ca6c52b87bb5187298a876db5c6048de32df8e9de24f297b94d801bbe61680"
                )
                with np.load(cache) as f:
                    caches[arm] = tuple(
                        torch.from_numpy(f[k]).cuda() for k in ("features", "valid")
                    )
                receipts[arm] = receipt["sha256"]
            for arm, (model, ck) in models.items():
                path = folder / f"{stage}_{arm}_{group}.npz"
                if path.with_suffix(".json").is_file():
                    seen = base.read(path.with_suffix(".json"))
                    assert seen["checkpoint"] == ck["sha256"] and seen[
                        "sha256"
                    ] == base.sha(path)
                    continue
                state, step = base.fingerprint(model.state_dict()), model.left.step
                begin = time.monotonic()
                results = {}
                with torch.no_grad():
                    for edit in controlled.EDITS:
                        x = z[edit]
                        np.testing.assert_array_equal(x[:, :256], raw[:, :256])
                        rows = []
                        for i in range(0, len(x), READOUT_BATCH):
                            take = slice(i, i + READOUT_BATCH)
                            ctx = tuple(v[take] for v in caches.get(arm, ()))
                            rows.append(
                                model.score(
                                    torch.from_numpy(x[take, 256:]).cuda(),
                                    past[take],
                                    lengths[take],
                                    *ctx,
                                )
                                .cpu()
                                .numpy()
                            )
                        results[edit] = np.concatenate(rows)
                assert (
                    state == base.fingerprint(model.state_dict())
                    and step == model.left.step
                )
                np.savez_compressed(path, **results)
                base.dump(
                    path.with_suffix(".json"),
                    dict(
                        sha256=base.sha(path),
                        stage=stage,
                        checkpoint=ck["sha256"],
                        inputs=source_hash,
                        features=receipts.get(arm),
                        native_forwards=math.ceil(len(raw) / READOUT_BATCH)
                        * MASKS
                        * len(results),
                        window_forwards=len(raw) * MASKS * len(results),
                        seconds=time.monotonic() - begin,
                        nonoverlap_replacement="NA: archived controlled records have one legal history",
                    ),
                )


def read_bank(path):
    md = base.read(path.with_suffix(".json"))
    assert base.sha(path) == md["sha256"]
    with np.load(path) as z:
        index, score = z["index"], z["score"]
        if "history_end_exclusive" in z:
            assert np.all(z["history_end_exclusive"] <= z["current_start"])
        np.testing.assert_array_equal(z["available"], z["current_start"] + WINDOW - 1)
    assert np.isfinite(score).all()
    return index, score, md["sha256"]


def metrics(w, domain, entity):
    folder = w / train.OUT / "public" / domain / entity
    if (folder / "METRICS.json").exists():
        return
    rows = []
    current = {
        stage: {
            a: read_bank(folder / f"{stage}_{a}.npz")
            for a in (
                (*train.ARMS, "native", "B_nonoverlap")
                if stage == "N"
                else (*train.ARMS, "B_nonoverlap")
            )
        }
        for stage in ("Z", "N")
    }
    offset = 0 if domain == "SMD" else prior.ROLES[domain]["TEST"][0]
    md = base.read(folder / "INPUTS.json")
    assert md["labels_read"] is False
    end = md["role_stop"]
    external = archive.archived_scores(w, domain, entity, end - offset)
    external.update(
        {
            "e59_native_Z": read_bank(
                w / prior.OUT / "public" / domain / entity / "A_TEST.npz"
            ),
            "e59_history_Z": read_bank(
                w / prior.OUT / "public" / domain / entity / "C_TEST.npz"
            ),
        }
    )
    y = prior.truth(w, domain, entity) if domain == "SMD" else prior.labels(w, domain)
    index = current["Z"]["A"][0]
    assert index[0] == md["starts"][0] and index[-1] == md["starts"][-1] + WINDOW - 1
    ext_end = (
        min(index[-1] + 1, end // 100 * 100)
        if domain in ("PSM", "SMD")
        else index[-1] + 1
    )
    metric = prior.load_author_metric(w / prior.AUTHOR)
    previous_metrics = w / prior.OUT / "metrics" / domain / entity
    archive_rows = {}
    for file, alias in (
        ("archived_TEST.json", None),
        ("A_TEST.json", "e59_native_Z"),
        ("C_TEST.json", "e59_history_Z"),
    ):
        for row in base.read(previous_metrics / file)["rows"]:
            name = alias or row["method"]
            assert row["score_sha256"] == external[name][2]
            archive_rows[(name, row["start"], row["stop"])] = row
    measured_rows = {}
    for stage, scores in current.items():
        ablation = scores["B_nonoverlap"][0]
        for panel, axis, methods in (
            (
                "complete",
                index,
                {a: b for a, b in scores.items() if a != "B_nonoverlap"},
            ),
            ("nonoverlap", ablation, {a: scores[a] for a in ("B", "B_nonoverlap")}),
            (
                "external",
                index[index < ext_end],
                {
                    **external,
                    **{a: b for a, b in scores.items() if a != "B_nonoverlap"},
                },
            ),
        ):
            for name, (ix, s, digest) in methods.items():
                loc = np.searchsorted(ix, axis)
                np.testing.assert_array_equal(ix[loc], axis)
                key = (name, int(axis[0]), int(axis[-1] + 1))
                saved = archive_rows.get(key)
                if saved:
                    assert saved["positives"] == int(y[axis].sum())
                    measured = {m: saved[m] for m in prior.METRICS}
                    error = saved["independent_AP_ROC_error"]
                else:
                    value_key = (digest, *key[1:])
                    if value_key not in measured_rows:
                        measured_rows[value_key] = prior.ranking(
                            y[axis], s[loc], metric
                        )
                    measured, error = measured_rows[value_key]
                rows.append(
                    dict(
                        domain=domain,
                        entity=entity,
                        stage=stage,
                        panel=panel,
                        method=name,
                        start=int(axis[0]),
                        stop=int(axis[-1] + 1),
                        points=len(axis),
                        positives=int(y[axis].sum()),
                        score_sha256=digest,
                        AP_ROC_check_error=error,
                        **measured,
                    )
                )
    base.dump(folder / "METRICS.json", rows)


def controlled_metrics(w, domains):
    rows = []
    for domain in domains:
        for stage in ("Z", "N"):
            for arm in train.ARMS:
                for group in controlled.GROUPS:
                    path = (
                        w
                        / train.OUT
                        / "controlled"
                        / domain
                        / f"{stage}_{arm}_{group}.npz"
                    )
                    assert (
                        base.sha(path) == base.read(path.with_suffix(".json"))["sha256"]
                    )
                    with np.load(path) as bank:
                        for region, indices in controlled.REGIONS.items():
                            scores = {
                                k: bank[k][:, indices].mean(1) for k in bank.files
                            }
                            for edit, ref in (
                                ("weak", "clean"),
                                ("weak", "weak_control"),
                                ("weak_control", "clean"),
                                ("lag", "clean"),
                                ("lag", "lag_amplitude"),
                                ("lag", "synchronous"),
                                ("lag_amplitude", "clean"),
                                ("synchronous", "clean"),
                            ):
                                a, b = scores[edit], scores[ref]
                                direction = a - b
                                y = np.r_[np.ones(len(a)), np.zeros(len(b))]
                                s = np.r_[a, b]
                                rows.append(
                                    dict(
                                        domain=domain,
                                        stage=stage,
                                        arm=arm,
                                        group=group,
                                        region=region,
                                        edit=edit,
                                        reference=ref,
                                        n=len(a),
                                        AP=float(average_precision_score(y, s)),
                                        ROC_AUC=float(roc_auc_score(y, s)),
                                        mean_delta=float(direction.mean()),
                                        median_delta=float(np.median(direction)),
                                        wins=int((direction > 0).sum()),
                                        ties=int((direction == 0).sum()),
                                        losses=int((direction < 0).sum()),
                                    )
                                )
    save_rows(w / train.OUT / "controlled" / "comparisons.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("domain", choices=("PSM", "SWAT", "SMD"))
    parser.add_argument(
        "action", choices=("score", "metrics", "controlled", "controlled-metrics")
    )
    parser.add_argument("--stage", choices=("Z", "N"), default="Z")
    parser.add_argument("--entity")
    args = parser.parse_args()
    torch.set_num_threads(1)
    entity = args.entity or args.domain
    if args.action == "score":
        public(args.workspace, args.domain, entity, args.stage)
    elif args.action == "metrics":
        metrics(args.workspace, args.domain, entity)
    elif args.action == "controlled":
        controlled_bank(args.workspace, args.domain, args.stage)
    else:
        controlled_metrics(args.workspace, ("PSM", "SWAT"))
