"""Label-free input footprints and sealed-score evaluation for TimesFM-conditioned LEFT."""

import argparse
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import left_history_public as previous
from . import left_multisource as multi
from . import left_smd_targets as smd
from .left_decoder_context_metrics import AUTHOR
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .left_swat_author import dump, read, sha
from .native_full_window_metrics import truth
from .trajectory_evaluation import labels, independent_check, PSM_CAL, SW_CAL
from .trajectory_experiment import ROLES, target

OUT = Path("results/left_timesfm_hidden_20261126_v1")
WINDOW, HISTORY = 192, 512
ARMS = ("A", "B", "C", "D", "E")
METRICS = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def role_input(w, domain, entity, role):
    """Return values confined to one permitted role and the original role origin."""
    if domain == "SMD":
        rec = smd.smd.bindings(w)[entity]
        normal, scaler = smd.normal_scaler(w, rec, role == "TEST")
        raw = normal if role == "CAL" else smd.smd.checked_data(w, rec, "test")
        lo, hi = smd.roles(rec)[role]
        anchor = 0
        scaler_role = smd.roles(rec)[
            "public_scaler" if role == "TEST" else "scaler_fit"
        ]
        record = entity + ("/train" if role == "CAL" else "/test")
        full = raw[lo:hi]
    else:
        _, _, scaler, _ = multi.target_inputs(w, domain)
        full = target(w, domain, role)
        lo, hi = ROLES[domain][role]
        anchor = ROLES[domain]["TEST"][0]
        scaler_role = [0, ROLES[domain]["SELECT"][1]]
        record = domain + "/" + ("test" if domain == "PSM" else "original")
    values = scaler.transform(full).astype(np.float32)
    assert values.shape[0] == hi - lo and np.isfinite(values).all()
    start = anchor + ((lo + HISTORY - anchor + WINDOW - 1) // WINDOW) * WINDOW
    starts = np.arange(start, hi - WINDOW + 1, WINDOW, dtype=np.int64)
    assert len(starts) and starts[0] - HISTORY >= lo
    descriptor = dict(
        domain=domain,
        entity=entity,
        role=role,
        record_id=record,
        role_start=lo,
        role_stop=hi,
        grid_origin=anchor,
        history=HISTORY,
        current=WINDOW,
        starts=starts.tolist(),
        channel_mapping=list(range(values.shape[1])),
        raw_sha256=array_sha(full),
        input_sha256=array_sha(values),
        scaler_mean_sha256=array_sha(scaler.mean_),
        scaler_scale_sha256=array_sha(scaler.scale_),
        scaler_role=scaler_role,
        labels_read=False,
    )
    return values, lo, starts, descriptor


def cal_points(w, domain, entity):
    if domain != "SMD":
        return PSM_CAL if domain == "PSM" else SW_CAL
    rec = smd.smd.bindings(w)[entity]
    lo, hi = smd.roles(rec)["CAL"]
    first, stop = ((lo + WINDOW - 1) // WINDOW) * WINDOW, hi // WINDOW * WINDOW
    return np.linspace(first, stop - 1, 512, dtype=np.int64)


def freeze_axes(w, domains):
    """Bind time support before labels are loaded by metrics."""
    rows = []
    for domain in domains:
        for entity in previous.members(w, domain):
            for role in ("TEST", "U", "CAL", "QUERY"):
                _, lo, starts, desc = role_input(w, domain, entity, role)
                if role == "CAL":
                    wanted = cal_points(w, domain, entity)
                    valid = (wanted >= starts[0]) & (wanted < starts[-1] + WINDOW)
                    desc.update(
                        cal_requested=wanted.tolist(),
                        cal_legal_count=int(valid.sum()),
                        cal_complete=bool(valid.all()),
                        operations="eligible"
                        if valid.all()
                        else "NA: original CAL points lack full legal history",
                    )
                rows.append(desc)
    dest = w / OUT / "TARGET_INPUTS.json"
    dest.parent.mkdir(exist_ok=True, parents=True)
    record = dict(history=HISTORY, window=WINDOW, records=rows, labels_read=False)
    if dest.exists():
        assert read(dest) == record
    else:
        dump(dest, record)
    donor_rows, external_rows = [], []
    for r in rows:
        starts = np.asarray(r["starts"], dtype=np.int64)
        donor = history_donors(len(starts), r["domain"], r["entity"], r["role"])
        assert np.all(starts[donor] <= starts)
        assert np.all(starts[donor] - HISTORY >= r["role_start"])
        donor_rows.append(
            dict(
                domain=r["domain"],
                entity=r["entity"],
                role=r["role"],
                recipient_current_start=starts.tolist(),
                donor_history_end=starts[donor].tolist(),
                donor_history_start=(starts[donor] - HISTORY).tolist(),
                donor_index=donor.tolist(),
                identity_count=int((donor == np.arange(len(starts))).sum()),
            )
        )
        if r["role"] != "TEST":
            continue
        n = r["role_stop"] - r["role_start"]
        archives = previous.archived_scores(w, r["domain"], r["entity"], n)
        first, last = int(starts[0]), int(starts[-1] + WINDOW)
        external_end = (
            min(last, n // 100 * 100) if r["domain"] in ("PSM", "SMD") else last
        )
        for name, (index, values, digest) in archives.items():
            stop = (
                external_end
                if name in ("dada", "timercd", "timeradar_endpoint")
                else last
            )
            points = np.arange(first, stop)
            loc = np.searchsorted(index, points)
            assert np.all(loc < len(index)), (r["domain"], r["entity"], name)
            np.testing.assert_array_equal(index[loc], points)
            assert np.isfinite(values[loc]).all()
            external_rows.append(
                dict(
                    domain=r["domain"],
                    entity=r["entity"],
                    method=name,
                    start=first,
                    stop=stop,
                    points=len(points),
                    sha256=digest,
                )
            )
    evidence = dict(
        labels_read=False,
        donor_rule="deterministic isolated RNG, previous same-record legal grid history, identity at first legal window only",
        donors=donor_rows,
        archived_banks=external_rows,
    )
    path = w / OUT / "DONORS_AND_ARCHIVE_AXES.json"
    if path.exists():
        assert read(path) == evidence
    else:
        dump(path, evidence)
    mixed = []
    for r in rows:
        if r["role"] != "QUERY":
            continue
        starts = np.asarray(r["starts"])
        points = np.arange(starts[0], starts[-1] + WINDOW)
        for name, (index, values, digest) in archived_mixed(
            w, r["domain"], r["entity"]
        ).items():
            loc = np.searchsorted(index, points)
            assert np.all(loc < len(index))
            np.testing.assert_array_equal(index[loc], points)
            assert np.isfinite(values[loc]).all()
            mixed.append(
                dict(
                    domain=r["domain"],
                    entity=r["entity"],
                    method=name,
                    start=int(points[0]),
                    stop=int(points[-1] + 1),
                    points=len(points),
                    sha256=digest,
                )
            )
    path = w / OUT / "MIXED_ARCHIVE_AXES.json"
    if path.exists():
        assert read(path) == dict(labels_read=False, records=mixed)
    else:
        dump(path, dict(labels_read=False, records=mixed))
    return record


def archived_mixed(w, domain, entity):
    out = {}
    if domain == "SMD":
        from .left_smd_reliability_eval import bank as old_bank

        for arm in (
            "source",
            "random_u",
            "ordinary",
            "replay",
            "window",
            "source_only",
        ):
            i, s, _, digest = old_bank(w, entity, arm, "QUERY")
            out["old_" + arm] = i, s, digest
    else:
        from .left_multisource_eval import read_score

        for arm in ("source", "random_u", "ordinary", "replay"):
            i, s, _ = read_score(w, domain, arm, "QUERY")
            receipt = read(w / multi.OUT / domain / "banks" / (arm + "_QUERY.json"))
            out["old_" + arm] = i, s, receipt["sha256"]
        path = (
            w / "results/timesfm3_frozen_20261121_v1/score" / domain / "QUERY/BANK.npz"
        )
        digest = read(path.with_name("SEALED.json"))["sha256"]
        assert sha(path) == digest
        with np.load(path) as z:
            out["timesfm_forecast"] = (
                z["absolute_time_index"],
                z["canonical_score"],
                digest,
            )
    return out


def history_donors(count, domain, entity, role):
    # Older windows preserve the strict past restriction at every recipient.
    identity = f"2025/{domain}/{entity}/{role}/history_donors".encode()
    seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return np.array(
        [0, *[int(rng.integers(i)) for i in range(1, count)]], dtype=np.int64
    )


def seal_score(w, domain, entity, arm, role, scores, metadata, elapsed, checkpoint):
    starts = np.array(metadata["starts"], dtype=np.int64)
    scores = np.asarray(scores)
    assert scores.shape == (len(starts), WINDOW) and np.isfinite(scores).all()
    index = (starts[:, None] + np.arange(WINDOW)).ravel()
    history_end = np.asarray(
        metadata.get("donor_current_start", starts), dtype=np.int64
    )
    assert history_end.shape == starts.shape and np.all(history_end <= starts)
    folder = w / OUT / "public" / domain / entity
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / f"{arm}_{role}.npz"
    np.savez_compressed(
        path,
        index=index,
        score=scores.ravel(),
        current_start=np.repeat(starts, WINDOW),
        history_start=np.repeat(history_end - HISTORY, WINDOW),
        history_end_exclusive=np.repeat(history_end, WINDOW),
        recipient_current_start=np.repeat(starts, WINDOW),
        available=np.repeat(starts + WINDOW - 1, WINDOW),
    )
    dump(
        path.with_suffix(".json"),
        dict(
            **metadata,
            arm=arm,
            seconds=elapsed,
            checkpoint=checkpoint,
            sha256=sha(path),
            points=len(index),
            source="native LEFT infer",
        ),
    )
    return path


def feature_bank(w, domain, entity, role, encoder, random=False):
    """Cache unchanged fp32 variable/patch tokens with the original input identity."""
    import torch
    from .timesfm_hidden import encode, model_sha, WEIGHT_SHA, SOURCE_COMMIT

    values, lo, starts, meta = role_input(w, domain, entity, role)
    dest = w / OUT / "features" / domain / entity / role
    dest.mkdir(exist_ok=True, parents=True)
    name = "random" if random else "pretrained"
    path = dest / f"{name}.npy"
    receipt = path.with_suffix(".json")
    if receipt.exists():
        old = read(receipt)
        parent = w / old["parent_file"] if "parent_file" in old else path
        assert sha(parent) == old["sha256"] and old["inputs"] == meta
        hidden = np.load(parent, mmap_mode="r")
        if "parent_file" in old:
            hidden = hidden[old["parent_rows"][0] : old["parent_rows"][1]]
        return hidden, old
    parent = w / OUT / "features" / domain / entity / "TEST" / (name + ".npy")
    if domain != "SMD" and role != "TEST" and parent.with_suffix(".json").exists():
        identity = read(parent.with_suffix(".json"))
        assert sha(parent) == identity["sha256"]
        ref = identity["inputs"]
        assert all(
            ref[k] == meta[k]
            for k in ("scaler_mean_sha256", "scaler_scale_sha256", "channel_mapping")
        )
        location = np.searchsorted(ref["starts"], starts)
        np.testing.assert_array_equal(np.asarray(ref["starts"])[location], starts)
        assert np.all(np.diff(location) == 1)
        for i, origin in enumerate(starts):
            observed = array_sha(values[origin - lo - HISTORY : origin - lo])
            assert observed == identity["history_sha256"][location[i]]
        identity = dict(identity)
        identity.update(
            inputs=meta,
            shape=[len(starts), values.shape[1], 16, 1280],
            parent_file=str(parent.relative_to(w)),
            parent_rows=[int(location[0]), int(location[-1] + 1)],
            history_sha256=[identity["history_sha256"][i] for i in location],
            seconds=0.0,
        )
        dump(receipt, identity)
        return np.load(parent, mmap_mode="r")[location[0] : location[-1] + 1], identity
    before = model_sha(encoder)
    shaped = (len(starts), values.shape[1], 16, 1280)
    hidden = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shaped)
    input_hashes = []
    torch.cuda.synchronize()
    began = time.monotonic()
    for i in range(0, len(starts), 4):
        selected = starts[i : i + 4]
        history = np.ascontiguousarray(
            values[selected[:, None] - lo + np.arange(-HISTORY, 0)]
        )
        input_hashes.extend(array_sha(x) for x in history)
        length = torch.full((len(history),), HISTORY, device="cuda", dtype=torch.long)
        features, valid = encode(encoder, torch.from_numpy(history).cuda(), length)
        assert valid.all()
        hidden[i : i + len(history)] = features.cpu().numpy()
    torch.cuda.synchronize()
    hidden.flush()
    elapsed = time.monotonic() - began
    assert model_sha(encoder) == before and all(
        p.grad is None for p in encoder.parameters()
    )
    identity = dict(
        inputs=meta,
        encoder_state_sha256=before,
        checkpoint_sha256=None if random else WEIGHT_SHA,
        code_commit=SOURCE_COMMIT,
        random_initialization=random,
        shape=list(shaped),
        dtype="float32",
        sha256=sha(path),
        seconds=elapsed,
        history_sha256=input_hashes,
    )
    dump(receipt, identity)
    return hidden, identity


def features_only(w, domain, entities, role, arms):
    from .timesfm_hidden import load_encoder

    for random in (False, True):
        if random and "E" not in arms:
            continue
        if not random and not any(a.removesuffix("_U") in ("C", "D") for a in arms):
            continue
        encoder = load_encoder(w, random=random)
        for entity in entities:
            feature_bank(w, domain, entity, role, encoder, random)
            print("features sealed", domain, entity, role, random, flush=True)
        del encoder


def score_models(w, domain, entities, role, arms):
    import torch
    from .left_timesfm_condition import load_model, initial
    from .timesfm_hidden import load_encoder

    encoders = {}
    if any(a.removesuffix("_U") in ("C", "D") for a in arms):
        encoders[False] = load_encoder(w)
    if "E" in arms:
        encoders[True] = load_encoder(w, random=True)
    for entity in entities:
        models = {}
        for name in arms:
            arm = name.removesuffix("_U")
            if name.endswith("_U"):
                folder = w / OUT / domain / "U" / entity / arm
                ck = read(folder / "receipt.json")
                assert ck["sha256"] == sha(folder / "selected.pt")
                model, _ = initial(w, domain, arm)
                state = torch.load(
                    folder / "selected.pt", map_location="cpu", weights_only=True
                )
                model.load_state_dict(state["state"], strict=True)
                model.left.step = state["step"]
                models[name] = model.eval(), ck
            else:
                models[name] = load_model(w, domain, arm)
        values, lo, starts, desc = role_input(w, domain, entity, role)
        caches = {
            r: feature_bank(w, domain, entity, role, en, r)
            for r, en in encoders.items()
        }
        donors = history_donors(len(starts), domain, entity, role)
        for name, (model, checkpoint) in models.items():
            arm = name.removesuffix("_U")
            path = w / OUT / "public" / domain / entity / f"{name}_{role}.npz"
            if path.with_suffix(".json").exists():
                old = read(path.with_suffix(".json"))
                assert (
                    sha(path) == old["sha256"]
                    and old["checkpoint"] == checkpoint["sha256"]
                )
                continue
            model.eval()
            state, step = multi.fingerprint(model.state_dict()), model.left.step
            out = []
            torch.cuda.synchronize()
            began = time.monotonic()
            for i in range(0, len(starts), 32):
                selected = starts[i : i + 32]
                current = np.ascontiguousarray(
                    values[selected[:, None] - lo + np.arange(WINDOW)]
                )
                history = np.ascontiguousarray(
                    values[selected[:, None] - lo + np.arange(-HISTORY, 0)]
                )
                feature = valid = None
                if arm in ("C", "D", "E"):
                    cache = caches[arm == "E"][0]
                    loc = (
                        donors[i : i + len(current)]
                        if arm == "D"
                        else np.arange(i, i + len(current))
                    )
                    feature = torch.from_numpy(np.array(cache[loc])).cuda()
                    valid = torch.ones(
                        feature.shape[:3], device="cuda", dtype=torch.bool
                    )
                lengths = torch.full(
                    (len(current),), HISTORY, device="cuda", dtype=torch.long
                )
                out.append(
                    model.infer(
                        torch.from_numpy(current).cuda(),
                        torch.from_numpy(history).cuda(),
                        lengths,
                        feature,
                        valid,
                    )[0]
                    .mean(-1)
                    .cpu()
                    .numpy()
                )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - began
            assert model.left.step == step and state == multi.fingerprint(
                model.state_dict()
            )
            receipt = dict(desc)
            if arm in ("C", "D", "E"):
                receipt["feature_cache_sha256"] = caches[arm == "E"][1]["sha256"]
            if arm == "D":
                receipt["history_donors"] = donors.tolist()
                receipt["self_assigned_histories"] = int(
                    (donors == np.arange(len(donors))).sum()
                )
                receipt["donor_current_start"] = starts[donors].tolist()
                receipt["donor_rule"] = (
                    "uniform previous legal window within record, first window identity"
                )
            seal_score(
                w,
                domain,
                entity,
                name,
                role,
                np.concatenate(out),
                receipt,
                elapsed,
                checkpoint["sha256"],
            )
            print(
                "sealed", domain, entity, name, role, len(starts) * WINDOW, flush=True
            )
        del models


def controlled_score(w, domain, arms):
    """Reuse archived edits exactly, with 256 legitimate past observations."""
    import torch
    from . import left_history_control as old
    from .left_timesfm_condition import load_model
    from .timesfm_hidden import encode, load_encoder, model_sha, WEIGHT_SHA

    old_root = w / old.OUT / domain / "controlled"
    old_meta = read(old_root / "input_receipt.json")
    folder = w / OUT / "controlled" / domain
    folder.mkdir(exist_ok=True, parents=True)
    models = {a: load_model(w, domain, a) for a in arms}
    encoders = {}
    if any(a in arms for a in ("C", "D")):
        encoders[False] = load_encoder(w)
    if "E" in arms:
        encoders[True] = load_encoder(w, random=True)
    encoder_hashes = {random: model_sha(model) for random, model in encoders.items()}
    for group in old.GROUPS:
        path = old_root / f"{group}.npz"
        assert sha(path) == old_meta["arrays"][path.name]
        with np.load(path) as archived:
            clean = archived["clean"]
            n, _, c = clean.shape
            history = np.zeros((n, HISTORY, c), dtype=np.float32)
            history[:, -256:] = clean[:, :256]
            valid_lengths = torch.full((n,), 256, dtype=torch.long, device="cuda")
            hs = torch.from_numpy(history).cuda()
            caches = {}
            for random, encoder in encoders.items():
                fp = (
                    folder
                    / f"{group}_{'random' if random else 'pretrained'}_features.npz"
                )
                if fp.with_suffix(".json").exists():
                    identity = read(fp.with_suffix(".json"))
                    assert sha(fp) == identity["sha256"]
                    assert identity["input_archive_sha256"] == sha(path)
                    assert identity["historical_input_sha256"] == array_sha(history)
                    assert identity["encoder_state_sha256"] == encoder_hashes[random]
                    with np.load(fp) as feature_file:
                        caches[random] = (
                            torch.from_numpy(feature_file["features"]),
                            torch.from_numpy(feature_file["valid"]),
                        )
                    continue
                feats, masks = [], []
                tic = time.monotonic()
                for i in range(0, n, 4):
                    f, mask = encode(encoder, hs[i : i + 4], valid_lengths[i : i + 4])
                    feats.append(f.cpu())
                    masks.append(mask.cpu())
                torch.cuda.synchronize()
                caches[random] = torch.cat(feats), torch.cat(masks)
                np.savez_compressed(
                    fp,
                    features=caches[random][0].numpy(),
                    valid=caches[random][1].numpy(),
                )
                dump(
                    fp.with_suffix(".json"),
                    dict(
                        sha256=sha(fp),
                        seconds=time.monotonic() - tic,
                        input_archive_sha256=sha(path),
                        historical_input_sha256=array_sha(history),
                        effective_history=256,
                        padded_history=512,
                        encoder_state_sha256=encoder_hashes[random],
                        checkpoint_sha256=None if random else WEIGHT_SHA,
                    ),
                )
            donors = (
                history_donors(n, domain, domain, "normal_SELECT")
                if group == "normal_SELECT"
                else np.arange(n)
            )
            for arm, (model, ck) in models.items():
                fp = folder / f"{arm}_{group}.npz"
                if fp.with_suffix(".json").exists():
                    existing = read(fp.with_suffix(".json"))
                    assert sha(fp) == existing["sha256"]
                    assert ck["sha256"] == existing["checkpoint"]
                    assert sha(path) == existing["input_archive_sha256"]
                    continue
                state, step = multi.fingerprint(model.state_dict()), model.left.step
                outputs = {}
                tic = time.monotonic()
                for name in old.EDITS:
                    source = archived[name]
                    np.testing.assert_array_equal(source[:, :256], clean[:, :256])
                    chunks = []
                    for i in range(0, n, 16):
                        take = slice(i, i + 16)
                        current = torch.from_numpy(
                            np.ascontiguousarray(source[take, 256:])
                        ).cuda()
                        f = valid = None
                        if arm in ("C", "D", "E"):
                            features, masks = caches[arm == "E"]
                            loc = donors[take] if arm == "D" else np.arange(n)[take]
                            f, valid = features[loc].cuda(), masks[loc].cuda()
                        score = model.infer(
                            current, hs[take], valid_lengths[take], f, valid
                        )[0].mean(-1)
                        chunks.append(score.cpu().numpy())
                    outputs[name] = np.concatenate(chunks)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - tic
                assert (
                    state == multi.fingerprint(model.state_dict())
                    and step == model.left.step
                )
                np.savez_compressed(fp, **outputs)
                dump(
                    fp.with_suffix(".json"),
                    dict(
                        domain=domain,
                        arm=arm,
                        group=group,
                        checkpoint=ck["sha256"],
                        sha256=sha(fp),
                        seconds=elapsed,
                        effective_history=256,
                        input_archive_sha256=sha(path),
                        history_donors=donors.tolist(),
                        singleton_record_identity=(group == "coupled_AR"),
                        labels_read=False,
                    ),
                )
    assert all(
        model_sha(model) == encoder_hashes[random] for random, model in encoders.items()
    )


def controlled_metrics(w, domains):
    from scipy.stats import rankdata
    from sklearn.metrics import average_precision_score, roc_auc_score
    from .left_history_control import GROUPS, REGIONS
    from .trajectory_evaluation import save_rows

    rows, per_record = [], []
    for domain in domains:
        for arm in ARMS:
            for group in GROUPS:
                fp = w / OUT / "controlled" / domain / f"{arm}_{group}.npz"
                assert sha(fp) == read(fp.with_suffix(".json"))["sha256"]
                with np.load(fp) as z:
                    for region, indices in REGIONS.items():
                        scores = {k: z[k][:, indices].mean(1) for k in z.files}
                        for target, reference in (
                            ("weak", "clean"),
                            ("weak", "weak_control"),
                            ("weak_control", "clean"),
                            ("lag", "clean"),
                            ("lag", "lag_amplitude"),
                            ("lag", "synchronous"),
                            ("lag_amplitude", "clean"),
                            ("synchronous", "clean"),
                        ):
                            a, b = scores[target], scores[reference]
                            y = np.r_[np.ones(len(a)), np.zeros(len(b))]
                            s = np.r_[a, b]
                            roc = float(roc_auc_score(y, s))
                            check = (
                                rankdata(s)[: len(a)].sum() - len(a) * (len(a) + 1) / 2
                            ) / (len(a) * len(b))
                            assert abs(check - roc) < 1e-12
                            diff = a - b
                            per_record.extend(
                                dict(
                                    domain=domain,
                                    arm=arm,
                                    group=group,
                                    record=k,
                                    region=region,
                                    target=target,
                                    reference=reference,
                                    target_score=float(a[k]),
                                    reference_score=float(b[k]),
                                    delta=float(diff[k]),
                                )
                                for k in range(len(diff))
                            )
                            rows.append(
                                dict(
                                    domain=domain,
                                    arm=arm,
                                    group=group,
                                    region=region,
                                    target=target,
                                    reference=reference,
                                    n=len(a),
                                    ROC_AUC=roc,
                                    AP=float(average_precision_score(y, s)),
                                    mean_delta=float(diff.mean()),
                                    median_delta=float(np.median(diff)),
                                    wins=int((diff > 0).sum()),
                                    ties=int((diff == 0).sum()),
                                    losses=int((diff < 0).sum()),
                                )
                            )
    save_rows(w / OUT / "controlled" / "comparisons.csv", rows)
    save_rows(w / OUT / "controlled" / "per_record_differences.csv", per_record)


def bank(w, domain, entity, arm, role):
    path = w / OUT / "public" / domain / entity / f"{arm}_{role}.npz"
    meta = read(path.with_suffix(".json"))
    assert sha(path) == meta["sha256"]
    with np.load(path) as z:
        i, s, av = z["index"], z["score"], z["available"]
        np.testing.assert_array_equal(
            z["history_start"] + HISTORY, z["history_end_exclusive"]
        )
        assert np.all(z["history_end_exclusive"] <= z["current_start"])
        if arm != "D":
            np.testing.assert_array_equal(
                z["history_end_exclusive"], z["current_start"]
            )
        np.testing.assert_array_equal(z["current_start"] + WINDOW - 1, av)
        assert np.isfinite(s).all() and np.all((i <= av) & (i > av - WINDOW))
    return i, s, av, meta


def ranking(y, score, metric):
    if len(np.unique(y)) < 2:
        return dict.fromkeys(METRICS), None
    measured, _ = continuous_metrics(y, score, metric)
    error = float(
        np.max(
            np.abs(
                independent_check(y, score)
                - np.array([measured["AP"], measured["ROC_AUC"]])
            )
        )
    )
    assert error < 1e-10
    return {k: measured[k] for k in METRICS}, error


def measure(task):
    w, domain, entity, arm, role = task
    y = truth(w, domain, entity) if domain == "SMD" else labels(w, domain)
    if arm == "archived":
        n = (
            len(y)
            if domain != "SWAT"
            else ROLES[domain]["TEST"][1] - ROLES[domain]["TEST"][0]
        )
        data = (
            previous.archived_scores(w, domain, entity, n)
            if role == "TEST"
            else archived_mixed(w, domain, entity)
        )
        _, _, starts, _ = role_input(w, domain, entity, role)
        first, last = int(starts[0]), int(starts[-1] + WINDOW)
    else:
        idx, s, av, meta = bank(w, domain, entity, arm, role)
        data = {arm: (idx, s, meta["sha256"])}
        first, last = int(idx[0]), int(idx[-1] + 1)
    metric = load_author_metric(w / AUTHOR)
    begin = time.monotonic()
    rows, operations, events = [], [], []
    end_ext = min(last, len(y) // 100 * 100) if domain in ("PSM", "SMD") else last
    panels = (
        (("complete", last), ("external", end_ext))
        if role == "TEST"
        else (("mixed", last),)
    )
    for name, (index, values, digest) in data.items():
        cached = {}
        for panel, end in panels:
            if (
                name in ("dada", "timeradar_endpoint", "timercd")
                and panel != "external"
            ):
                continue
            query = np.arange(first, end)
            loc = np.searchsorted(index, query)
            np.testing.assert_array_equal(index[loc], query)
            if end not in cached:
                cached[end] = ranking(y[query], values[loc], metric)
            met, err = cached[end]
            rows.append(
                dict(
                    domain=domain,
                    entity=entity,
                    method=name,
                    panel=panel,
                    role=role,
                    start=first,
                    stop=end,
                    points=len(query),
                    positives=int(y[query].sum()),
                    score_sha256=digest,
                    **met,
                    independent_AP_ROC_error=err,
                )
            )
    if role == "QUERY" and arm != "archived":
        ci, cs, _, cm = bank(w, domain, entity, arm, "CAL")
        wanted = cal_points(w, domain, entity)
        loc = np.searchsorted(ci, wanted)
        good = (loc < len(ci)) & (ci[np.minimum(loc, len(ci) - 1)] == wanted)
        if good.all():
            from .timercd_swat_evaluation import evaluate_score

            threshold = float(np.quantile(cs[loc], 0.99, method="higher"))
            operations, events, _, _ = evaluate_score(
                idx,
                s,
                y,
                threshold,
                av,
                query_start=first,
                query_stop=last,
                original_length=len(y),
                expected_events=None,
                ranking=rows[0],
            )
            for r in operations + events:
                r.update(
                    domain=domain, entity=entity, method=arm, cal_points=len(wanted)
                )
        else:
            operations = [
                dict(
                    domain=domain,
                    entity=entity,
                    method=arm,
                    cal_points=len(wanted),
                    cal_legal_points=int(good.sum()),
                    status="NA: fixed CAL lacks legal full history",
                )
            ]
    dest = w / OUT / "metrics" / domain / entity / f"{arm}_{role}.json"
    dest.parent.mkdir(exist_ok=True, parents=True)
    dump(
        dest,
        dict(
            rows=rows,
            operations=operations,
            events=events,
            seconds=time.monotonic() - begin,
        ),
    )
    return dest


def metrics(w, domains, arms, role, workers, entities=None):
    tasks = [
        (w, d, e, a, role)
        for d in domains
        for e in (entities or previous.members(w, d))
        for a in arms
    ]
    tasks = [
        t
        for t in tasks
        if not (w / OUT / "metrics" / t[1] / t[2] / f"{t[3]}_{role}.json").exists()
    ]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(measure, tasks))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument(
        "action",
        choices=(
            "prepare",
            "metrics",
            "features",
            "score",
            "controlled",
            "controlled_metrics",
        ),
    )
    p.add_argument("domains", nargs="+", choices=("PSM", "SWAT", "SMD"))
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--role", choices=("TEST", "QUERY", "CAL"), default="TEST")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--entities", nargs="+")
    a = p.parse_args()
    if a.action == "prepare":
        freeze_axes(a.workspace, a.domains)
    elif a.action in ("score", "features"):
        for domain in a.domains:
            (score_models if a.action == "score" else features_only)(
                a.workspace,
                domain,
                a.entities or previous.members(a.workspace, domain),
                a.role,
                a.arms,
            )
    elif a.action == "controlled":
        for domain in a.domains:
            assert domain in ("PSM", "SWAT")
            controlled_score(a.workspace, domain, a.arms)
    elif a.action == "controlled_metrics":
        controlled_metrics(a.workspace, a.domains)
    else:
        metrics(a.workspace, a.domains, a.arms, a.role, a.workers, a.entities)
