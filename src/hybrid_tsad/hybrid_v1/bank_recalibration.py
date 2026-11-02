"""Label-free immutable-producer -> scalar overlay, using accepted CAL only.

No inference or fitting. Original residual/replay bank bytes remain in place;
the small new bank points to original chunk/position rather than copying replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .chunked_bank import ChunkedBank, read_records
from .scoring import apply_calibration, recipe

OPERATION="HYBRID_V1_RAW_BANK_RECALIBRATION"
SCORING_SHA="a87517e1f4a02b8193ad1ffbc667a88710ec22c5db884a8f50d85b818961576d"
READER_SHA="17a298cb5f7255ae9d429630ec4f07152a65c7711a2657bba1e6de6d9ec4d2b4"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def bound(spec):
    if not spec.get("sha256") or sha(spec["path"])!=spec["sha256"]:raise PermissionError("Missing/changed bound asset")
    return json.loads(Path(spec["path"]).read_text())


def code_gate(cfg,signed,required):
    from . import named_evaluate_cli as accepted
    base=Path(__file__).resolve().parents[2];own=cfg["source_sha256"]
    pinned={"hybrid_tsad/hybrid_v1/scoring.py":SCORING_SHA,"hybrid_tsad/hybrid_v1/chunked_bank.py":READER_SHA,
        "hybrid_tsad/hybrid_v1/named_evaluate_cli.py":"d0fe221e2ce6ba9f6f18f3a143ee6aaca05179b92d018cc2e0d4964afef85957",
        "hybrid_tsad/hybrid_v1/evaluation.py":accepted.EVALUATOR_SHA,"hybrid_tsad/hybrid_v1/statistics.py":accepted.STATISTICS_SHA,
        "hybrid_tsad/hybrid_v1/evaluate_cli.py":"4fe88b0e9b7792ee95845f087113e4e04813fc732e15257fa1a482bbbcd697cd"}
    if signed.get("source_sha256")!=own or not (set(required)|set(pinned)|accepted.REQUIRED_SOURCE)<=set(own):raise PermissionError("Incomplete code closure")
    for name,digest in own.items():
        path=(base/name).resolve()
        if base not in path.parents or path.suffix!=".py" or sha(path)!=digest or (name in pinned and digest!=pinned[name]):raise PermissionError("Code identity changed")


def bank_metadata(e):
    directory=Path(e["directory"]);m=bound(dict(path=str(directory/"manifest.json"),sha256=e["manifest_sha256"]))
    c=m["contract"];ranges=e["ranges"];n=sum(r["end"]-r["start"] for r in ranges)
    if (m.get("complete") is not True or m.get("committed_records")!=n or n<=0 or c["identity"]!=e["identity"]
            or c["ranges"]!=ranges or c["chunk_size"]!=64 or c["confirmation_points"]!=3):raise ValueError("Incomplete/changed bank contract")
    previous={};position=0
    for r in ranges:
        stream=tuple(r[k] for k in ("domain","entity","run"))
        if type(r["start"]) is not int or type(r["end"]) is not int or not 0<=r["start"]<r["end"] or r["start"]<previous.get(stream,0):raise ValueError("Bad original ranges")
        previous[stream]=r["end"]
    for chunk in m["chunks"]:
        path=directory/chunk["file"]
        if (path.parent!=directory or path.name!=chunk["file"] or chunk["start_position"]!=position or not 0<chunk["records"]<=64
                or (chunk["records"]<64 and position+chunk["records"]!=n) or sha(path)!=chunk["sha256"]):raise ValueError("Chunk identity/count changed")
        position+=chunk["records"]
    if position!=n:raise ValueError("Incomplete chunk coverage")
    return m


def calibration(e):
    c=bound(e["calibration"]);receipt=bound(e["calibration_receipt"])
    if e["calibration"]["sha256"]!=e["identity"]["calibration_sha256"]:raise ValueError("CAL identity differs")
    source=e["calibration_origin"]=="source"
    if e["calibration_origin"] not in {"source","target"} or c.get("source_domain_equal_weight") is not source:raise ValueError("Source/target CAL cannot masquerade")
    active,weights=recipe(c.get("active_components"))
    if list(active)!=e["active_components"] or c.get("weights")!=weights or set(c["scales"])!=set(active) or any(not np.isfinite(v) or v<=0 for v in c["scales"].values()) or not np.isfinite(c["threshold"]):raise ValueError("Fixed CAL recipe/scales invalid")
    phase="calibrate_source" if source else "target_calibrate"
    if receipt.get("phase")!=phase or any(receipt.get(k)!=v for k,v in e["receipt_expect"].items()):raise ValueError("CAL receipt phase/lineage changed")
    if not source:
        if (receipt.get("mode")!=e["adaptation_mode"] or receipt.get("mode") not in {"Z_TCAL","U"} or receipt.get("updates")!=0
                or receipt.get("status")!="COMPLETED" or receipt.get("source_checkpoint_sha256")!=e.get("parent_source_checkpoint_sha256",e["identity"]["checkpoint_sha256"])
                or receipt.get("query_or_labels") is not False or receipt.get("loaded_roles")!=["CAL"]
                or receipt.get("target_calibration_contact") is not True or receipt.get("calibrated_checkpoint_sha256")!=e["identity"]["checkpoint_sha256"]):raise ValueError("Target CAL checkpoint/role/mode invalid")
    return c


def expected_keys(ranges):
    for r in ranges:
        for i in range(r["start"],r["end"]):yield tuple(r[k] for k in ("domain","entity","run"))+(i,)


def recipe_identity(parent,destination):
    # No circular hash of a config containing its own digest. Root also signs
    # the exact complete operation config independently at the CLI boundary.
    value=dict(producer_manifest_sha256=parent["manifest_sha256"],input_sha256=parent["identity"]["input_sha256"],checkpoint_sha256=parent["identity"]["checkpoint_sha256"],
        calibration_sha256=destination["calibration"]["sha256"],active_components=destination["active_components"],calibration_origin=destination["calibration_origin"],mode=destination["adaptation_mode"])
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def checked_rows(e):
    expected=iter(expected_keys(e["ranges"]))
    for row in read_records(e["directory"]):
        key=next(expected,None)
        if key is None or tuple(row.get(k) for k in ("domain","entity","run","index"))!=key or type(row.get("index")) is not int or type(row.get("available_index")) is not int or row["available_index"]!=row["index"]:raise ValueError("Original axis/arrived clock differs")
        yield row
    if next(expected,None) is not None:raise ValueError("Missing raw original points")


def gate(cfg):
    if any(k in cfg for k in ("labels","weights","fit","ensemble")):raise PermissionError("No labels/fitting/custom weights")
    seal=bound(cfg["input_seal"])
    if seal!={"status":"ACCEPTED_RECALIBRATION_INPUTS","approved_by":"root","producer":cfg["producer"],"destination":cfg["destination"]}:raise PermissionError("Exact producer/CAL root seal required")
    m=bank_metadata(cfg["producer"]);e=cfg["destination"];c=calibration(e)
    if (e["identity"]["input_sha256"]!=cfg["producer"]["identity"]["input_sha256"] or e["identity"]["checkpoint_sha256"]!=cfg["producer"]["identity"]["checkpoint_sha256"] or e["ranges"]!=m["contract"]["ranges"]):raise ValueError("Recalibration cannot change model/input/axis")
    if e["identity"]["config_sha256"]!=recipe_identity(cfg["producer"],e) or e["identity"].get("input_runs")!=cfg["producer"]["identity"].get("input_runs"):raise ValueError("Overlay recipe/raw-input identity differs")
    if Path(e["directory"]).exists():raise FileExistsError("Fresh overlay namespace required")
    return m,c


def run(cfg):
    m,c=gate(cfg);e=cfg["destination"];parent=cfg["producer"];position=0
    # The existing writer computes strict/confirm3 with gap reset. No custom alarm code.
    with ChunkedBank(e["directory"],identity=e["identity"],ranges=e["ranges"],threshold=c["threshold"],chunk_size=64) as bank:
        for raw in checked_rows(parent):
            values={k:raw.get(k) for k in e["active_components"]};derived=apply_calibration(values,c)
            row={k:raw[k] for k in ("domain","entity","run","index","available_index")}
            row.update(values);row.update(derived);row["reason"]=derived.get("reason")
            if "input_identity" in raw.get("replay",{}):row["replay"]={"input_identity":raw["replay"]["input_identity"]}
            row["producer_position"]=position;row["producer_manifest_sha256"]=parent["manifest_sha256"]
            bank.append(row);position+=1
        bank.finalize()
    # Original producer chunks and manifest must still match. No source bank opened for writing.
    bank_metadata(parent)
    receipt=dict(status="COMPLETED_PENDING_ACCEPTANCE",operation=OPERATION,producer=parent,destination=e,
        manifest_sha256=sha(Path(e["directory"])/"manifest.json"),records=position,raw_payload="Original immutable producer, not duplicated; scalar overlay only",
        model_or_fit_executed=False,labels_accessed=False,threshold_fitted=False,code=cfg.get("source_sha256"),input_seal=cfg["input_seal"])
    Path(e["directory"],"recalibration_receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    Path(e["directory"],"bank_recalibration.py").write_bytes(Path(__file__).read_bytes())
    return receipt


def iter_materialized(overlay):
    """Lazy full raw/prediction view, preserving original score/valid/reason.

    The overlay is small on disk. This view keeps the complete original replay
    and predictions via its authenticated producer rather than duplicating them.
    """
    receipt=bound(overlay["receipt"]);parent=overlay["producer"];destination=overlay["destination"]
    if receipt.get("operation")!=OPERATION or receipt.get("producer")!=parent or receipt.get("destination")!=destination:raise ValueError("Overlay receipt lineage differs")
    bank_metadata(parent)
    e=dict(destination,manifest_sha256=receipt["manifest_sha256"]);bank_metadata(e);c=calibration(destination)
    rows=iter(checked_rows(e))
    for position,raw in enumerate(checked_rows(parent)):
        row=next(rows,None);expected=apply_calibration(raw,c)
        if (row is None or row.get("producer_position")!=position or row.get("producer_manifest_sha256")!=parent["manifest_sha256"]
                or row["index"]!=raw["index"] or row.get("valid")!=expected["valid"]
                or (expected["valid"] and row.get("score")!=expected["score"])
                or (not expected["valid"] and row.get("reason")!=expected["reason"])):raise ValueError("Noncanonical overlay")
        from .chunked_bank import _fingerprint
        if _fingerprint({k:row.get(k) for k in destination["active_components"]})!=_fingerprint({k:raw.get(k) for k in destination["active_components"]}):raise ValueError("Overlay residual changed")
        materialized=dict(raw)
        materialized["producer_derived"]={k:raw.get(k) for k in ("score","valid","reason","alarm")}
        materialized.update(expected);materialized["reason"]=expected.get("reason")
        # An invalid canonical score has no new alarm; never inherit old alarm.
        if not expected["valid"]:materialized.pop("alarm",None)
        materialized["producer_position"]=position;materialized["producer_manifest_sha256"]=parent["manifest_sha256"]
        yield materialized
    if next(rows,None) is not None:raise ValueError("Overlay has extra points")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("config","release","trusted-release-sha256"):p.add_argument("--"+k)
    p.add_argument("--execute",action="store_true");p.add_argument("--preflight-only",action="store_true");a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,bank_or_labels_opened=False)));return
    cfg=json.loads(Path(a.config).read_text());signed=bound(dict(path=a.release,sha256=a.trusted_release_sha256))
    if (signed.get("status")!="APPROVED" or signed.get("approved_by")!="root" or signed.get("operation")!=OPERATION or cfg.get("operation")!=OPERATION
            or signed.get("allowed_actions")!=["recalibrate_raw_bank"] or signed.get("allow_labels") is not False or signed.get("config_sha256")!=sha(a.config)
            or signed.get("input_seal_sha256")!=cfg["input_seal"]["sha256"]):raise PermissionError("Exact root label-free CPU release required")
    code_gate(cfg,signed,{"hybrid_tsad/hybrid_v1/bank_recalibration.py"})
    if a.preflight_only:gate(cfg);print(json.dumps(dict(metadata_PASS=True,labels_opened=False)));return
    run(cfg)


if __name__=="__main__":main()
