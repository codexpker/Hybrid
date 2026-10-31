"""Separate Stage2 operation identity. Never relax the source/Z policy."""
import hashlib
import json
from pathlib import Path
import re

from ..config import OFFICIAL_SHA, read_json, resolve_path, sha256
from .identity import operation_manifest, SCORE_SPACE

STAGE = "U_NORMAL_HEAD"
CONTRACT_SHA = "f345106750b89e6f7da3d0788ed51cb5aa636a1043c70f50781a2456e82e80af"
RECIPE_CANONICAL_SHA = "e997ac28c6a38c48b1f78495ef6aadb82a59d6e415ef50f8b32ce84e7f91cb5d"
NORMAL_RAW_SHA = "12d1055b852763fac09ef36bd7c9612a800c8bf7b6f75f344d12119206f9e940"
NORMAL_CACHE_SHA = "43cff9a26cbcf15ffbf66fd40db1b9b10ee114ebd9e8fbbcb10b741b99b15fdd"
LINEAGE_SHA = "39dbfa5849ee0ad1a555d58e76e6046f056b1e44d033aa89a5d720b921d0edb8"
PARENT_MANIFEST_SHA = "9799d406d7297cebda845e9b78a5583f387194c02271536f5387540931b8fc26"
PREPARATION_SHA = "f73fb60b113b18cdc43a9e816d37a2477af65220d3ad88dcfe1d9a80b1c4f291"
SOURCE_REFERENCES_SHA = "095bd53b7b9a5faf04428270c635c28264094359afc6f2b5743497f287640236"
L0_REQUIREMENTS_SHA = "efd079dad5b5e6743333ac54a3b8abc28b2e959f4654aa2408b9dff728d55be0"
PAIR = {f"u_normal_head_hybrid_s0_{i}" for i in ("source_pretrained", "random_init")}
BASE_INPUTS = {"recipe", "preparation", "target_manifest", "target_plan", "source_references", "source_cal_trace", "runtime"}
HEAD_NAMES = ("head.weight", "head.bias")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_stage2(path):
    cfg = read_json(path)
    if (set(cfg) != {"engine_schema_version", "operation", "stage", "contract", "approval", "inputs", "parameters", "output", "score_space"}
            or cfg["engine_schema_version"] != 5 or cfg["stage"] != STAGE
            or cfg["operation"] not in ("stage2-train", "stage2-calibrate", "stage2-l0") or cfg["score_space"] != SCORE_SPACE):
        raise ValueError("Stage2 requires its own schema5 stage/operation/score-space")
    expected = BASE_INPUTS | ({"entry_gate"} if cfg["operation"] == "stage2-train" else {"requirements"} if cfg["operation"] == "stage2-l0" else {"checkpoint", "checkpoint_metadata", "training_summary"})
    if set(cfg["inputs"]) != expected or set(cfg["parameters"]) != {"job_id", "device"}:
        raise ValueError("Stage2 inputs/parameters are fixed; no labels/query/step override")
    if (not re.fullmatch(r"u_normal_head_(temporal|tf|hybrid)_s[012]_(source_pretrained|random_init)", cfg["parameters"]["job_id"])
            or cfg["parameters"]["device"] not in ("cpu", "cuda:0")):
        raise ValueError("Unapproved Stage2 job/device")
    records = {"contract": cfg["contract"], "approval": cfg["approval"], **cfg["inputs"]}
    for ref in records.values():
        if set(ref) != {"path", "sha256"} or not re.fullmatch("[0-9a-f]{64}", ref["sha256"]):
            raise ValueError("Operation assets require exact path/SHA records")
    paths = {name: resolve_path(ref["path"], path) for name, ref in records.items()}
    output = resolve_path(cfg["output"], path)
    manifest, _ = operation_manifest(cfg, paths, output)
    manifest.update(stage=STAGE, permissions="known-normal head only; no query/labels/F")
    return cfg, paths, output, manifest, canonical_sha(manifest)


def runtime_identity():
    root=Path(__file__).resolve().parents[1]
    return dict(stage=STAGE,files={"src/hybrid_tsad/"+str(p.relative_to(root)).replace('\\','/'):sha256(p) for p in sorted(root.rglob('*.py'))})


def input_identity(cfg):
    return {name:cfg['inputs'][name]['sha256'] for name in ('recipe','target_manifest','target_plan','source_references','source_cal_trace','runtime')}


class Assets:
    def __init__(self):
        self.hashes = {}

    def file(self, ref, base):
        if (not isinstance(ref, dict) or not {"path", "sha256"} <= set(ref)
                or set(ref) - {"path", "sha256", "bytes"} or not re.fullmatch("[0-9a-f]{64}", ref["sha256"])):
            raise ValueError("Malformed external Stage2 asset record")
        p = resolve_path(ref["path"], base)
        if p in self.hashes and self.hashes[p] != ref["sha256"]:
            raise ValueError("Conflicting asset identities")
        if p not in self.hashes and (not p.is_file() or sha256(p) != ref["sha256"]):
            raise ValueError("Stage2 external input SHA mismatch: " + p.name)
        if "bytes" in ref and p.stat().st_size != ref["bytes"]:
            raise ValueError("Stage2 asset byte count mismatch")
        self.hashes[p] = ref["sha256"]
        return p

    def same(self, a, b, base_a, base_b):
        if self.file(a, base_a) != self.file(b, base_b) or a["sha256"] != b["sha256"]:
            raise ValueError("Stage2 asset does not match its prepared identity")

    def recheck(self):
        if any(sha256(p) != h for p, h in self.hashes.items()):
            raise ValueError("Frozen Stage2 inputs changed during execution")


def validate_assets(cfg, paths):
    """Only called after explicit approval; returns no query/label accessor."""
    from .stage2_data import validate_target, validate_source_trace
    assets = Assets()
    for name, ref in {"contract": cfg["contract"], "approval": cfg["approval"], **cfg["inputs"]}.items():
        assets.file({"path": str(paths[name]), "sha256": ref["sha256"]}, paths[name])
    if cfg["contract"]["sha256"] != CONTRACT_SHA:
        raise ValueError("Stage2 contract is not the accepted U_NORMAL_HEAD contract")
    if read_json(paths['runtime'])!=runtime_identity():raise ValueError('Stage2 runtime bytes differ from the approved snapshot')
    recipe = read_json(paths["recipe"])
    if canonical_sha(recipe) != RECIPE_CANONICAL_SHA:
        raise ValueError("Stage2 recipe changed; no budget/loss/score override")
    prep = read_json(paths["preparation"])
    if sha256(paths["preparation"])!=PREPARATION_SHA or sha256(paths["source_references"])!=SOURCE_REFERENCES_SHA:
        raise ValueError("Stage2 original matrix/source-parent reference identity changed")
    if prep.get("schema_version") != "RESEARCH_PREPARATION_V1_NOT_ENGINE_OPERATION" or prep.get("stage") != STAGE:
        raise ValueError("Stage2 preparation is not the fixed metadata matrix")
    jobs = prep["jobs"]
    ids = {f"u_normal_head_{m}_s{s}_{i}" for m in ("temporal", "tf", "hybrid") for s in range(3) for i in ("source_pretrained", "random_init")}
    if len(jobs) != 18 or {j["job_id"] for j in jobs} != ids:
        raise ValueError("Static Stage2 matrix must retain all18 jobs; it is not permission")
    job = next(j for j in jobs if j["job_id"] == cfg["parameters"]["job_id"])
    mode, seed, init = job["mode"], job["seed"], job["initialization"]
    if job["job_id"] != f"u_normal_head_{mode}_s{seed}_{init}" or job["stage"] != STAGE or job["head_reset_after_load"] is not False:
        raise ValueError("Job identity/head reset contradicts preparation")
    if (init == "random_init" and (job["initial_checkpoint"] is not None or job["random_init_seed"] != seed)
            or init == "source_pretrained" and job["random_init_seed"] is not None):
        raise ValueError("Random whole initialization cannot load a parent/head")
    assets.same({"path":str(paths["target_manifest"]),"sha256":cfg["inputs"]["target_manifest"]["sha256"]}, prep["target_manifest"], paths["target_manifest"], paths["preparation"])
    for name in ("target_plan", "source_cal_trace"):
        resolved={"path":str(paths[name]),"sha256":cfg["inputs"][name]["sha256"]}
        assets.same(resolved, prep["plans"][str(seed)][name], paths[name], paths["preparation"])
        assets.same(resolved, job[name], paths[name], paths["preparation"])
    refs = read_json(paths["source_references"])
    if prep["plans"][str(seed)]["original_source_assets"] != refs["source_assets"][str(seed)]:
        raise ValueError("Preparation source assets differ from original source references")
    original = read_json(assets.file(refs["parent_catalog_definition"], paths["source_references"]))
    if any(refs[k] != original[k] for k in ("models", "source_assets", "official")):
        raise ValueError("Stage2 references differ from the original source definition")
    parents = refs["models"]
    if len(parents) != 9 or {(r["seed"], r["mode"]) for r in parents} != {(s, m) for s in range(3) for m in ("temporal", "tf", "hybrid")}:
        raise ValueError("Original nine parent identities are incomplete")
    parent = next(r for r in parents if r["seed"] == seed and r["mode"] == mode)
    for name in ("checkpoint", "checkpoint_metadata", "calibration", "cal_bank", "cal_scores"):
        assets.file(parent[name], paths["source_references"])
    assets.same(job["paired_reference_parent"], parent["checkpoint"], paths["preparation"], paths["source_references"])
    if init == "source_pretrained":assets.same(job["initial_checkpoint"], parent["checkpoint"], paths["preparation"], paths["source_references"])
    metadata = read_json(assets.file(parent["checkpoint_metadata"], paths["source_references"]))
    expected = dict(checkpoint_sha256=parent["checkpoint"]["sha256"], mode=mode, seed=seed, history=100, horizon=5,
                    copies=2, temporal_depth=4, step=10000, excluded_target="TEP")
    if (any(metadata.get(k) != v for k, v in expected.items()) or set(metadata.get("source_domains", [])) != {"PSM", "SMD_SUBSET"}
            or metadata.get("stage") not in (None, "SOURCE_PRETRAIN") or "adaptation_target" in metadata):
        raise ValueError("Wrong parent: Stage2/target-adapted checkpoint cannot be a source/Z parent")
    official = assets.file(refs["official"], paths["source_references"])
    if sha256(official) != OFFICIAL_SHA:raise ValueError("External official implementation changed")
    target, plan = validate_target(paths["target_manifest"], paths["target_plan"], seed, assets)
    source = validate_source_trace(refs, paths["source_references"], paths["source_cal_trace"], seed, metadata, assets)
    if cfg['operation']=='stage2-l0':
        req=read_json(paths['requirements'])
        if (sha256(paths['requirements'])!=L0_REQUIREMENTS_SHA or req.get('stage')!=STAGE or job['job_id'] not in PAIR
                or {j['job_id'] for j in req['allowed_jobs_if_separately_released']}!=PAIR):
            raise ValueError('L0_DISCARD only accepts the original requirements and fixed hybrid seed0 pair')
        for n,key in [('target_manifest','target_normal_manifest'),('target_plan','target_plan'),('source_cal_trace','source_cal_trace')]:
            assets.same({'path':str(paths[n]),'sha256':cfg['inputs'][n]['sha256']},req['shared_inputs'][key],paths[n],paths['requirements'])
        assets.same({'path':str(paths['contract']),'sha256':cfg['contract']['sha256']},req['contract'],paths['contract'],paths['requirements'])
    return dict(assets=assets, recipe=recipe, preparation=prep, job=job, parent=parent, parent_metadata=metadata,
                refs=refs, official=official, target=target, plan=plan, source=source, paths=paths)


def approve_stage2(cfg, paths, output, digest):
    if output.exists():raise ValueError("Stage2 output must be a new immutable directory")
    if sha256(paths["approval"]) != cfg["approval"]["sha256"] or sha256(paths["contract"]) != cfg["contract"]["sha256"]:
        raise ValueError("Stage2 approval/contract SHA mismatch")
    a = read_json(paths["approval"])
    permissions = dict(train=cfg["operation"] == "stage2-train", calibrate=cfg["operation"] == "stage2-calibrate",l0=cfg['operation']=='stage2-l0', query=False, labels=False, F=False)
    if (a.get("status") != "APPROVED" or a.get("stage") != STAGE or a.get("contract_sha256") != cfg["contract"]["sha256"]
            or a.get("operation_manifest_sha256") != digest or cfg["parameters"]["job_id"] not in a.get("allowed_jobs", [])
            or a.get("permissions") != permissions or not all(type(v) is bool for v in a.get("permissions", {}).values())):
        raise ValueError("Stage2 approval does not bind this exact operation/job/permissions")
    if a.get('scope')=='SYNTHETIC_ENGINEERING_ONLY' and cfg['parameters']['device']!='cpu':
        raise ValueError('Synthetic fixture approvals are CPU-only')
    if cfg['operation']=='stage2-l0' and (cfg['parameters']['job_id'] not in PAIR
            or a.get('scope')!='U_NORMAL_HEAD_L0_DISCARD_ONLY' and not (a.get('scope')=='SYNTHETIC_ENGINEERING_ONLY' and cfg['parameters']['device']=='cpu')):
        raise ValueError('L0 requires explicit U_NORMAL_HEAD_L0_DISCARD_ONLY scope')
    if cfg["operation"] == "stage2-train":
        gate_path = paths["entry_gate"]
        if sha256(gate_path) != cfg["inputs"]["entry_gate"]["sha256"]:raise ValueError("Stage2 entry gate SHA mismatch")
        gate = read_json(gate_path)
        synthetic = a.get("scope") == "SYNTHETIC_ENGINEERING_ONLY" and cfg["parameters"]["device"] == "cpu"
        if (gate.get("stage") != STAGE or cfg["parameters"]["job_id"] not in gate.get("allowed_jobs", [])
                or gate.get("status") != ("CPU_ENGINEERING_ONLY" if synthetic else "PRIMARY_ACCEPTED_STAGE2_L0")
                or gate.get('accepted_inputs_by_job',{}).get(cfg['parameters']['job_id'])!=input_identity(cfg)):
            raise ValueError("Formal Stage2 requires a separately accepted discarded-weight L0 gate")
        if not synthetic:
            g=Assets();g.file(gate['independent_review'],gate_path)
            if not set(gate['allowed_jobs'])<=PAIR:
                accepted=read_json(g.file(gate['formal_pair_acceptance'],gate_path))
                if accepted.get('status')!='PRIMARY_ACCEPTED_FORMAL_PAIR' or accepted.get('stage')!=STAGE:
                    raise ValueError('Additional jobs need a new named acceptance after the formal pair')
            if set(gate['l0_reports'])!=PAIR:raise ValueError('Primary entry acceptance requires both original L0 reports')
            for job,ref in gate['l0_reports'].items():
                report=read_json(g.file(ref,gate_path))
                if (report.get('status')!='U_NORMAL_HEAD_L0_DISCARD_COMPLETE' or report.get('job_id')!=job
                        or report.get('GPU_executed') is not True or report.get('weights_discarded') is not True
                        or report.get('steps')!=5 or report.get('query_or_labels_read') is not False
                        or report.get('inputs')!=gate['l0_input_identity'] or report.get('frozen_before')!=report.get('frozen_after')
                        or not report.get('diagnostics',{}).get('rng_restored') or not report.get('module_flags_preserved')):
                    raise ValueError('L0 report does not establish the accepted discarded-weight diagnostic')
            if gate['l0_input_identity']['runtime']!=cfg['inputs']['runtime']['sha256'] or gate['l0_input_identity']['recipe']!=cfg['inputs']['recipe']['sha256']:
                raise ValueError('Accepted L0 used a different runtime/recipe')
            a['_entry_assets']=g.hashes
    return a
