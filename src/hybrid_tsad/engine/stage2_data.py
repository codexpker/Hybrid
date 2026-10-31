"""Execution-side normal-target/source trace access. No label/query loader."""
import copy
import numpy as np

from ..config import read_json

ROLES = {'fit': list(range(1, 21)) + list(range(51, 61)), 'select': list(range(61, 81)), 'cal': list(range(81, 101))}


def traces(seed):
    result = {}
    for role, offset, size in [('fit', 191100, 64000), ('select', 191200, 256)]:
        rng = np.random.default_rng(offset + seed)
        runs = np.asarray(ROLES[role], dtype=np.uint16)
        # Uniform over legal run/origin/channel triples, with replacement.
        code = rng.integers(len(runs) * 80 * 52, size=size, dtype=np.int64)
        result[role + '_run'] = runs[code // (80 * 52)]
        result[role + '_origin'] = (100 + 5 * ((code // 52) % 80)).astype(np.uint16)
        result[role + '_channel'] = (code % 52).astype(np.uint8)
    # Calibration is B1,C52: 256 distinct multichannel origins, balanced by run.
    # Sixteen runs contribute 13 origins and four runs contribute 12, selected
    # by a seeded run permutation; no labels, model predictions, or loss used.
    rng = np.random.default_rng(191300 + seed)
    runs = np.array(ROLES['cal'], dtype=np.uint16)
    order = rng.permutation(20)
    rr, oo = [], []
    for rank, j in enumerate(order):
        n = 13 if rank < 16 else 12
        origins = 100 + 5 * rng.choice(80, size=n, replace=False)
        rr.extend([int(runs[j])] * n)
        oo.extend(map(int, origins))
    result['cal_run'] = np.array(rr, dtype=np.uint16)
    result['cal_origin'] = np.array(oo, dtype=np.uint16)
    return result


def validate_target(manifest_path, plan_path, seed, assets):
    from . import stage2_identity as policy
    m = read_json(manifest_path)
    if (m.get('schema_version') != 'U_NORMAL_HEAD_PREPARATION_V1' or m.get('stage') != policy.STAGE
            or m.get('data_prior') != 'KNOWN_NORMAL_TRAIN' or m.get('roles') != ROLES
            or m.get('excluded_training_run_numbers') != list(range(21, 51))
            or m.get('faulty_testing_access_allowed') is not False or m.get('original_manifests_modified') is not False):
        raise ValueError('Target manifest stage/roles/permissions changed')
    parent_path = assets.file(m['parent_manifest'], manifest_path)
    lineage_path = assets.file(m['normal_lineage'], manifest_path)
    cache_path = assets.file(m['cache'], manifest_path)
    if (m['parent_manifest']['sha256'] != policy.PARENT_MANIFEST_SHA or m['normal_lineage']['sha256'] != policy.LINEAGE_SHA
            or m['cache']['sha256'] != policy.NORMAL_CACHE_SHA):
        raise ValueError('Target normal parent/lineage/cache differs from fixed public assets')
    lineage = read_json(lineage_path)
    if lineage.get('status') != 'PASS' or lineage['raw']['sha256'] != policy.NORMAL_RAW_SHA:
        raise ValueError('Normal raw/cache lineage has not passed')
    assets.file(lineage['raw'], lineage_path)
    assets.same(lineage['cache'], m['cache'], lineage_path, manifest_path)
    parent = read_json(parent_path)
    original = {r['selector']['simulationRun']: r for r in parent['records'] if r['dataset'] == 'TEP'}
    expected = []
    for role, runs in ROLES.items():
        for run in runs:
            r = copy.deepcopy(original[run])
            if (r['selector'] != {'faultNumber': 0, 'simulationRun': run} or r['cache_row'] != run-1
                    or r['history'] != 100 or r['horizon'] != 5 or r['local_start'] != 0 or r['local_end_exclusive'] != 500
                    or r['forecast_origins'] != list(range(100, 496, 5)) or len(r['columns']) != 52):
                raise ValueError('Target normal parent run/footprint changed')
            r.update(parent_role=r['role'], role=role, stage=policy.STAGE, array_path=str(cache_path), source_identity=policy.NORMAL_RAW_SHA)
            if assets.file({'path': r['source'], 'sha256': r['source_identity']}, manifest_path) != assets.file(lineage['raw'], lineage_path):
                raise ValueError('Target source is not the fixed fault-free training file')
            expected.append(r)
    if m['records'] != expected:
        raise ValueError('Target role/run/footprint records differ from the frozen parent')
    with np.load(plan_path, allow_pickle=False) as f:plan = {k: f[k].copy() for k in f.files}
    rebuilt = traces(seed)
    if set(plan) != set(rebuilt) or any(plan[k].dtype != rebuilt[k].dtype or not np.array_equal(plan[k], rebuilt[k]) for k in plan):
        raise ValueError('Target trace differs from fixed PCG64 seed/role/run/origin/channel plan')
    return m, plan


def validate_source_trace(refs, base, trace_path, seed, parent_metadata, assets):
    src = refs['source_assets'][str(seed)]
    paths = {k: assets.file(v, base) for k, v in src.items()}
    rows = read_json(paths['rows']);manifest = read_json(paths['source_manifest']);inputs = read_json(paths['source_inputs'])
    if (rows != manifest['records'] or any(r['dataset'] not in ('PSM', 'SMD_SUBSET') or r['history'] != 100 for r in rows)
            or parent_metadata['source_manifest_sha256'] != src['source_manifest']['sha256']
            or parent_metadata['shared_plan_sha256'] != src['plan']['sha256']):
        raise ValueError('Original source-cal manifest/parent/trace identity changed')
    from ..config import resolve_path
    if resolve_path(inputs['manifest'], paths['source_inputs']) != paths['source_manifest']:
        raise ValueError('Source input manifest changed')
    for p, h in inputs['hashes'].items():assets.file({'path': p, 'sha256': h}, paths['source_inputs'])
    needed = {str(paths['source_manifest'])} | {r['source'] for r in rows}
    if not needed <= set(inputs['hashes']):raise ValueError('Actual source-cal raw paths lack fixed hashes')
    with np.load(paths['plan'], allow_pickle=False) as f:original = {k: f[k].copy() for k in ('cal_record', 'cal_origin', 'cal_channel')}
    with np.load(trace_path, allow_pickle=False) as f:trace = {k: f[k].copy() for k in f.files}
    if (set(trace) != set(original) or any(trace[k].dtype != original[k].dtype or not np.array_equal(trace[k], original[k]) for k in original)
            or trace['cal_record'].shape != (256,) or trace['cal_record'].dtype != np.uint16
            or trace['cal_origin'].dtype != np.uint32 or trace['cal_channel'].dtype != np.uint8):
        raise ValueError('Source-cal trace is not the original256 C1 trace')
    for ri, b, c in zip(trace['cal_record'], trace['cal_origin'], trace['cal_channel']):
        if ri >= len(rows):raise ValueError('Source-cal record outside manifest')
        r = rows[int(ri)]
        if (r['role'] != 'cal' or int(b) not in r['forecast_origins'] or not 0 <= c < len(r['columns'])
                or int(b)-100 < r['local_start'] or int(b)+5 > r['local_end_exclusive']):
            raise ValueError('Source-cal role/channel/footprint changed')
    return dict(paths=paths, rows=rows, trace=trace)


class TargetNormal:
    def __init__(self, manifest, plan, cache_path):
        self.manifest, self.plan = manifest, plan
        self.data = np.load(cache_path, mmap_mode='r', allow_pickle=False)
        if self.data.shape != (100, 500, 52) or self.data.dtype != np.float32:
            raise ValueError('Normal target cache shape/dtype mismatch')
        if not np.isfinite(self.data[np.asarray(sum(ROLES.values(), []))-1]).all():
            raise ValueError('Authorized normal target values are nonfinite; no repair/rejection relaxation')

    def batch(self, role, start, *, device, n=64):
        import torch
        if role not in ('fit', 'select'):raise ValueError('C1 target batch only accepts fit/select')
        runs, origins, channels = (self.plan[role+'_'+k][start:start+n] for k in ('run', 'origin', 'channel'))
        if len(runs) != n:raise ValueError('Incomplete C1 target trace batch')
        if (not np.isin(runs,ROLES[role]).all() or not np.isin(origins,np.arange(100,496,5)).all()
                or np.any(channels<0) or np.any(channels>=52)):
            raise ValueError('Target batch role/run/channel/footprint violation')
        x = np.stack([self.data[int(r)-1, int(b)-100:int(b), int(c)] for r, b, c in zip(runs, origins, channels)])[:, :, None]
        y = np.stack([self.data[int(r)-1, int(b):int(b)+5, int(c)] for r, b, c in zip(runs, origins, channels)])[:, :, None]
        return torch.tensor(x, device=device), torch.tensor(y, device=device)

    def cal_block(self, index, *, device):
        import torch
        r, b = int(self.plan['cal_run'][index]), int(self.plan['cal_origin'][index])
        if r not in ROLES['cal'] or b not in range(100,496,5):raise ValueError('Target calibration role/footprint violation')
        return (torch.tensor(self.data[r-1:r, b-100:b].copy(), device=device),
                self.data[r-1, b:b+5].copy(), r, b)
