"""Fixed source-randomness repeats; target evaluation requires the global seal."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from . import native_position_aligned_training as aligned
from . import native_event_rank_training as old
from .native_v_conditioning import sha
from .native_v_conditioning_runner import read, dump

OUT=Path('results/native_position_aligned_repeats_20261124_v1')
PRIOR=Path('results/native_event_rank_seed_repeats_20261122_v1')
SEEDS=(2026,2027)
SELECT_KEYS=('normal_select','positive_select','select_stratum_sizes','swat_select','smd_select')

def prepare(w):
    original=w/aligned.OUT
    fixed=old.load_plan(original)
    root=w/OUT;root.mkdir(parents=True,exist_ok=False)
    for s in SEEDS:
        prior=w/PRIOR/f'seed{s}';plan=old.load_plan(prior)
        assert read(prior/'PLAN.json')['seed']==s
        for key in SELECT_KEYS:np.testing.assert_array_equal(plan[key],fixed[key])
        assert not np.array_equal(plan['swat_fit'],fixed['swat_fit'])
        out=root/f'seed{s}';out.mkdir()
        for name in ('PLAN.npz','PLAN.json'):shutil.copyfile(prior/name,out/name)
        # The original scale is deliberately shared, not refitted per seed.
        for name in ('SCALE.json','FIT_NORMAL_V.npz'):shutil.copyfile(original/name,out/name)
        dump(out/'CONFIG.json',dict(**{k:v for k,v in read(original/'CONFIG.json').items()
            if k not in ('seed','source_plan_sha256','legacy_trajectory_scan')},seed=s,
            source_plan_sha256=sha(out/'PLAN.npz'),fixed_select_keys=list(SELECT_KEYS),
            select_panel_identical_to_2025=True,source_scale_origin=str(aligned.OUT/'SCALE.json'),
            source_scale_sha256=sha(out/'SCALE.json'),training_rng_seed=s,
            compatibility='same numerical training code and original frozen source scale; only seed/archived FIT draws vary',
            existing_checkpoints=[str(p.relative_to(w)) for p in (prior/'train').rglob('*.pt')],
            action='rebuild seven-checkpoint trajectory under common full-position selection'))
    dump(root/'CONFIG.json',dict(seeds=SEEDS,arms=aligned.ARMS,source_only_until_all_sealed=True,
        source_2025_reused=str(aligned.OUT),target_read=False))

def seal(w):
    seeds={}
    for s in SEEDS:
        relative=OUT/f'seed{s}';aligned.seal(w,output=relative)
        seeds[str(s)]=read(w/relative/'ALL_MODELS_FROZEN.json')
    dump(w/OUT/'ALL_MODELS_FROZEN.json',dict(status='COMPLETED_BEFORE_TARGET',seeds=seeds,
        query_read=False,seed2025_reused=str(aligned.OUT)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','check','train','seal'))
    p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--arm',choices=aligned.ARMS);a=p.parse_args()
    w=Path.cwd()
    if a.action=='prepare':prepare(w)
    elif a.action=='seal':seal(w)
    else:
        assert a.seed is not None
        kwargs=dict(output=OUT/f'seed{a.seed}',training_seed=a.seed)
        if a.action=='check':aligned.check(w,**kwargs)
        else:
            assert a.arm is not None
            aligned.train(w,a.arm,**kwargs)
