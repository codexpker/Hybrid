"""Final pre-forward correction: input+target contamination, original A/C reused.

The earlier target-only B remains an unrun historical preparation. This module
creates a new B file/manifest only, not another training arm or loss.
"""
import copy
import json
from pathlib import Path
import numpy as np

from .stage2_u_data import identity, write_json_create_only
from .stage2_u_psm_data import std_observed, write_role
from .stage2_u_matched import role_metadata, load_roles, resolve_u_plan


def inject_input_and_target(values,observed,donors):
    result=values.copy();sigma,count=std_observed(values,observed)
    mask=np.zeros(len(values),bool)
    for target in donors:
        if target-128<0 or target+32>len(values):raise ValueError('Final160 interval outside FIT')
        mask[target-128:target+32]=True
    affected=mask[:,None]&observed
    shifted=(values.astype(np.float64)+2*sigma[None]).astype(np.float32)
    result[affected]=shifted[affected]
    if not np.array_equal(result[~affected],values[~affected],equal_nan=True):
        raise AssertionError('Undeclared cells modified')
    return result,mask,affected,sigma,count


def prepare_final_B(config):
    def read_asset(key):
        a=config[key]
        if identity(a['path'])['sha256']!=a['sha256']:raise ValueError('Original asset changed: '+key)
        return json.loads(Path(a['path']).read_text())
    old=read_asset('original_data_receipt');base_manifest=read_asset('manifest_A')
    plan_B=read_asset('plan_B');plan_A=read_asset('plan_A')
    role=next(x for x in base_manifest['runs'] if x['role']=='FIT')
    if identity(role['path'])['sha256']!=role['sha256']:raise ValueError('Original FIT array changed')
    with np.load(role['path'],allow_pickle=False) as z:values=z['values'].copy();observed=z['observed'].copy()
    if values.shape!=(79488,25):raise ValueError('Original physical FIT geometry')
    changed,mask,affected,sigma,counts=inject_input_and_target(values,observed,old['donor_origins'])
    if not np.array_equal(sigma,np.asarray(old['injection']['sigma'])):raise ValueError('FIT-only std changed')
    slots=old['changed_slots']
    for i,e in enumerate(plan_A['FIT']):
        if i not in slots:
            lo,hi=e['footprint']
            if mask[lo:hi].any() or not np.array_equal(changed[lo:hi],values[lo:hi],equal_nan=True):
                raise AssertionError('Untouched180 background window changed')
    touched=past=arrived=future=0
    for e in plan_B['FIT']:
        t=e['index'];a=int(mask[t-512:t].sum());b=int(mask[t:t+32].sum())
        touched+=int(a+b>0);past+=a;future+=b;arrived+=int(mask[t-511:t+1].sum())
    if (touched,past,arrived,future,int(mask.sum()))!=(20,2560,2580,640,3200):
        raise AssertionError('Final fixed160 exposure mismatch')
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    learner=output/'learner';audit=output/'audit_only';learner.mkdir();audit.mkdir()
    new_role=write_role(learner/'FIT_B_INPUT160.npz',changed,observed,[0,len(values)]);new_role['role']='FIT'
    manifest=copy.deepcopy(base_manifest)
    manifest['runs']=[new_role if e['role']=='FIT' else e for e in manifest['runs']]
    mp=learner/'MANIFEST_B.json';write_json_create_only(mp,manifest);role_metadata(mp)
    runs=load_roles(mp,{'FIT'})
    resolve_u_plan(runs,plan_B['FIT'],512,32,'FIT')
    report=dict(status='FINAL_B_INPUT_AND_TARGET_CPU_DATA_READY',
        previous_target_only_status='UNRUN_SUPERSEDED_TARGET_ONLY_NOT_FULL_INPUT_CONTAMINATION',
        source_checkpoint_sha256=old['source_checkpoint_sha256'],
        original_data_receipt=config['original_data_receipt'],
        unique_change_from_unrun_preparation='B interval [t,t+32) -> [t-128,t+32); all else byte-reused',
        condition_B_manifest=identity(mp),condition_B_array=identity(new_role['path']),
        reuse={key:config[key] for key in ('manifest_A','manifest_C','plan_A','plan_B','plan_C','query_plan')},
        final_fixed_formula='all observed25 cells in [t-128,t+32): float32(float64(raw)+2*originalFIT_observed_std_ddof0)',
        touched_windows=touched,window_fraction=touched/200,unique_injected_points=int(mask.sum()),
        unique_modified_cells=int((affected&(changed!=values)).sum()),
        total_update_footprint_points=200*544,injected_footprint_exposure_fraction=int(mask.sum())/(200*544),
        forecast_history_injected_exposure_points=past,forecast_history_fraction=past/(200*512),
        reconstruction_context_injected_exposure_points=arrived,reconstruction_context_fraction=arrived/(200*512),
        future_GT_injected_exposure_points=future,future_GT_injected_fraction=future/(200*32),
        planted_intervals=[[int(t)-128,int(t)+32] for t in old['donor_origins']],
        sigma=sigma.tolist(),observed_counts=counts.tolist(),
        original_NaNs_and_observed_unchanged=True,unreplaced180_windows_unchanged=True,
        SELECT_CAL_QUERY_unchanged=True,A_C_unchanged=True,no_new_scores_or_training=True,
        reason='Input-contamination identifiability correction before any candidate forward/performance; not a dose search',
        natural_background_health='UNKNOWN; neither A nor C certified clean',
        anomaly_family='synthetic synchronous short level shift; not natural fault or Cross-lag evidence',
        source_config=config,GPUh=0)
    if report['unique_modified_cells']!=80000:raise AssertionError('Expected80000 observed cells')
    write_json_create_only(audit/'FINAL_B_RECEIPT.json',report)
    return report
