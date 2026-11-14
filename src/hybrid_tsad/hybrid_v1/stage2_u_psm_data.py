"""PSM matched plain-U data only: physical roles and one fixed injection.

Reuses audited role/plan validators. PSM retains native NaNs and masks; its
writer differs from the old finite-only SMD writer only in NaN-safe equality.
No Torch, detector, loss, score, threshold or QUERY donor is used here.
"""
import copy
import json
from pathlib import Path
import numpy as np

from .stage2_u_data import identity, write_json_create_only
from .stage2_u_matched import load_roles, resolve_u_plan, role_metadata
from .contamination_data import no_supervision

H, P, ATTEMPTS = 512, 32, 200
BOUNDS = {'FIT': [0,79488], 'SELECT': [80032,105984], 'CAL': [106528,132481]}


def std_observed(values, observed):
    count=observed.sum(0)
    if (count==0).any():raise ValueError('Every channel requires FIT observations')
    safe=np.where(observed,values,0.).astype(np.float64)
    mean=safe.sum(0)/count
    variance=np.where(observed,(safe-mean)**2,0.).sum(0)/count
    return np.sqrt(variance), count


def choose_donors(values, observed, fit_plan):
    """Root fixed: 20 slots PCG64(99179); earliest nonoverlap donor geometry."""
    if len(fit_plan)!=ATTEMPTS:raise ValueError('Exact original FIT200 required')
    slots=np.random.default_rng(99179).choice(ATTEMPTS,20,replace=False)
    untouched=np.ones(ATTEMPTS,bool);untouched[slots]=False
    forbidden=np.zeros(len(values),bool)
    for i,entry in enumerate(fit_plan):
        if untouched[i]: forbidden[slice(*entry['footprint'])]=True
    bad=forbidden|~observed.all(1)
    prefix=np.r_[0,np.cumsum(bad)]
    targets=np.arange(H,len(values)-P+1)
    candidates=targets[(prefix[targets+P]-prefix[targets-H])==0]
    greedy=[];end=-1
    for t in candidates:
        if t-H>=end:greedy.append(int(t));end=int(t+P)
    if len(greedy)<20:
        raise ValueError('Insufficient 20 disjoint complete donor footprints; no relaxation')
    return slots, np.asarray(greedy[:20]), dict(eligible_origins=len(candidates),
        earliest_greedy_nonoverlap_count=len(greedy),all_greedy_origins=greedy,
        unchanged_background_union_points=int(forbidden.sum()))


def inject_fixed(values, observed, donors):
    result=values.copy(); sigma,count=std_observed(values,observed)
    injected=np.zeros(len(values),bool)
    for t in donors:injected[t:t+P]=True
    affected=observed & injected[:,None]
    shifted=(values.astype(np.float64)+2*sigma[None]).astype(np.float32)
    result[affected]=shifted[affected]
    if not np.array_equal(result[~affected],values[~affected],equal_nan=True):
        raise AssertionError('Injection modified undeclared cells')
    return result,injected,affected,sigma,count


def write_role(path,values,observed,bounds):
    lo,hi=bounds
    with Path(path).open('xb') as handle:
        np.savez(handle,values=np.asarray(values[lo:hi],np.float32),observed=np.asarray(observed[lo:hi],bool))
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'values','observed'} or not np.array_equal(z['values'],values[lo:hi],equal_nan=True) or not np.array_equal(z['observed'],observed[lo:hi]):
            raise AssertionError('Physical role differs from exact native slice')
    return dict(domain='PSM',entity='anonymous_PSM',run='train_original',role=None,
        path=str(Path(path).resolve()),format='npz',key='values',observed_key='observed',
        sha256=identity(path)['sha256'],expected_shape=[hi-lo,values.shape[1]],bounds=[0,hi-lo],original_start=lo)


def select_query(labels):
    """Metadata/time-only event selection; no score or error enters selection."""
    labels=np.asarray(labels)
    if labels.shape!=(87841,) or not np.isin(labels,[0,1]).all():raise ValueError('Exact original PSM truth axis')
    edge=np.diff(np.r_[0,labels,0].astype(np.int8))
    events=[(int(s),int(e)) for s,e in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1))]
    qualified=[(s,e) for s,e in events if s>=H and e<len(labels)]
    chosen=qualified[:5]
    if len(chosen)!=5:raise ValueError('First five full uncensored QUERY events unavailable')
    if not any(e-s>=1024 for s,e in chosen):
        long=next(((s,e) for s,e in qualified if e-s>=1024),None)
        if long is None:raise ValueError('No full long event for the preset duration check')
        if long not in chosen:chosen.append(long)
    ranges=[[max(H,s-H),min(len(labels),e+512)] for s,e in chosen]
    normal_edges=np.diff(np.r_[0,(labels==0),0].astype(np.int8))
    normals=[]
    for s,e in zip(np.flatnonzero(normal_edges==1),np.flatnonzero(normal_edges==-1)):
        s=max(H,int(s));e=int(e)
        if e-s>=512:normals.append([s,s+512])
        if len(normals)==5:break
    if len(normals)!=5:raise ValueError('Five fixed normal negative-control runs unavailable')
    mask=np.zeros(len(labels),bool)
    for s,e in ranges+normals:mask[s:e]=True
    index=np.flatnonzero(mask)
    intersected=[dict(start=s,end_exclusive=e,points=e-s,scored_points=int(mask[s:e].sum()),
        complete=bool(mask[s:e].all()),selected_main=(s,e) in chosen)
        for s,e in events if mask[s:e].any()]
    merge_edge=np.diff(np.r_[0,mask,0].astype(np.int8))
    merged=[[int(s),int(e)] for s,e in zip(np.flatnonzero(merge_edge==1),np.flatnonzero(merge_edge==-1))]
    return index,dict(selection='first5 full events by original time; add earliest duration>=1024 if absent; no performance selection',
        selected_events=[dict(start=s,end_exclusive=e,points=e-s) for s,e in chosen],
        event_context_ranges=ranges,preset_normal_ranges=normals,union_ranges=merged,
        scored_points=len(index),positive_points=int(labels[index].sum()),negative_points=int((labels[index]==0).sum()),
        all_intersected_original_events=intersected,
        other_intersected_anomalies='retain original labels/coverage; never relabel context as normal',
        discontinuity='original time indices retained; no concatenated event/confirmation sequence')


def prepare(config):
    def asset(key):
        a=config[key]
        if identity(a['path'])['sha256']!=a['sha256']:raise ValueError('Input bytes changed: '+key)
        return Path(a['path'])
    native=asset('native_npz'); old_manifest=json.loads(asset('old_manifest').read_text())
    original_plan=json.loads(asset('old_plan').read_text()); replay=json.loads(asset('fit_replay').read_text())
    if len(replay)!=200 or [r['attempt'] for r in replay]!=list(range(200)) or any(r['actual_mask_seed']!=2025+i for i,r in enumerate(replay)):
        raise ValueError('Fixed original attempt clock required')
    with np.load(native,allow_pickle=False) as z: values=z['values'].copy();observed=z['observed'].copy()
    if values.shape!=(132481,25) or values.dtype!=np.float32 or observed.dtype!=bool or observed.shape!=values.shape:
        raise ValueError('Original PSM values/observed geometry')
    if np.isinf(values).any() or not np.isfinite(values[observed]).all():raise ValueError('Invalid observed values')
    if {r['role']:r['bounds'] for r in old_manifest['runs']}!=BOUNDS:raise ValueError('Preserve original role boundaries')
    fields=('domain','entity','run','role','index','footprint')
    fit=[{k:r[k] for k in fields} for r in replay]
    for entry in fit:
        t=entry['index']
        if entry['footprint']!=[t-H,t+P] or entry['role']!='FIT' or not H<=t<=79488-P:
            raise ValueError('Illegal old FIT footprint')
    slots,donors,eligibility=choose_donors(values[:79488],observed[:79488],fit)
    changed,injected,affected,sigma,counts=inject_fixed(values[:79488],observed[:79488],donors)
    output=Path(config['output']); output.mkdir(parents=True,exist_ok=False)
    learner=output/'learner'; audit=output/'audit_only'; learner.mkdir();audit.mkdir()
    descriptors={}
    for role,bounds in BOUNDS.items():
        descriptor=write_role(learner/(role+'.npz'),values,observed,bounds);descriptor['role']=role;descriptors[role]=descriptor
    descriptor=write_role(learner/'FIT_B.npz',changed,observed[:79488],[0,79488]);descriptor['role']='FIT'
    replaced=copy.deepcopy(fit)
    for slot,t in zip(slots,donors):replaced[int(slot)].update(index=int(t),footprint=[int(t)-H,int(t)+P])
    artifacts=[]
    for name in ('A','B','C'):
        current=dict(history=H,horizon=P,seed=2025,FIT=fit if name=='A' else replaced,
            SELECT=copy.deepcopy(original_plan['SELECT']),CAL=copy.deepcopy(original_plan['CAL']))
        manifest={'runs':[descriptor if role=='FIT' and name=='B' else descriptors[role] for role in BOUNDS]}
        no_supervision(current);no_supervision(manifest)
        mp=learner/('MANIFEST_'+name+'.json');pp=learner/('PLAN_'+name+'.json')
        write_json_create_only(mp,manifest);write_json_create_only(pp,current)
        role_metadata(mp)
        runs=load_roles(mp,{'FIT','SELECT','CAL'})
        for role in BOUNDS:resolve_u_plan([r for r in runs if r.role==role],current[role],H,P,role,arrived=role=='CAL')
        artifacts.append(dict(condition=name,manifest=identity(mp),plan=identity(pp)))
    # All untouched background windows remain byte-equivalent modulo NaN equality.
    for slot,entry in enumerate(fit):
        if slot not in slots:
            lo,hi=entry['footprint']
            if not np.array_equal(changed[lo:hi],values[lo:hi],equal_nan=True):raise AssertionError('Unselected window polluted')
    truth=asset('query_truth')
    with np.load(truth,allow_pickle=False) as z:labels=z['full_labels'].copy()
    query_index,query_report=select_query(labels)
    with (audit/'QUERY_SELECTION.npz').open('xb') as handle:np.savez(handle,index=query_index,full_labels=labels)
    write_json_create_only(audit/'QUERY_SELECTION.json',query_report)
    modified_cells=int((affected & (changed!=values[:79488])).sum())
    report=dict(status='CPU_DATA_PREPARED_NO_MODEL',seed=2025,source_checkpoint_sha256=config['source_checkpoint_sha256'],
        original_roles=BOUNDS,original_background_health='UNKNOWN: nominal benchmark training background, not certified clean',
        history=H,horizon=P,attempts=200,changed_slots=slots.tolist(),donor_origins=donors.tolist(),
        donor_eligibility=eligibility,donor_footprints=[[int(t)-H,int(t)+P] for t in donors],
        injection=dict(family='synthetic synchronous short level shift/change; not real or cross-lag anomaly',
            affected_channels=list(range(25)),formula='float32(float64(raw)+2*FIT_observed_std_ddof0)',
            sigma=sigma.tolist(),observed_counts=counts.tolist(),constant_channels=np.flatnonzero(sigma==0).tolist(),
            learner_windows_replaced=20,window_fraction=.1,forecast_GT_injected_exposure_fraction=.1,
            unique_injected_points=int(injected.sum()),unique_modified_cells=modified_cells,
            actual_modified_cell_fraction_of_all_update_cells=modified_cells/(200*544*25),
            injected_point_exposure_fraction_HplusP=int(injected.sum())/(200*544),
            unique_disjoint_synthetic_intervals=20,overlapping_window_independence='20 inserted intervals are construction units, not 20 independent real faults',
            source_FIT_unique_modified_point_fraction=int(injected.sum())/79488),
        missing_preserved={role:int((~observed[s:e]).sum()) for role,(s,e) in BOUNDS.items()},
        no_QUERY_donor=True,SELECT_CAL_original_sequences_exact=True,artifacts=artifacts,query_selection=query_report,
        previous_U200='Historical original same-timepoint200 remains frozen; repaired mode/RNG/physical-role protocol is new, no numeric reproduction claim',
        C_control='Same FIT source and same donor windows unchanged; not certified anomaly-free',
        no_training=True,GPUh=0,source_config=config)
    write_json_create_only(audit/'PREPARATION_RECEIPT.json',report)
    return report
