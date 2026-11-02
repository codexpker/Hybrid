"""Separate F training/CAL assembly. Metadata preview imports no Torch.

Real execution additionally needs an exact root phase release and external
resource supervision; this module does not authorize GPU/query work.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import load_manifest, resolve_plan, audit_events
from .event_data import ROLES

OPERATION = 'HYBRID_V1_STAGE2_EVENT_F'
PHASES = {'target_event_train': {'SUPPORT', 'NORMAL_SELECT'}, 'target_event_calibrate': {'NORMAL_CAL'}}
ACCEPTED = {'manifest': 'd9ce76a096eda0330dce37636a7a5a4d4971d5ea4f0dccf01112e4c1d14d8174',
            'plans': '1c35bb76984db226b516c3288f98161d0523ea89f25fbf958b2fa10bfc300952'}
SUPPORT = {'K1_choice0': '08d67ef6fbee89d6a80855ee5507898ffb90a8ce10e38bcaf1e5428202fb8194',
           'K1_choice1': '9b43b7834cb6ac198553e0c047a9c5c047a61158191c7292000e11affd6f8a99',
           'K1_choice2': '8e7ef8b069d05216a2a02b8f0624aa71dbe505da2647fd81746e7151f4caafb7',
           'K3_choice3': '6dcf0c67e4718416ea20024aab24744cfb6480d1cc5412744154baeed3637006'}
UPDATE = {'fusion','tf_fusion_delta','ms_decoder','forecast_head','event_head'}
RUNTIME_FILES={'hybrid_tsad/__init__.py'}|{'hybrid_tsad/hybrid_v1/'+n+'.py' for n in
    ('__init__','event_adaptation','event_data','target_data','data','config','provenance','cli','training',
     'objectives','scoring','model','temporal','fractional','lag','multiview')}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)


def bound(c,key):
    if sha(c[key]) != c[key+'_sha256']: raise PermissionError('Changed '+key)
    return Path(c[key])


def no_labels(value):
    if isinstance(value,dict):
        if any('label' in k.lower() or 'ledger' in k.lower() for k in value):
            raise PermissionError('Annotation ledger/labels are not model metadata')
        for v in value.values(): no_labels(v)
    elif isinstance(value,list):
        for v in value: no_labels(v)


def preflight(c,phase):
    """Metadata and checkpoint-byte hashes only; no arrays or model construction."""
    if phase not in PHASES or c.get('operation')!=OPERATION or c.get('scope') not in {'synthetic','public_target'}:
        raise PermissionError('Explicit independent F operation/phase required')
    if c.get('permissions') != {'query':False,'labels':False,'phase':phase}:
        raise PermissionError('Exact phase permissions required')
    if c.get('choice') not in SUPPORT: raise ValueError('K5 NA; four fixed choices only')
    if c.get('sampling_plan') is not None: raise PermissionError('F sampling_plan is unsupported')
    if c['scope']=='synthetic' and c.get('device')!='cpu': raise PermissionError('Synthetic CPU only')
    if Path(c['output']).exists(): raise FileExistsError('Fresh phase output required')
    if c.get('scope')=='public_target':
        if c.get('source_checkpoint_sha256')!='c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812':
            raise PermissionError('Original accepted source2025 required; never chained U/F state')
        expected=dict(history=512,horizon=32,seed=2025,steps=300,validate_every=50,lr=1e-4,weight_decay=.01,kind='huber',select_count=32,cal_count=512)
        if any(c.get(k)!=v for k,v in expected.items()): raise PermissionError('Fixed F300 recipe changed')
        if any(c.get(k+'_sha256')!=v for k,v in ACCEPTED.items()) or c.get('support_sha256')!=SUPPORT[c['choice']]:
            raise PermissionError('Accepted exact normal plans/support/manifest required')
    for k in ('manifest','plans','support','source_config','source_checkpoint','source_receipt'): bound(c,k)
    m,p,s=read(c['manifest']),read(c['plans']),read(c['support'])
    no_labels(m);no_labels(p);no_labels(s)
    if set(p)!={'history','horizon','seed','NORMAL_SELECT','NORMAL_CAL'} or any(p[k]!=c[k] for k in ('history','horizon','seed')):
        raise ValueError('Both explicit normal plans required; no role-wide fallback')
    entries=m['runs']
    if len(entries)!=3 or {r['role'] for r in entries}!=set(ROLES): raise ValueError('Three exact adaptation roles required')
    identity=tuple(entries[0][k] for k in ('domain','entity','run'))
    for r in entries:
        if tuple(r[k] for k in ('domain','entity','run'))!=identity or r.get('original_start')!=0 or r.get('format')!='npz' or r.get('observed_key')!='observed':
            raise ValueError('One original record with explicit mask required')
        if c['scope']=='public_target' and (r['bounds']!=ROLES[r['role']] or r['expected_shape']!=[16585,38]): raise ValueError('Fixed original role axes required')
    H,P=c['history'],c['horizon']
    for role,n in [('NORMAL_SELECT',c['select_count']),('NORMAL_CAL',c['cal_count'])]:
        points=p[role]; r=next(r for r in entries if r['role']==role);lo,hi=r['bounds']
        if len(points)!=n or len({x['index'] for x in points})!=n: raise ValueError('Exact unique normal plan count required')
        for x in points:
            if (tuple(x[k] for k in ('domain','entity','run'))!=identity or x['role']!=role
                or x.get('footprint')!=[x['index']-H,x['index']+P] or not lo+H<=x['index']<=hi-P):
                raise ValueError('Explicit full H/P normal footprint changed')
    k=int(c['choice'][1]); audit_events(s['events'],s['normal'],s['forbidden'],k,H,P)
    if s['K']!=k: raise ValueError('Selected K mismatch')
    for item in s['events']+s['normal']:
        if tuple(item[v] for v in ('domain','entity','run'))!=identity: raise ValueError('Support original identity changed')
    sc=read(c['source_config'])
    for key in ('model','sources','history','horizon','active_components','seed'):
        if sc.get(key)!=c.get(key): raise ValueError('Source architecture/recipe/seed changed: '+key)
    if phase=='target_event_calibrate':
        bound(c,'adapted_checkpoint'); receipt=read(bound(c,'adapted_receipt'))
        if receipt.get('status')!='COMPLETED' or receipt.get('phase')!='target_event_train': raise ValueError('Accepted F training receipt required')
        for key in ('source_checkpoint_sha256','manifest_sha256','plans_sha256','support_sha256','choice','adapted_checkpoint_sha256'):
            if receipt.get(key)!=c.get(key): raise ValueError('F CAL lineage changed: '+key)
    return m,p,s


def backend():
    # Lazy runtime imports keep cold metadata independent of Torch/official code.
    from . import cli,training
    return cli,training


def state(model):
    return {n:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for n,v in model.state_dict().items()}


def verify_trace(trace,support):
    required={e['event_id'] for e in support['events']};positive=set();negative=set()
    def contained(fp,item,H,P): return item['start']<=fp[0]+H<item['end'] and fp[1]==fp[0]+H+P
    H,P=trace['history'],trace['horizon']
    for row in trace['steps']:
        if not row.get('updated'): continue
        i=(row['attempt']//2)%support['K'];ev,nm=support['events'][i],support['normal'][i]
        pos=row['attempt']%2==0; item=ev if pos else nm
        if row['label']!=float(pos) or row['event_id']!=ev['event_id'] or not contained(row['footprint'],item,H,P) or not contained(row['normal_ssl_footprint'],nm,H,P):
            raise ValueError('Actual F/normalSSL footprint or round-robin trace changed')
        (positive if pos else negative).add(ev['event_id'])
    if positive!=required or negative!=required or set(trace['selected_seen_event_ids'])!=required or set(trace['selected_seen_matched_normal_ids'])!=required:
        raise ValueError('All K positive/normal updates and selected eligibility required')


def _action(c,phase,metadata):
    m,p,s=metadata; out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    write(out/'config.json',c)
    runs=load_manifest(c['manifest'],roles=PHASES[phase])
    if {r.role for r in runs}!=PHASES[phase]: raise ValueError('Wrong loaded roles')
    write(out/'input_runs.json',[dict(domain=r.domain,entity=r.entity,run=r.run,role=r.role,start=r.start,shape=list(r.values.shape),
          values_sha256=hashlib.sha256(r.values.tobytes()).hexdigest(),observed_sha256=hashlib.sha256(r.observed.tobytes()).hexdigest()) for r in runs])
    role='NORMAL_SELECT' if phase=='target_event_train' else 'NORMAL_CAL'
    selected=[r for r in runs if r.role==role];plan=p[role]
    resolved=resolve_plan(selected,plan,c['history'],c['horizon'])
    write(out/'normal_full_eligibility_plan.json',plan)
    for run,t in resolved:
        if not run.observed[t].all() or not (run.observed[t-c['history']+1:t].sum(0)>=2).all(): raise ValueError('Normal plan observed eligibility changed')
    cli,kernel=backend();path=c['source_checkpoint'] if phase=='target_event_train' else c['adapted_checkpoint']
    model=cli.load_model(c,path);cli.save_runtime(out,model);before=state(model)
    write(out/'initial_state_identity.json',before)
    params=set(dict(model.named_parameters()));flags={n:p.requires_grad for n,p in model.named_parameters()}
    if phase=='target_event_train':
        adapted,trace=kernel.adapt(model,'F',[r for r in runs if r.role=='SUPPORT'],selected,support=s,
            history=c['history'],horizon=c['horizon'],steps=c['steps'],lr=c['lr'],kind=c['kind'],seed=c['seed'],
            active_components=c.get('active_components'),validate_every=c['validate_every'],weight_decay=c['weight_decay'],validation_plan=plan,artifact_dir=out)
        if state(model)!=before or flags!={n:p.requires_grad for n,p in model.named_parameters()}: raise ValueError('Source state/flags mutated')
        after=state(adapted);allowed={n for n in params if n.split('.')[0] in UPDATE}
        write(out/'adapted_state_identity.json',after);write(out/'source_post_state_identity.json',state(model))
        if set(after)!=set(before) or any(after[n]!=before[n] for n in before if n not in allowed): raise ValueError('Frozen parameters/buffers changed')
        if set(trace['updated_names'])!=allowed: raise ValueError('F update policy changed')
        trace['history']=c['history'];trace['horizon']=c['horizon'];verify_trace(trace,s)
        write(out/'training.json',trace);write(out/'state_difference.json',dict(source_unchanged=True,allowed=sorted(allowed),changed=[n for n in before if before[n]!=after[n]],buffers_frozen=True))
        cli.save_model(out/'adapted.pt',adapted)
        extra=dict(adapted_checkpoint_sha256=sha(out/'adapted.pt'),updates=trace['updates'],selected_attempt=trace['selected_attempt'])
    else:
        # Builder eligibility was conservatively full H/P. Inference consumes
        # exactly the same origins with arrived H+current footprints; resolve_plan
        # enforces the runtime footprint rather than accepting a mismatched P.
        arrived_plan=[dict(x,footprint=[x['index']-c['history'],x['index']+1]) for x in plan]
        write(out/'CAL_arrived_execution_plan.json',arrived_plan)
        records=list(kernel.iter_infer(model,selected,history=c['history'],horizon=c['horizon'],seed=c['seed'],event_logits=True,active_components=c.get('active_components'),index_plan=arrived_plan))
        if len(records)!=len(plan) or any(r['index']!=x['index'] or
                any(r.get(k)!=x[k] for k in ('domain','entity','run')) or
                r.get('replay',{}).get('input_footprint')!=x['footprint'] or
                not np.isfinite(r.get('event_logit',np.nan)) for r,x in zip(records,arrived_plan)):
            raise ValueError('CAL must retain all fixed continuous valid event logits; no zero filling')
        if state(model)!=before: raise ValueError('CAL changed model state')
        write(out/'calibration_post_state_identity.json',state(model))
        threshold=kernel.logit_threshold(records)
        if not np.isfinite(threshold): raise ValueError('Invalid logit q99')
        write(out/'normal_event_logit_records.json',records);write(out/'event_logit_calibration.json',dict(score_kind='event_logit',q=.99,method='higher',strict_greater=True,threshold=threshold))
        extra=dict(updates=0,score_kind='event_logit',threshold=threshold)
    if sha(c['source_checkpoint'])!=c['source_checkpoint_sha256']: raise ValueError('Source checkpoint file changed')
    receipt=dict(status='COMPLETED',phase=phase,choice=c['choice'],loaded_roles=sorted(PHASES[phase]),query_or_labels=False,
                 **{k:c[k] for k in ('source_checkpoint_sha256','manifest_sha256','plans_sha256','support_sha256')},**extra)
    write(out/'receipt.json',receipt);return receipt


def run_phase(config,release,trusted,phase):
    c=read(config)
    if not release or not trusted or sha(release)!=trusted: raise PermissionError('Exact root F phase release required')
    r=read(release);closure=c['source_sha256'];base=Path(__file__).resolve().parents[2]
    if (r.get('status')!='APPROVED' or r.get('approved_by')!='root' or r.get('operation')!=OPERATION
        or r.get('allowed_actions')!=[phase] or r.get('config_sha256')!=sha(config) or r.get('source_sha256')!=closure): raise PermissionError('Changed F operation/source/config release')
    if not RUNTIME_FILES.issubset(closure): raise PermissionError('Complete F import/runtime source closure required')
    for name,digest in closure.items():
        f=(base/name).resolve()
        if base not in f.parents or sha(f)!=digest: raise PermissionError('Source closure mismatch')
    metadata=preflight(c,phase)
    result=_action(c,phase,metadata)
    write(Path(c['output'])/'execution_identity.json',dict(config_sha256=sha(config),release_sha256=trusted,source_sha256=closure))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--phase',required=True,choices=sorted(PHASES))
    p.add_argument('--execute',action='store_true');p.add_argument('--release');p.add_argument('--trusted-release-sha256');a=p.parse_args(argv)
    if a.execute: result=run_phase(a.config,a.release,a.trusted_release_sha256,a.phase)
    else:
        preflight(read(a.config),a.phase);result=dict(metadata_preflight='PASS',arrays_or_model_opened=False)
    print(json.dumps(result))


if __name__=='__main__':main()
