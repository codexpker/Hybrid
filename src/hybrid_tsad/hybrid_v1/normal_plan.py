"""Metadata-only normal exposure plan from an accepted F trace; no data loader."""
import argparse
from collections import Counter
from pathlib import Path
from . import normal_control as normal
from . import event_adaptation as event

KEYS = ('manifest','plans','support','reference_trace','reference_receipt','reference_acceptance')


def derive(config, workspace):
    """Map signed remote metadata names to an existing local workspace, no PT/NPZ."""
    c=event.read(config);root=Path(workspace).resolve();prefix='/home/a312/tsad_research/'
    if (c.get('operation')!=normal.OPERATION or c.get('mode')!='N_NEG300'
        or c.get('permissions')!={'phase':'target_normal_train','query':False,'labels':False}
        or c.get('support_sha256')!=event.SUPPORT.get(c.get('choice'))
        or any(c.get(k+'_sha256')!=v for k,v in event.ACCEPTED.items())
        or any(c.get(k)!=v for k,v in dict(history=512,horizon=32,seed=2025,steps=300,select_count=32,cal_count=512).items())):
        raise PermissionError('Exact accepted public F-normal metadata recipe required')
    values={};inputs={}
    for key in KEYS:
        name=c[key]
        if (not name.startswith(prefix) or not name.endswith('.json') or
            any(word in Path(name).name.lower() for word in ('label','ledger'))):raise PermissionError('Metadata JSON only; no annotation file')
        path=(root/name[len(prefix):]).resolve()
        if root not in path.parents or event.sha(path)!=c[key+'_sha256']:raise PermissionError('Changed '+key)
        values[key]=event.read(path);inputs[key]=dict(path=str(path),sha256=c[key+'_sha256'])
    m,p,s,tr,r,a=[values[k] for k in KEYS]
    case=next((v for v in a.get('cases',[]) if v.get('choice')==c['choice']),{}) if 'cases' in a else a
    if (a.get('approved_by')!='root' or not a.get('status','').startswith('ACCEPTED')
        or case.get('adapted_checkpoint_sha256',case.get('checkpoint_sha256'))!=r.get('adapted_checkpoint_sha256')
        or case.get('receipt_sha256',c['reference_receipt_sha256'])!=c['reference_receipt_sha256']
        or r.get('phase')!='target_event_train' or r.get('status')!='COMPLETED'
        or any(r.get(k)!=c.get(k) for k in ('choice','manifest_sha256','plans_sha256','support_sha256','source_checkpoint_sha256'))):
        raise PermissionError('Accepted F train lineage required')
    if (set(p)!={'history','horizon','seed','NORMAL_SELECT','NORMAL_CAL'} or
        any(p.get(k)!=c[k] for k in ('history','horizon','seed'))):raise ValueError('Fixed common plans required')
    for role,count in [('NORMAL_SELECT',32),('NORMAL_CAL',512)]:
        run=next(v for v in m['runs'] if v['role']==role);lo,hi=run['bounds'];points=p[role]
        if (run['bounds']!=event.ROLES[role] or len(points)!=count or len({x['index'] for x in points})!=count):
            raise ValueError('Common normal count/role differs')
        for x in points:
            if (any(x[k]!=run[k] for k in ('domain','entity','run','role')) or
                x['footprint']!=[x['index']-512,x['index']+32] or not lo+512<=x['index']<=hi-32):
                raise ValueError('Common normal plan footprint differs')
    plan=normal.normal_plan_from_f_trace(tr,s,history=512,horizon=32,seed=2025,
        choice=c['choice'],trace_sha256=c['reference_trace_sha256'])
    if len(plan['rows'])!=300:raise ValueError('All original F attempts required')
    counts=Counter((x['group'],key,x[key]['index']) for x in plan['rows'] for key in ('A','B') if x[key] is not None)
    summary=dict(choice=c['choice'],inputs=inputs,source_config_sha256=c['source_config_sha256'],
        source_checkpoint_sha256=c['source_checkpoint_sha256'],reference_adapted_checkpoint_sha256=r['adapted_checkpoint_sha256'],
        attempts=300,normal_SSL=300,normal_negative_BCE=150,normal_forward_exposures=450,positive_forward_exposures=0,
        unique_normal_targets=len({key[2] for key in counts}),
        unique_targets_by_group={g:len({i for group,_,i in counts if group==g}) for g in sorted({k[0] for k in counts})},
        exposure_counts=[dict(group=g,path=k,index=i,count=n) for (g,k,i),n in sorted(counts.items())],
        arrays_PT_query_or_new_labels_opened=False,plan_is_metadata_not_training=True)
    return plan,summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','workspace','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args(argv);plan,report=derive(a.config,a.workspace);out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False);event.write(out/'NORMAL_PLAN.json',plan)
    event.write(out/'PLAN_AUDIT.json',dict(report,normal_plan_sha256=event.sha(out/'NORMAL_PLAN.json')))


if __name__=='__main__':main()
