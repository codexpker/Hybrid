"""Post-hoc fixed-score public diagnostics; no AP, refit, model, or new score."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hybrid_tsad.hybrid_v1.scoring import apply_calibration,weighted_quantile
from hybrid_tsad.engine.events import binary_events

METHODS=('HybridV1_Z_S','persistence','history_mean512')
QUANTILES=(0.,.25,.5,.75,.9,.95,.99,1.)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bound(spec):
    if sha(spec['path'])!=spec['sha256']:raise PermissionError('Changed original input: '+str(spec['path']))
    return json.loads(Path(spec['path']).read_text(encoding='utf-8'))
def dump(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def ratio(a,b):return float(a/b) if b else None
def local(cfg,path):
    prefix=cfg['remote_root'].rstrip('/')+'/'
    if not path.startswith(prefix) or '..' in Path(path[len(prefix):]).parts:raise PermissionError('One archived workspace only')
    return Path(cfg['local_root'])/path[len(prefix):]
def quantile_description(values):
    x=np.asarray(values,np.float64)
    if not np.isfinite(x).all():raise ValueError('Describe only valid original scores')
    q=np.quantile(x,QUANTILES) if len(x) else np.full(len(QUANTILES),np.nan)
    result={('descriptive_q%03d'%round(p*100)):float(v) if np.isfinite(v) else None for p,v in zip(QUANTILES,q)}
    result.update(n=len(x),mean=float(x.mean()) if len(x) else None,std=float(x.std()) if len(x) else None,
        IQR=float(q[3]-q[1]) if len(x) else None,quantile_interpolation='linear; description only, never a target threshold')
    return result

def roc_pair_cells(entities):
    """P(score_pos>score_neg)+0.5ties; integer U2 avoids a huge pair matrix."""
    cells=[]
    negatives={e:np.sort(a['score'][a['label']==0]) for e,a in entities.items()}
    for positive_entity,a in sorted(entities.items()):
        positive=a['score'][a['label']==1]
        for negative_entity,negative in sorted(negatives.items()):
            left=np.searchsorted(negative,positive,side='left');right=np.searchsorted(negative,positive,side='right')
            wins=int(left.sum(dtype=np.int64));ties=int((right-left).sum(dtype=np.int64));pairs=len(positive)*len(negative)
            cells.append(dict(positive_entity=positive_entity,negative_entity=negative_entity,
                relation='within_entity' if positive_entity==negative_entity else 'cross_entity',
                positive_points=len(positive),negative_points=len(negative),pairs=pairs,wins=wins,ties=ties,
                U2=2*wins+ties,ROC_AUC=ratio(2*wins+ties,2*pairs)))
    total_pairs=sum(r['pairs'] for r in cells);total_u2=sum(r['U2'] for r in cells)
    summary=dict(total_pairs=total_pairs,total_U2=total_u2,total_ROC_AUC=ratio(total_u2,2*total_pairs))
    for kind in ('within_entity','cross_entity'):
        subset=[r for r in cells if r['relation']==kind];pairs=sum(r['pairs'] for r in subset);u2=sum(r['U2'] for r in subset)
        summary[kind]=dict(pairs=pairs,U2=u2,pair_weight=ratio(pairs,total_pairs),ROC_AUC=ratio(u2,2*pairs),
            weighted_contribution=ratio(u2,2*total_pairs),NA_reason=None if pairs else 'NO_POSITIVE_NEGATIVE_PAIRS_IN_THIS_STRATUM')
    if total_pairs:
        reconstructed=sum(v['weighted_contribution'] or 0 for v in (summary['within_entity'],summary['cross_entity']))
        if abs(reconstructed-summary['total_ROC_AUC'])>2e-15:raise ValueError('ROC within/cross weighted identity failed')
        summary['weighted_identity_error']=abs(reconstructed-summary['total_ROC_AUC'])
    return cells,summary

def alarm_runs(index,strict,confirmed,labels):
    if np.any(confirmed & ~strict):raise ValueError('Current confirmed alarm is not a strict subset')
    if not len(index):return []
    full=np.zeros(int(index[-1])+1,bool);full[index]=strict
    cf=np.zeros(len(full),bool);cf[index]=confirmed
    y=np.zeros(len(full),np.int8);y[index]=labels
    output=[]
    for lo,hi in binary_events(full):
        length=hi-lo
        if int(cf[lo:hi].sum())!=max(length-2,0):raise ValueError('Existing confirmed alarm contradicts strict contiguous run')
        output.append(dict(start=lo,end_exclusive=hi,length=length,
            length_bin=str(length) if length<=3 else '4..8' if length<=8 else '9..32' if length<=32 else '>32',
            strict_TP=int((y[lo:hi]==1).sum()),strict_FP=int((y[lo:hi]==0).sum()),
            confirmed_TP=int((cf[lo:hi] & (y[lo:hi]==1)).sum()),confirmed_FP=int((cf[lo:hi] & (y[lo:hi]==0)).sum()),
            lost_TP=int((~cf[lo:hi] & (y[lo:hi]==1)).sum()),lost_FP=int((~cf[lo:hi] & (y[lo:hi]==0)).sum())))
    return output

def load_original_arrays(cfg,production,summary):
    base=Path(cfg['production_summary']['path']).parent;arrays={};bindings=[];normal_rows=[]
    accepted={(r['method'],r['domain'],r['entity'],r['run']):r for r in summary['per_entity']}
    if len(accepted)!=162 or len(production['banks'])!=162:raise PermissionError('All original162 banks required')
    for desc in production['banks']:
        mk=tuple(desc[k] for k in ('method','domain','entity','run'));method,domain,entity,_=mk
        if mk not in accepted:raise PermissionError('Unknown production entity')
        directory=base/'entities'/domain/entity/method;calc=directory/'calculation_inputs.npz';prov=json.loads((directory/'provenance.json').read_text())
        if sha(calc)!=prov['calculation_inputs_sha256']:raise PermissionError('Original evaluation arrays changed')
        if sha(local(cfg,desc['directory'])/'manifest.json')!=desc['manifest_sha256']:raise PermissionError('Original bank changed')
        with np.load(calc,allow_pickle=False) as z:
            valid=z['valid'];index=z['index'][valid];score=z['score'][valid];full_labels=z['full_labels'];labels=full_labels[index]
        if not np.isfinite(score).all():raise ValueError('Nonfinite valid original score')
        # Existing frozen alarms are read, not reevaluated with a new rule.
        point=[]
        with (directory/'point_coverage.csv').open(encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['valid']=='True':point.append(row)
        if len(point)!=len(index) or not np.array_equal(np.array([int(r['index']) for r in point]),index):
            raise ValueError('Original point alarm axes differ')
        strict=np.array([r['strict_point']=='True' for r in point],bool);confirmed=np.array([r['confirmed3_current']=='True' for r in point],bool)
        tau=accepted[mk]['streams']['strict_point']['point']['threshold']
        if not np.array_equal(strict,score>tau) or not np.array_equal(np.array([int(r['label']) for r in point]),labels):
            raise ValueError('Original point score/label/threshold replay mismatch')
        a=dict(index=index,score=score,label=labels,strict=strict,confirmed=confirmed,threshold=tau)
        arrays[mk]=a
        for name,alarm in [('strict_point',strict),('confirmed3_current',confirmed)]:
            old=accepted[mk]['streams'][name]
            if (int((alarm&(labels==1)).sum())!=old['point']['TP'] or int((alarm&(labels==0)).sum())!=old['point']['FP']):
                raise ValueError('Original fixed alarm counts differ')
            normal_rows.append(dict(method=method,domain=domain,entity=entity,stream=name,
                source_threshold=tau,**old['normal']))
        bindings.append(dict(method=method,domain=domain,entity=entity,run=mk[3],path=str(calc),sha256=sha(calc),
            point_coverage_sha256=sha(directory/'point_coverage.csv'),bank_manifest_sha256=desc['manifest_sha256']))
    return arrays,bindings,accepted,normal_rows

def source_calibration(cfg):
    source={};records_by={};keys=None;outputs=[]
    for method in METHODS:
        records=bound(cfg['source_CAL_records'][method]);cal=bound(cfg['source_calibrations'][method])
        locations=[tuple(r[k] for k in ('domain','entity','run','index')) for r in records]
        if len(records)!=1024 or Counter(r['domain'] for r in records)!={'SMD':512,'Sunspot':512}:
            raise ValueError('Existing1024 shared source CAL sample grid required')
        if keys is not None and locations!=keys:raise ValueError('Source CAL positions differ between methods')
        keys=locations
        replay=[apply_calibration(r,cal) for r in records]
        if not all(r['valid'] for r in replay):raise ValueError('Original source CAL has invalid required component')
        scores=np.array([r['score'] for r in replay]);tau=cal['threshold']
        # Verification of the ORIGINAL source quantile, not target fitting.
        replay_q99=weighted_quantile(scores,np.ones(1024)/1024,.99)
        if abs(replay_q99-tau)>1e-12:raise ValueError('Original source threshold replay exceeds numerical tolerance')
        source[method]=scores;records_by[method]=locations
        outputs.append(dict(method=method,origin='ORIGINAL_SOURCE_CAL_COMPONENTS_ARITHMETIC_REPLAY',**quantile_description(scores),source_threshold=tau,
            replayed_source_q99=replay_q99,source_q99_minus_original_tau=replay_q99-tau,
            source_tau_boundary_sensitive_points_1e12=int((np.abs(scores-tau)<=1e-12).sum()),
            numerical_boundary_note='Original tau unchanged; components replay may differ by ULP; no query canonical score is recomputed',
            strict_points_above_source_tau=int((scores>tau).sum()),strict_fraction_above_source_tau=float((scores>tau).mean()),
            confirmed3=None,confirmed3_NA_reason='Random sampled sparse CAL positions are not a continuous trajectory',
            healthy_source_certified=False,no_CAL_fit_performed=True))
    return source,records_by,outputs

def phase_loss_table(cfg):
    p=cfg['event_phase']['path']
    if sha(p)!=cfg['event_phase']['sha256']:raise PermissionError('Original fixed event phases changed')
    types=bound(cfg['official_type_join']);class_by={(e['entity'],e['event_id']):e['official_class'] for e in types['events']}
    base=Path(cfg['production_summary']['path']).parent;coverage={}
    for domain,entity in sorted({(r['domain'],r['entity']) for r in cfg['production_entities']}):
        with (base/'entities'/domain/entity/'HybridV1_Z_S'/'event_coverage.csv').open() as f:
            for e in csv.DictReader(f):coverage[domain,entity,e['source_event_id']]=e
    by={}
    with Path(p).open() as f:
        for r in csv.DictReader(f):
            k=tuple(r[x] for x in ('method','domain','entity','source_event_id','phase'))
            by.setdefault(k,{})[r['stream']]=r
    rows=[]
    for k,pair in sorted(by.items()):
        if set(pair)!={'strict_point','confirmed3_current'}:raise ValueError('Both original event streams required')
        a,b=pair['strict_point'],pair['confirmed3_current'];method,domain,entity,eid,phase=k
        meta=coverage[domain,entity,eid];strict=int(a['detected_points']);confirmed=int(b['detected_points'])
        if strict<confirmed:raise ValueError('Confirmed TP not a strict subset in a fixed phase')
        rows.append(dict(method=method,domain=domain,entity=entity,source_event_id=eid,phase=phase,
            official_type=class_by[entity,eid] if domain=='SMAP' else 'NOT_AVAILABLE',
            start=int(a['start']),end_exclusive=int(a['end_exclusive']),duration=int(a['duration']),duration_bin=a['duration_bin'],
            main_eligible=meta['main_eligible']=='True',boundary_censored=meta['boundary_censored']=='True',
            score_complete=meta['complete']=='True',phase_original_points=int(a['total_points']),
            scored_positive_points=int(a['scored_points']),unknown_points=int(a['unknown_points']),
            strict_TP=strict,confirmed_TP=confirmed,lost_TP=strict-confirmed,
            strict_any_observed_hit=bool(strict),confirmed_any_observed_hit=bool(confirmed),
            lost_observed_hit=bool(strict and not confirmed),
            strict_matching=a['one_to_one_matched'],confirmed_matching=b['one_to_one_matched'],
            strict_longest_miss=a['longest_observed_miss'],confirmed_longest_miss=b['longest_observed_miss']))
    return rows

def run(cfg):
    began=time.monotonic();scope=bound(cfg['scope']);summary=bound(cfg['production_summary']);production=bound(cfg['production_config'])
    receipt=bound(cfg['production_receipt'])
    if (sha(__file__)!=cfg['script_sha256'] or scope['methods']!=list(METHODS) or scope['new_score']
            or scope['training'] or scope['GPUh']!=0 or receipt.get('status')!='COMPLETED'
            or not receipt.get('all_raw_before_truth') or receipt.get('banks')!=162):
        raise PermissionError('Fixed post-production descriptive scope/source required')
    if cfg['source_scoring_sha256']!=sha(ROOT/'src/hybrid_tsad/hybrid_v1/scoring.py'):raise PermissionError('Original CAL arithmetic changed')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    arrays,bindings,accepted,normal_rows=load_original_arrays(cfg,production,summary)
    source,locations,cal_descriptions=source_calibration(cfg)
    source_npz={}
    for m,x in source.items():
        source_npz[m+'_score']=x;source_npz[m+'_identity']=np.array([json.dumps(k) for k in locations[m]])
    np.savez_compressed(out/'ORIGINAL_SOURCE_CAL_SCORES.npz',**source_npz)
    roc=[];cell_rows=[];distribution=[];runs=[];loss=[]
    old_dataset={(r['domain'],r['method']):r for r in summary['dataset']}
    for domain in ('SMAP','SMD'):
        for method in METHODS:
            group={k[2]:a for k,a in arrays.items() if k[:2]==(method,domain)}
            cells,decomposition=roc_pair_cells(group);old=old_dataset[domain,method]
            original=old['micro_point_ranking']['ROC_AUC']
            if abs(original-decomposition['total_ROC_AUC'])>2e-14:raise ValueError('Total original AUC weighted identity failed')
            perentity_errors=[]
            for cell in cells:
                if cell['relation']=='within_entity':
                    match=next(v for k,v in accepted.items() if k[:3]==(method,domain,cell['positive_entity']))
                    if cell['ROC_AUC'] is not None:
                        perentity_errors.append(abs(cell['ROC_AUC']-match['ranking']['ROC_AUC']))
            if max(perentity_errors,default=0)>2e-14:raise ValueError('Within original entity AUC differs')
            roc.append(dict(domain=domain,method=method,**decomposition,
                original_micro_ROC_AUC=original,original_macro_ROC_AUC=old['macro_entity_ranking']['ROC_AUC'],
                original_micro_AP=old['micro_point_ranking']['AP'],original_macro_AP=old['macro_entity_ranking']['AP'],
                original_micro_AUC_error=abs(original-decomposition['total_ROC_AUC']),
                max_original_entity_AUC_error=max(perentity_errors,default=0),
                limitation='Exact ROC pair identity, NOT an additive AP decomposition; dependence means no pairwise IID inference'))
            cell_rows.extend(dict(domain=domain,method=method,**r) for r in cells)
    for mk,a in sorted(arrays.items()):
        method,domain,entity,run_id=mk;score=a['score'];y=a['label'];tau=a['threshold']
        src=quantile_description(source[method])
        for label,name in [(None,'ALL'),(0,'OFFICIAL_NORMAL'),(1,'OFFICIAL_ANOMALY')]:
            x=score if label is None else score[y==label];q=quantile_description(x)
            distribution.append(dict(domain=domain,entity=entity,run=run_id,method=method,group=name,
                **q,source_threshold=tau,posthoc_fraction_above_fixed_source_tau=float((x>tau).mean()) if len(x) else None,
                descriptive_median_minus_source_CAL_median=q['descriptive_q050']-src['descriptive_q050'] if len(x) else None,
                descriptive_IQR_ratio_to_source_CAL=ratio(q['IQR'],src['IQR']) if len(x) else None,
                target_quantiles_may_not_be_used_as_thresholds=True))
        entity_runs=alarm_runs(a['index'],a['strict'],a['confirmed'],y)
        runs.extend(dict(method=method,domain=domain,entity=entity,run=run_id,**r) for r in entity_runs)
        strict=a['strict'];confirm=a['confirmed']
        lost=strict & ~confirm
        loss.append(dict(method=method,domain=domain,entity=entity,run=run_id,
            valid_points=len(y),positive_points=int((y==1).sum()),normal_points=int((y==0).sum()),
            strict_TP=int((strict&(y==1)).sum()),strict_FP=int((strict&(y==0)).sum()),
            confirmed_TP=int((confirm&(y==1)).sum()),confirmed_FP=int((confirm&(y==0)).sum()),
            lost_TP=int((lost&(y==1)).sum()),lost_FP=int((lost&(y==0)).sum()),
            lost_TP_fraction_of_strict_TP=ratio(int((lost&(y==1)).sum()),int((strict&(y==1)).sum())),
            lost_FP_fraction_of_strict_FP=ratio(int((lost&(y==0)).sum()),int((strict&(y==0)).sum())),
            strict_runs=len(entity_runs),runs_length1_or2=sum(r['length']<=2 for r in entity_runs)))
    phases=phase_loss_table(dict(cfg,production_entities=production['banks']))
    csvwrite(out/'ROC_PAIR_CELLS.csv',cell_rows);dump(out/'ROC_WITHIN_CROSS_DECOMPOSITION.json',roc)
    csvwrite(out/'SOURCE_CAL_SCORE_DESCRIPTIONS.csv',cal_descriptions)
    csvwrite(out/'TARGET_SCORE_DESCRIPTIONS.csv',distribution)
    csvwrite(out/'ORIGINAL_FIXED_TAU_NORMAL_ALARMS.csv',normal_rows)
    csvwrite(out/'STRICT_ALARM_RUNS.csv',runs);csvwrite(out/'STRICT_TO_CONFIRMED_ENTITY_LOSS.csv',loss)
    csvwrite(out/'ORIGINAL_EVENT_PHASE_CONFIRMATION_LOSS.csv',phases)
    # Descriptive aggregations preserve each entity/event identity above.
    aggregates=[]
    for domain in ('SMAP','SMD'):
        for method in METHODS:
            ls=[r for r in loss if (r['domain'],r['method'])==(domain,method)];rs=[r for r in runs if (r['domain'],r['method'])==(domain,method)]
            counts={k:sum(r[k] for r in ls) for k in ('valid_points','positive_points','normal_points','strict_TP','strict_FP','confirmed_TP','confirmed_FP','lost_TP','lost_FP')}
            groups=[]
            for phase in ('all','onset512','steady_after512'):
                for duration in ('ALL','<=32','33..512','>512'):
                    for kind in ('ALL','contextual','point') if domain=='SMAP' else ('ALL',):
                        for qualification in ('ALL','MAIN','BOUNDARY_COMPLETE','UNSCORED_OR_PARTIAL'):
                            selected=[r for r in phases if (r['domain'],r['method'],r['phase'])==(domain,method,phase)
                                and (duration=='ALL' or r['duration_bin']==duration) and (kind=='ALL' or r['official_type']==kind)
                                and (qualification=='ALL' or qualification=='MAIN' and r['main_eligible']
                                    or qualification=='BOUNDARY_COMPLETE' and r['boundary_censored'] and r['score_complete']
                                    or qualification=='UNSCORED_OR_PARTIAL' and not r['score_complete'])]
                            groups.append(dict(phase=phase,duration=duration,official_type=kind,qualification=qualification,
                                catalog_intervals=len(selected),intervals_with_scored_phase=sum(r['scored_positive_points']>0 for r in selected),
                                scored_positive_points=sum(r['scored_positive_points'] for r in selected),
                                strict_TP=sum(r['strict_TP'] for r in selected),confirmed_TP=sum(r['confirmed_TP'] for r in selected),
                                lost_TP=sum(r['lost_TP'] for r in selected),
                                lost_observed_hit_intervals=sum(r['lost_observed_hit'] for r in selected)))
            aggregates.append(dict(domain=domain,method=method,**counts,
                lost_TP_fraction_of_strict_TP=ratio(counts['lost_TP'],counts['strict_TP']),
                lost_FP_fraction_of_strict_FP=ratio(counts['lost_FP'],counts['strict_FP']),
                strict_runs=len(rs),short_run_count=sum(r['length']<=2 for r in rs),
                strict_TP_in_short_runs=sum(r['strict_TP'] for r in rs if r['length']<=2),
                strict_FP_in_short_runs=sum(r['strict_FP'] for r in rs if r['length']<=2),
                run_length_counts=dict(Counter(r['length_bin'] for r in rs)),event_phase_groups=groups))
    dump(out/'CONFIRMATION_AGGREGATES.json',aggregates);dump(out/'ORIGINAL_ARRAY_BINDINGS.json',bindings)
    dump(out/'CONFIG.json',cfg)
    dump(out/'RESULT.json',dict(status='COMPLETED_DESCRIPTIVE_PENDING_PRIMARY_REVIEW',banks=162,entities=54,
        methods=list(METHODS),new_score=False,new_threshold=False,new_AP_computation=False,
        new_model=False,new_forward=False,new_training=False,GPUh=0,wall_seconds=time.monotonic()-began,
        original_production_summary=cfg['production_summary'],source_CAL_reused=True,original_source_CAL_locations=1024,
        ROC=roc,confirmation=aggregates,limitations=scope['limitations']))
    return roc

def self_test():
    e={'a':dict(score=np.array([1.,2.]),label=np.array([1,0])),
       'b':dict(score=np.array([2.,2.,3.]),label=np.array([1,0,1]))}
    cells,s=roc_pair_cells(e);p=np.array([1.,2.,3.]);n=np.array([2.,2.]);brute=float(((p[:,None]>n).sum()+.5*(p[:,None]==n).sum())/6)
    assert s['total_ROC_AUC']==brute and s['within_entity']['pairs']==3 and s['cross_entity']['pairs']==3
    _,single=roc_pair_cells({'a':e['a']});assert single['cross_entity']['ROC_AUC'] is None
    runs=alarm_runs(np.array([1,2,4,5,6]),np.ones(5,bool),np.array([0,0,0,0,1],bool),np.array([1,0,1,1,0]))
    assert [r['length'] for r in runs]==[2,3] and sum(r['lost_TP'] for r in runs)==3
    print(json.dumps(dict(status='CPU_FIXTURES_PASS',fixtures=3,no_public_data_read=True)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--trusted-config-sha256');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:return self_test()
    if not a.config or sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Fixed diagnostic config SHA required')
    run(json.loads(Path(a.config).read_text()))

if __name__=='__main__':main()
