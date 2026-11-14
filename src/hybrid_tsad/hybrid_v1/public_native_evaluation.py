"""Fixed native-public CPU evaluation; no model, fit, or score selection.

Every one of the three methods' 54 raw banks is authenticated and replayed
before the sole numerical truth opener. Event mathematics stays in evaluation.
Dataset point ranking may pool points; confirmation and event matching never do.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import evaluation
from .chunked_bank import read_records
from .controlled_wave_evaluation import scalar_record, require_metric_dependencies, reject_dependency_na
from .public_native_zs import SOURCE_PT, SOURCE_CAL, BASELINE_CAL, native_identity, ranges
from .stage2_u_evaluation import long_miss
from .stage2_u_matched import load_roles
from ..engine.events import binary_events
from ..engine.evaluation import evaluate_point_ranking
from ..engine.official_affiliation import load_official_affiliation

METHODS=('HybridV1_Z_S','persistence','history_mean512')
SCOPE_SHA='1794b6dc2df31574a8e8eb2e158d7e04b8d82421ab02388bb3e291bc2e78788d'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def bound(asset):
    if sha(asset['path'])!=asset['sha256']:raise PermissionError('Frozen asset changed: '+str(asset['path']))
    return json.loads(Path(asset['path']).read_text(encoding='utf-8'))
def ratio(a,b):return float(a/b) if b else None
def key(run):return (run.domain,run.entity,run.run)
def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def scope_entries(scope):
    return scope['cohort']['SMAP']['entry_identities']+[scope['cohort']['SMD']['entry_identity']]

def prelabel(cfg):
    """No truth deserialization; pin complete scope/inputs/CAL/raw scores first."""
    scope=bound(cfg['scope'])
    if (cfg['scope']['sha256']!=SCOPE_SHA or scope['models']!=list(METHODS)
            or scope['source_checkpoint_sha256']!=SOURCE_PT or scope['source_CAL_sha256']!=SOURCE_CAL
            or scope['source_seed']!=2025 or scope['inference_seed']!=2025):
        raise PermissionError('Original first-source-seed scope required')
    for path,h in cfg['source_sha256'].items():
        if sha(path)!=h:raise PermissionError('Scientific CPU source changed')
    bound(cfg['query_manifest'])
    runs=load_roles(cfg['query_manifest']['path'],{'QUERY'})
    expected={tuple(e[k] for k in ('domain','entity','run')):e for e in scope_entries(scope)}
    by={key(r):r for r in runs}
    if len(runs)!=54 or set(by)!=set(expected):raise PermissionError('Exactly all54 native entities required')
    for k,r in by.items():
        e=expected[k]
        if (r.start!=e['original_start'] or list(r.values.shape)!=e['expected_shape']
                or r.role!='QUERY'):
            raise PermissionError('Native identity/shape/start changed')
    manifest_entries=json.loads(Path(cfg['query_manifest']['path']).read_text())['runs']
    for e in manifest_entries:
        k=tuple(e[x] for x in ('domain','entity','run'))
        if e['sha256']!=expected[k]['sha256']:raise PermissionError('Native values file changed')
    cal={m:bound(cfg['calibrations'][m]) for m in METHODS}
    hashes=dict(BASELINE_CAL,HybridV1_Z_S=SOURCE_CAL)
    if any(cfg['calibrations'][m]['sha256']!=hashes[m] for m in METHODS):
        raise PermissionError('Own original source CAL required, never target CAL')
    desc={(d['method'],d['domain'],d['entity'],d['run']):d for d in cfg['banks']}
    universe={(m,)+k for m in METHODS for k in by}
    if len(cfg['banks'])!=162 or set(desc)!=universe:raise PermissionError('All3 methods x54 banks must seal before truth')
    # ALL manifest/identity gates precede even the first raw scalar replay.
    metas={}
    for mk,d in desc.items():
        method=mk[0];r=by[mk[1:]]
        p=Path(d['directory'])/'manifest.json'
        if sha(p)!=d['manifest_sha256']:raise PermissionError('Sealed bank changed')
        m=json.loads(p.read_text());identity=m['contract']['identity']
        if (not m.get('complete') or m['committed_records']!=len(r.values)-512
                or m['contract']['ranges']!=ranges(r) or identity!=d['identity']
                or identity['input_sha256']!=native_identity(r)
                or identity['calibration_sha256']!=hashes[method]
                or m['contract']['threshold']!=cal[method]['threshold']):
            raise PermissionError('Full native input/source threshold/identity incomplete')
        if method=='HybridV1_Z_S' and identity['checkpoint_sha256']!=SOURCE_PT:
            raise PermissionError('Not original source HybridV1')
        metas[mk]=m
    arrays={};reports=[]
    for mk in sorted(desc):
        d=desc[mk];r=by[mk[1:]];rows=[]
        for n,record in enumerate(read_records(d['directory'])):
            t=r.start+512+n
            if (tuple(record[x] for x in ('domain','entity','run'))!=mk[1:]
                    or record.get('index')!=t or record.get('available_index')!=t):
                raise ValueError('Raw original index/available clock differs')
            rows.append(scalar_record(record,cal[mk[0]]))
        if len(rows)!=len(r.values)-512:raise ValueError('Missing raw tail')
        a=dict(index=np.array([x[0] for x in rows],np.int64),score=np.array([x[1] for x in rows],float),
            valid=np.array([x[2] for x in rows],bool),available_index=np.array([x[3] for x in rows],np.int64),
            invalid_reasons=[x[4] for x in rows])
        arrays[mk]=a
        reports.append(dict(method=mk[0],domain=mk[1],entity=mk[2],run=mk[3],records=len(rows),
            valid_points=int(a['valid'].sum()),manifest_sha256=d['manifest_sha256'],
            all_source_scores_replayed=True,input_sha256=metas[mk]['contract']['identity']['input_sha256']))
    # Same raw point clock is required; actual invalid scores remain unknown.
    for k in by:
        base=arrays[(METHODS[0],)+k]
        for method in METHODS[1:]:
            a=arrays[(method,)+k]
            if not np.array_equal(a['index'],base['index']):
                raise ValueError('Methods have unequal raw native scoring coverage')
    return scope,by,desc,cal,arrays,reports

def load_truth(cfg,runs):
    """Only caller run() invokes this after all162 raw banks pass."""
    specs=cfg['truth'];smap=bound(specs['SMAP']);out={}
    for k,r in runs.items():
        if k[0]=='SMAP':
            y=np.zeros(len(r.values),np.int8)
            for e in smap['events']:
                if e['entity']==r.entity:
                    a,b=e['native_half_open'];y[a:b]=1
            out[k]=y
        else:
            spec=specs['SMD']
            if sha(spec['path'])!=spec['sha256']:raise PermissionError('Official full SMD labels changed')
            y=np.loadtxt(spec['path'],delimiter=',')
            if y.shape!=(23707,) or not np.isin(y,[0,1]).all():raise ValueError('Original SMD test axis required')
            out[k]=y.astype(np.int8)
    if (sum(len(binary_events(y)) for k,y in out.items() if k[0]=='SMAP')!=67
            or sum(len(binary_events(y)) for k,y in out.items() if k[0]=='SMD')!=12):
        raise ValueError('Frozen67 SMAP/12 original SMD event catalogue differs; no boundary repair')
    return out

def duration_bin(length):
    return '<=32' if length<=32 else '33..512' if length<=512 else '>512'

def event_diagnostics(result):
    """Bookkeeping on original per-run alarms; do not restart confirmation per phase."""
    rows=[];points=result['point_coverage']
    misses={(r['event_id'],r['stream']):r for r in long_miss(result)}
    for event in result['event_coverage']:
        lo,hi=event['local_start'],event['local_end'];length=hi-lo
        for stream in ('strict_point','confirmed3_current'):
            em=next(e for e in result['streams'][stream]['main_events']['events'] if e['event_id']==event['event_id'])
            common=dict(event_id=event['event_id'],source_event_id=event.get('source_event_id'),start=event['start'],end_exclusive=event['end_exclusive'],
                duration=length,duration_bin=duration_bin(length),stream=stream,
                eligible=event['main_eligible'],coverage_ratio=event['coverage_ratio'],
                any_hit=em['any_hit'],one_to_one_matched=em['one_to_one_matched'],delay_points=em['delay_points'],
                right_censored=em['delay_right_censored'],
                event_longest_miss=misses[(event['event_id'],stream)]['longest_observed_consecutive_miss_points'])
            for phase,a,b in [('all',lo,hi),('onset512',lo,min(hi,lo+512)),('steady_after512',min(hi,lo+512),hi)]:
                selected=points[a:b];valid=[p for p in selected if p['valid']]
                missed=[p['valid'] and not bool(p[stream]) for p in selected]
                spans=binary_events(missed)
                row=dict(common,phase=phase,total_points=b-a,scored_points=len(valid),unknown_points=b-a-len(valid),
                    detected_points=sum(bool(p[stream]) for p in valid),
                    point_recall=ratio(sum(bool(p[stream]) for p in valid),len(valid)),
                    longest_observed_miss=max((z-s for s,z in spans),default=0) if valid else None,
                    longest_miss_lower_bound_only=bool(b-a-len(valid)))
                rows.append(row)
    return rows

def aggregate_dataset(results,arrays,labels,domain,method):
    """Only point ranking is pooled; event/confirmed counts are already native."""
    keys=sorted(k for k in results if k[0]==method and k[1]==domain)
    r=[results[k] for k in keys];ys=[];ss=[]
    for k in keys:
        a=arrays[k];valid=a['valid'];ys.append(labels[k[1:]][a['index'][valid]]);ss.append(a['score'][valid])
    y=np.concatenate(ys);s=np.concatenate(ss)
    if len(y):
        micro=evaluate_point_ranking(np.arange(len(y),dtype=np.int64),s,y,np.ones(len(y),bool),query_start=0)['metrics']
    else:
        micro=dict(evaluated_points=0,positive_points=0,negative_points=0,positive_fraction=None,
            AP=None,ROC_AUC=None,trapezoid_PR_AUC=None,AP_divided_by_pi=None,
            ranking_metric_NA_reason='NO_VALID_SCORES')
    # The helper's synthetic pooled event geometry is deliberately not reported.
    micro={k:v for k,v in micro.items() if 'event' not in k}
    original_points=sum(x['coverage']['original_length'] for x in r)
    raw_points=sum(x['coverage']['declared_raw_records'] for x in r)
    # The ranking helper sees a compact valid-point pool, not native geometry.
    # Override its synthetic unscored=0 with the separately preserved full axes.
    micro.update(original_length_total=original_points,declared_raw_points=raw_points,
        invalid_raw_points=raw_points-len(y),unscored_points=original_points-len(y),
        unscored_original_points=original_points-len(y),
        non_raw_original_points=original_points-raw_points,
        coverage_definition='Full native original axes; ranking pools only actual valid scores')
    ranks=('AP','ROC_AUC','trapezoid_PR_AUC','AP_divided_by_pi')
    macro={}
    for metric in ranks:
        values=[x['ranking'][metric] for x in r if x['ranking'].get(metric) is not None]
        macro[metric]=dict(mean=float(np.mean(values)) if values else None,entities=len(values),total_entities=len(r))
    streams={}
    for stream in ('strict_point','confirmed3_current'):
        pts={n:sum(x['streams'][stream]['point'][n] for x in r) for n in ('TP','FP','TN','FN')}
        tp,fp,tn,fn=[pts[n] for n in ('TP','FP','TN','FN')]
        pts.update(Precision=ratio(tp,tp+fp),Recall=ratio(tp,tp+fn),F1=ratio(2*tp,2*tp+fp+fn))
        ev={n:sum(x['streams'][stream]['main_events']['metrics'][n] for x in r) for n in
            ('M','Npred','Ntrue','auxiliary_any_hit_events','delay_right_censored_events')}
        ev.update(one_to_one_eventPrecision=ratio(ev['M'],ev['Npred']),one_to_one_eventRecall=ratio(ev['M'],ev['Ntrue']),
            one_to_one_eventF1=ratio(2*ev['M'],ev['Npred']+ev['Ntrue']),
            auxiliary_any_hit_recall=ratio(ev['auxiliary_any_hit_events'],ev['Ntrue']))
        legacy={n:sum(x['streams'][stream]['legacy_complete_only_events']['metrics'][n] for x in r) for n in
            ('M','Npred','Ntrue','auxiliary_any_hit_events','delay_right_censored_events')}
        legacy.update(one_to_one_eventPrecision=ratio(legacy['M'],legacy['Npred']),
            one_to_one_eventRecall=ratio(legacy['M'],legacy['Ntrue']),
            one_to_one_eventF1=ratio(2*legacy['M'],legacy['Npred']+legacy['Ntrue']),
            auxiliary_any_hit_recall=ratio(legacy['auxiliary_any_hit_events'],legacy['Ntrue']))
        normal={n:sum(x['streams'][stream]['normal'][n] for x in r) for n in
            ('normal_exposure_points','normal_false_alarm_points','normal_alarm_fragments')}
        normal.update(normal_FPR=ratio(normal['normal_false_alarm_points'],normal['normal_exposure_points']),
            normal_alarm_fragments_per_1000_points=ratio(1000*normal['normal_alarm_fragments'],normal['normal_exposure_points']))
        aff=[x['streams'][stream]['affiliation']['global_result']['Aff_F1'] for x in r
             if x['streams'][stream]['affiliation']['global_result']['Aff_F1'] is not None]
        delays=[e['delay_points'] for x in r for e in x['streams'][stream]['main_events']['events']
                if e['eligible'] and e['delay_points'] is not None]
        streams[stream]=dict(point=pts,events=ev,legacy_complete_only_events=legacy,normal=normal,
            detected_event_delay_points=dict(mean=float(np.mean(delays)) if delays else None,
                median=float(np.median(delays)) if delays else None,detected_events=len(delays),
                missed_events_right_censored=ev['delay_right_censored_events'],units='original_sample_points'),
            official_Aff_F1_entity_macro=dict(mean=float(np.mean(aff)) if aff else None,entities=len(aff),total_entities=len(r)),
            global_Aff=None,global_Aff_NA_reason='NO_INVENTED_CROSS_ENTITY_EVENT_GEOMETRY')
    return dict(domain=domain,method=method,entities=len(r),micro_point_ranking=micro,macro_entity_ranking=macro,streams=streams,
        event_rule='Per-native-run confirmation and matching before summing counts; no concatenated events',
        statistical_confirmation=False,CI=None,seed=2025)

def run(cfg):
    start=time.monotonic();require_metric_dependencies()
    if sha(cfg['affiliation_identity']['path'])!=cfg['affiliation_identity']['sha256']:
        raise PermissionError('Official affiliation binding changed')
    load_official_affiliation(cfg['affiliation_identity']['path'])
    scope,runs,desc,cal,arrays,reports=prelabel(cfg)
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    dump(out/'ALL_RAW_PRELABEL_CHECKS.json',dict(status='ALL162_RAW_BANKS_COMPLETE_AND_SCORES_REPLAYED',
        reports=reports,labels_opened=False,scope_sha256=cfg['scope']['sha256']))
    labels=load_truth(cfg,runs)  # sole numerical opener, after ALL raw checks above
    smap_catalog=bound(cfg['truth']['SMAP'])['events']
    source_ids={(e['entity'],tuple(e['native_half_open'])):e['event_id'] for e in smap_catalog}
    results={};events=[];summary=[]
    for mk in sorted(arrays):
        method,domain,entity,run_id=mk;a=arrays[mk];r=runs[mk[1:]];y=labels[mk[1:]]
        warmup=512 if domain=='SMAP' else 8958
        result=evaluation.evaluate_arrays(full_labels=y,threshold=cal[method]['threshold'],bank_complete=True,
            original_length=len(y),original_start=0,warmup_points=warmup,
            affiliation_identity=cfg['affiliation_identity']['path'],**a)
        reject_dependency_na(result)
        if result['coverage']['main_eligible_events']!=(sum(e['main_eligible'] for e in result['event_coverage'])):
            raise ValueError('Event coverage disagreement')
        result['original_stream']=dict(domain=domain,entity=entity,run=run_id)
        if domain=='SMD':
            result['coverage']['unscored_prefix_interpretation']='[0,8446) outside frozen QUERY input role; [8446,8958) QUERY history warmup; all unknown, never normal'
        for number,event in enumerate(result['event_coverage']):
            event['source_event_id']=(source_ids[(entity,(event['start'],event['end_exclusive']))]
                if domain=='SMAP' else entity+':E%03d'%number)
        evaluation.save_evaluation(out/'entities'/domain/entity/method,result,full_labels=y,
            provenance=dict(bank=desc[mk],calibration=cfg['calibrations'][method],truth=cfg['truth'][domain],
                source_sha256=cfg['source_sha256'],scope=cfg['scope']),**a)
        diagnostics=event_diagnostics(result)
        events.extend(dict(method=method,domain=domain,entity=entity,run=run_id,**x) for x in diagnostics)
        results[mk]=result
        summary.append(dict(method=method,domain=domain,entity=entity,run=run_id,ranking=result['ranking'],
            coverage=result['coverage'],streams={k:dict(point=v['point'],events=v['main_events']['metrics'],
                legacy_complete_only_events=v['legacy_complete_only_events']['metrics'],
                normal=v['normal'],affiliation=v['affiliation']) for k,v in result['streams'].items()}))
    aggregates=[aggregate_dataset(results,arrays,labels,d,m) for d in ('SMAP','SMD') for m in METHODS]
    coverage_reports=[]
    for m in METHODS:
        for domain,n in [('SMAP',67),('SMD',10)]:
            relevant=[e for k,r in results.items() if k[:2]==(m,domain) for e in r['event_coverage']]
            complete=sum(e['complete'] for e in relevant)
            main=sum(e['main_eligible'] for e in relevant)
            if complete>n:raise ValueError('Complete event count exceeds frozen native geometry')
            coverage_reports.append(dict(method=m,domain=domain,geometrically_complete_events=n,
                original_catalog_intervals=len(relevant),actually_valid_complete_events=complete,
                main_uncensored_eligible_events=main,
                complete_boundary_censored_events=sum(e['complete'] and e['boundary_censored'] for e in relevant),
                original_run_end_censored_events=sum(e['touches_original_end'] for e in relevant),
                actual_invalid_score_coverage_gap=n-complete,
                status='COMPLETE_SCORE_COVERAGE_BOUNDARY_POLICY_RETAINED' if complete==n else 'INCOMPLETE_VALID_SCORE_COVERAGE_UNKNOWN_RETAINED'))
    table(out/'event_duration_phase_diagnostics.csv',events)
    table(out/'entity_ranking.csv',[dict(method=x['method'],domain=x['domain'],entity=x['entity'],run=x['run'],**x['ranking']) for x in summary])
    dump(out/'SUMMARY.json',dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT',per_entity=summary,dataset=aggregates,
        event_coverage_reports=coverage_reports,
        metadata_correction='Frozen scope67 SMAP complete intervals is score-complete geometry, not main-event eligibility: original run-end censor policy retains10 separately and main57. No interval/score/method changed.',
        score_selection=False,threshold_selection=False,source_CAL_only=True,
        statistics=dict(seeds=1,CI=None,scope='REAL_PUBLIC_EXPLORATION_NOT_MULTI_SEED_CONFIRMATION'),
        historical_native_results='SEPARATE_INCOMPATIBLE_TRACK_NOT_IN_MAIN_TABLE'))
    dump(out/'CONFIG.json',cfg)
    dump(out/'RECEIPT.json',dict(status='COMPLETED',banks=162,entities=54,methods=list(METHODS),
        scope_sha256=cfg['scope']['sha256'],wall_seconds=time.monotonic()-start,GPUh=0,
        no_model_no_fit=True,all_raw_before_truth=True))
    return aggregates

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Fixed evaluation config SHA required')
    run(json.loads(Path(a.config).read_text(encoding='utf-8')))

if __name__=='__main__':main()
