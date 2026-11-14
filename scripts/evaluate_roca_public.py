"""One fixed post-label consumer: RoCA/controls and explicit target-access views.

No neural model, fitting, threshold/score search, or repeat source-ZS evaluation.
All new banks and own-CAL identities must pass before the old full truth opens.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hybrid_tsad.hybrid_v1 import evaluation
from hybrid_tsad.hybrid_v1.public_native_evaluation import aggregate_dataset,event_diagnostics
from hybrid_tsad.hybrid_v1.chunked_bank import read_records
from hybrid_tsad.hybrid_v1.scoring import apply_calibration,weighted_quantile,recipe
from hybrid_tsad.hybrid_v1.controlled_wave_evaluation import require_metric_dependencies,reject_dependency_na
from hybrid_tsad.engine.official_affiliation import load_official_affiliation

MATRIX={'SMAP':['RoCA_FULL_T','RoCA_INIT_T','FIT_STATIC_T','History15_T','Persistence15_T'],
 'SMD':['RoCA_FULL_T','RoCA_INIT_T','FIT_STATIC_T','History31_T','Persistence31_T',
        'Hybrid_ZT','Hybrid_U200_T','History512_target_tau','Persistence512_target_tau']}
POLICY={
 'RoCA_FULL_T':('official_loss_n','target_FIT_full_RoCA_training','target_FIT_scaler_and_center'),
 'RoCA_INIT_T':('official_loss_n','no_optimizer_updates','target_FIT_scaler_and_initial_center'),
 'FIT_STATIC_T':('mean_squared_current_FIT_standardized','no_neural_training','target_FIT_scaler'),
 'History15_T':('raw_history_normalized_r_pred','no_neural_training','target_FIT_scaler;H15'),
 'Persistence15_T':('raw_history_normalized_r_pred','no_neural_training','target_FIT_scaler;H15'),
 'History31_T':('raw_history_normalized_r_pred','no_neural_training','target_FIT_scaler;H31'),
 'Persistence31_T':('raw_history_normalized_r_pred','no_neural_training','target_FIT_scaler;H31'),
 'Hybrid_ZT':('R3_own_target_CAL_scales_and_tau','frozen_source_pretrained_V1','no_target_FIT_updates'),
 'Hybrid_U200_T':('R3_own_target_CAL_scales_and_tau','source_V1_plus_target_FIT_200_updates','original_V1_normalization'),
 'History512_target_tau':('old_source_log1p_r_pred_score_new_target_tau','no_neural_training','original_scale;H512;no_RoCA_scaler'),
 'Persistence512_target_tau':('old_source_log1p_r_pred_score_new_target_tau','no_neural_training','original_scale;H512;no_RoCA_scaler')}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def bound(spec):
    if sha(spec['path'])!=spec['sha256']:raise PermissionError('Bound score/CAL/input changed: '+str(spec['path']))
    return json.loads(Path(spec['path']).read_text())
def table(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def calibration_value(spec):
    x=bound(spec)
    if 'select' in spec:
        rows=[r for r in x if all(r.get(k)==v for k,v in spec['select'].items())]
        if len(rows)!=1:raise PermissionError('One exact producer CAL selector required')
        x=rows[0]['calibration']
    return x

def validate_method(desc,cal,plan):
    name,domain=desc['method'],desc['domain']
    if name not in MATRIX[domain] or desc['score_semantics']!=POLICY[name][0]:
        raise PermissionError('Named method/score semantics do not match the fixed matrix')
    if desc.get('calibration_role')!='CAL' or not plan or any(r.get('role')!='CAL' or r.get('domain')!=domain for r in plan):
        raise PermissionError('QUERY/FIT/SELECT cannot calibrate the detector')
    if len(plan)!=(1696 if domain=='SMAP' else 512):raise PermissionError('Same fixed pooled CAL1696/512 required')
    if domain=='SMAP':
        from collections import Counter
        if len(Counter(r['entity'] for r in plan))!=53 or set(Counter(r['entity'] for r in plan).values())!={32}:
            raise PermissionError('SMAP32 per entity pooled, not per-entity thresholds')
    if not np.isfinite(cal['threshold']):raise PermissionError('Finite fixed own-CAL tau required')
    if name.startswith('RoCA') and cal.get('score')!='official_loss_n':
        raise PermissionError('Normality loss_n must not be relabeled R3/log1p')
    if name.startswith('Hybrid'):
        active,weights=recipe()
        if cal.get('active_components')!=list(active) or cal.get('weights')!=weights:
            raise PermissionError('Hybrid own-cal R3 recipe must remain intact')
    if name.endswith('_target_tau') and desc['format']!='original_components':
        raise PermissionError('Old H512 canonical bank must be reused, not a new prediction')

def read_scalar_bank(desc,cal,start,end):
    spec=desc['score_array']
    if sha(spec['path'])!=spec['sha256']:raise PermissionError('Canonical raw scalar array changed')
    with np.load(spec['path'],allow_pickle=False) as z:
        a={k:z[k].copy() for k in ('index','available_index','score','valid')}
        if 'alarm' in z.files and not np.array_equal(z['alarm'][a['valid']],a['score'][a['valid']]>cal['threshold']):
            raise ValueError('Archived own-CAL alarm differs; no floating query threshold')
    a['invalid_reasons']=[None if v else 'ORIGINAL_PRODUCER_INVALID' for v in a['valid']]
    if (not np.issubdtype(a['index'].dtype,np.integer) or not np.issubdtype(a['available_index'].dtype,np.integer)
            or not np.array_equal(a['index'],np.arange(start,end,dtype=np.int64)) or not np.array_equal(a['available_index'],a['index'])):
        raise ValueError('Native endpoint axis/arrived timing differs')
    return a,dict(format='canonical_scalar_npz',source_sha256=spec['sha256'],score_unchanged=True)

def read_component_bank(desc,cal,start,end):
    directory=Path(desc['bank_directory']);m=bound(dict(path=str(directory/'manifest.json'),sha256=desc['manifest_sha256']))
    old=calibration_value(desc['old_calibration'])
    if not m['complete'] or m['contract']['identity']!=desc['bank_identity'] or desc['old_calibration']['sha256']!=m['contract']['identity']['calibration_sha256']:
        raise PermissionError('Original complete raw component identity differs')
    tau_only=desc['method'].endswith('_target_tau')
    same_mapping=all(cal.get(k)==old.get(k) for k in ('scales','active_components','weights'))
    if tau_only and not same_mapping:
        raise PermissionError('H512 target-tau view may not change source scales or recipe')
    records=[];max_replay_error=0.
    for n,r in enumerate(read_records(directory)):
        if r['index']!=start+n or r['available_index']!=r['index'] or (r['domain'],r['entity'],r['run'])!=(desc['domain'],desc['entity'],desc['run']):
            raise ValueError('Original component point identity differs')
        s=apply_calibration(r,cal)
        if same_mapping and s['valid']:
            err=abs(float(r['score'])-s['score']);max_replay_error=max(max_replay_error,err)
            if err>1e-12:raise ValueError('Old source-score canonical mapping not reproducible')
            score=float(r['score'])  # preserve original tie ordering whenever the score map is unchanged
        else:score=s['score']
        records.append((r['index'],score,s['valid'],r['available_index'],s.get('reason')))
    if len(records)!=end-start:raise ValueError('Old complete bank misses query tail')
    return dict(index=np.array([r[0] for r in records],np.int64),score=np.array([r[1] for r in records],float),
        valid=np.array([r[2] for r in records],bool),available_index=np.array([r[3] for r in records],np.int64),
        invalid_reasons=[r[4] for r in records]),dict(format='old_components_fixed_new_CAL_replay',
            manifest_sha256=desc['manifest_sha256'],canonical_old_score_preserved=same_mapping,
            max_old_score_numeric_replay_error=max_replay_error,new_neural_forward=False)

def prelabel(cfg):
    old=bound(cfg['source_reference_summary']);production=bound(cfg['source_reference_config'])
    cohort={(d['domain'],d['entity'],d['run']):d for d in production['banks'] if d['method']=='HybridV1_Z_S'}
    expected={(method,)+k for k in cohort for method in MATRIX[k[0]]}
    desc={tuple(d[k] for k in ('method','domain','entity','run')):d for d in cfg['banks']}
    if len(cfg['banks'])!=274 or len(cohort)!=54 or set(desc)!=expected:
        raise PermissionError('Full fixed SMAP53x5 plus SMD1x9 matrix before truth required')
    arrays={};cals={};reports=[]
    # Every producer artifact and CAL plan is bound before original truth opens.
    for mk,d in sorted(desc.items()):
        cal=calibration_value(d['calibration'])
        if 'threshold_override' in d:
            if not d['method'].endswith('_target_tau'):raise PermissionError('Only named long source-unit target-tau override')
            cal=dict(cal,threshold=calibration_value(d['threshold_override'])['threshold'])
        plan=bound(d['CAL_plan']);validate_method(d,cal,plan)
        if d['format']=='scalar_npz' and 'CAL_scores' not in d:
            raise PermissionError('Every scalar method needs its original own-CAL score pool')
        if 'CAL_scores' in d:
            sp=d['CAL_scores']
            if sha(sp['path'])!=sp['sha256']:raise PermissionError('Original own-CAL score pool changed')
            with np.load(sp['path'],allow_pickle=False) as z:x=z['score'].astype(float)
            if x.shape!=(len(plan),) or not np.isfinite(x).all() or abs(weighted_quantile(x,np.ones(len(x))/len(x),.99)-cal['threshold'])>1e-12:
                raise ValueError('Own inverse-CDF q99 CAL score identity differs')
        for spec in d.get('provenance_assets',[]):
            if sha(spec['path'])!=spec['sha256']:raise PermissionError('FIT/scaler/checkpoint identity changed')
        start=512 if mk[1]=='SMAP' else 8958
        if mk[1]=='SMAP':
            # Authoritative native length, not a query-selected window.
            # root-bound copied original manifest metadata, supplied as an asset
            native=bound(d['original_axis_manifest']);end=native['contract']['ranges'][0]['end']
        else:end=23707
        if d['format']=='scalar_npz':a,report=read_scalar_bank(d,cal,start,end)
        elif d['format']=='original_components':a,report=read_component_bank(d,cal,start,end)
        else:raise PermissionError('No implicit score format or on-demand inference')
        if a['valid'].dtype!=np.bool_ or np.any(a['valid']&~np.isfinite(a['score'])):raise ValueError('Original valid score flags required')
        arrays[mk]=a;cals[mk]=cal;reports.append(dict(method=mk[0],domain=mk[1],entity=mk[2],points=len(a['index']),
            score_semantics=d['score_semantics'],train_access=POLICY[mk[0]][1],FIT_access=POLICY[mk[0]][2],
            CAL_access='same approved target nominal CAL1696/512; no QUERY fitting',**report))
    return cohort,desc,arrays,cals,reports,old

def run(cfg):
    begin=time.monotonic();require_metric_dependencies()
    if sha(__file__)!=cfg['script_sha256']:raise PermissionError('Frozen consumer source')
    for p,h in cfg['source_sha256'].items():
        if sha(p)!=h:raise PermissionError('Accepted metric/score source changed')
    if sha(cfg['affiliation_identity']['path'])!=cfg['affiliation_identity']['sha256']:raise PermissionError('Official Aff source identity changed')
    load_official_affiliation(cfg['affiliation_identity']['path'])
    cohort,desc,arrays,cals,reports,old=prelabel(cfg)
    types=bound(cfg['official_type_join'])
    type_by={(e['entity'],e['event_id']):e['official_class'] for e in types['events']}
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    dump(out/'ALL_NEW_RAW_PRELABEL_GATES.json',dict(status='ALL274_NEW_SCORE_VIEWS_SEALED',labels_opened=False,reports=reports))
    # Sole truth opener: reuse old accepted full original labels, not a new variant.
    olddir=Path(cfg['source_reference_summary']['path']).parent;labels={};catalog={}
    for k in sorted(cohort):
        domain,entity,_=k;directory=olddir/'entities'/domain/entity/'HybridV1_Z_S';calc=directory/'calculation_inputs.npz'
        prov=json.loads((directory/'provenance.json').read_text())
        if sha(calc)!=prov['calculation_inputs_sha256']:raise PermissionError('Original full truth inputs changed')
        with np.load(calc,allow_pickle=False) as z:labels[k]=z['full_labels'].copy()
        with (directory/'event_coverage.csv').open() as f:catalog[k]=list(csv.DictReader(f))
    results={};rows=[];phase=[]
    for mk,a in sorted(arrays.items()):
        method,domain,entity,run_id=mk;y=labels[mk[1:]]
        result=evaluation.evaluate_arrays(**a,full_labels=y,threshold=cals[mk]['threshold'],bank_complete=True,
            original_length=len(y),original_start=0,warmup_points=512 if domain=='SMAP' else 8958,
            affiliation_identity=cfg['affiliation_identity']['path'])
        reject_dependency_na(result)
        ids={(int(e['start']),int(e['end_exclusive'])):e['source_event_id'] for e in catalog[mk[1:]]}
        for event in result['event_coverage']:event['source_event_id']=ids[event['start'],event['end_exclusive']]
        evaluation.save_evaluation(out/'entities'/domain/entity/method,result,**a,full_labels=y,
            provenance=dict(descriptor=desc[mk],reference_truth_summary=cfg['source_reference_summary'],source_sha256=cfg['source_sha256']))
        results[mk]=result;phase.extend(dict(method=method,domain=domain,entity=entity,run=run_id,
            official_type=type_by[entity,r['source_event_id']] if domain=='SMAP' else 'NOT_AVAILABLE',**r) for r in event_diagnostics(result))
        rows.append(dict(method=method,domain=domain,entity=entity,run=run_id,score_semantics=POLICY[method][0],
            training_access=POLICY[method][1],FIT_access=POLICY[method][2],ranking=result['ranking'],coverage=result['coverage'],
            streams={s:dict(point=v['point'],events=v['main_events']['metrics'],legacy_complete_only_events=v['legacy_complete_only_events']['metrics'],
                normal=v['normal'],affiliation=v['affiliation']) for s,v in result['streams'].items()}))
    dataset=[aggregate_dataset(results,arrays,labels,domain,method) for domain in MATRIX for method in MATRIX[domain]]
    table(out/'event_duration_phase_diagnostics.csv',phase)
    dump(out/'SOURCE_ZS_REFERENCES_NOT_REEVALUATED.json',dict(original_summary=cfg['source_reference_summary'],dataset=old['dataset'],
        permissions='Original source-pretrained/source-CAL track, not target-FIT matched; no reevaluation'))
    dump(out/'SUMMARY.json',dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT',per_entity=rows,dataset=dataset,
        fixed_matrix=MATRIX,score_semantics=POLICY,statistics=dict(seed=2025,CI=None,independent_confirmation=False),
        cost_provenance=cfg.get('resource_receipts',[]),
        limitations=['SMAP target-pooled RoCA vs source-ZS cannot isolate normality objective',
            'SMD target access still differs in architecture/pretraining; full-init and StaticFIT constrain training increment',
            'Official contextual/point types are descriptive, not proof of weak-amplitude or directed lag mechanisms',
            'Source ZS references reused; H512 target tau differs from Hybrid full R3 CAL scale+tau changes']))
    dump(out/'CONFIG.json',cfg)
    dump(out/'RECEIPT.json',dict(status='COMPLETED',new_score_views=274,entities=54,GPUh=0,new_forward=False,new_training=False,
        source_ZS_metrics_reused=True,all_new_raw_before_truth=True,seconds=time.monotonic()-begin))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Root-bound fixed evaluation config')
    run(json.loads(Path(a.config).read_text()))

if __name__=='__main__':main()
