"""Fixed PSM plain-U ABC evaluation glue; no models or fitted score recipes."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np

from . import evaluation, scoring
from .pretrained_psm_comparison import sha, bound, read, write, _field, _scalar
from .public_native_evaluation import event_diagnostics
from .controlled_wave_evaluation import require_metric_dependencies, reject_dependency_na

SOURCE='c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812'
OWN=('Z_T','U_A','U_B','U_C')
PAIRS=(('Z_T','U_A'),('U_A','U_C'),('U_C','U_B'),
       ('U_A_ZT_FIXED','U_C_ZT_FIXED'),('U_C_ZT_FIXED','U_B_ZT_FIXED'))

def query_index(plan):
    if (plan.get('history'),plan.get('horizon'),plan.get('seed'))!=(512,32,2025):
        raise PermissionError('Original H512/P32/seed required')
    index=np.array([e['index'] for e in plan['QUERY']],np.int64)
    if len(index)!=14792 or not np.all(np.diff(index)>0):
        raise ValueError('Frozen complete original query index, no score-based mask')
    for e in plan['QUERY']:
        if (e['domain'],e['entity'],e['run'],e['role'])!=('PSM','anonymous_PSM','test_original','QUERY') or e['footprint']!=[e['index']-512,e['index']+1]:
            raise PermissionError('Whole original QUERY run and arrived footprint required')
    return index

def index_ranges(index):
    cut=np.r_[0,np.flatnonzero(np.diff(index)!=1)+1,len(index)]
    return [dict(domain='PSM',entity='anonymous_PSM',run='test_original',start=int(index[a]),end=int(index[b-1])+1) for a,b in zip(cut[:-1],cut[1:])]

def validate_calibration(cal):
    active,weights=scoring.recipe()
    if (cal.get('active_components')!=list(active) or cal.get('weights')!=weights
            or cal.get('valid_windows')!=512 or cal.get('source_domain_equal_weight') is not False
            or set(cal.get('scales',{}))!=set(active)
            or not all(np.isfinite(v) and v>0 for v in cal['scales'].values())
            or not np.isfinite(cal.get('threshold',np.nan))):
        raise PermissionError('Original target CAL512/h1/R3 recipe required')

def score_components(components,calibration):
    """Delegates all numerical score/validity rules to the original scorer."""
    values=[];valid=[]
    for row in components:
        scored=scoring.apply_calibration(dict(zip(scoring.COMPONENTS,row)),calibration)
        values.append(scored['score']);valid.append(scored['valid'])
    return np.array(values,float),np.array(valid,bool)

def crop(index,wanted):
    at=np.searchsorted(index,wanted)
    if np.any(at>=len(index)) or not np.array_equal(index[at],wanted):
        raise ValueError('Exact original scored indexes required; no interpolation/concatenation')
    return at

def admit_new_banks(config,index):
    seal=read(config['whole_raw_seal'])
    if (seal.get('approved_by')!='root' or seal.get('status')!='ACCEPTED_PSM_U_ABC_ALL_RAW'
            or seal.get('banks')!=config['new_banks'] or seal.get('query_plan')!=config['query_plan']):
        raise PermissionError('All three new raw banks need one actual root seal')
    if [d.get('condition') for d in config['new_banks']]!=['A','B','C']:
        raise ValueError('All ABC arms, no best-arm selection')
    calibrations={};common_plan=read(config['target_plan'])['CAL']
    if len(common_plan)!=512 or any(r['role']!='CAL' for r in common_plan):raise PermissionError('Fixed CAL512 plan')
    for d in config['new_banks']:
        if read(d['plan'])['CAL']!=common_plan:raise PermissionError('Identical original CAL positions/order required')
        train,cal,query=[read(d[k]) for k in ('training_receipt','calibration_receipt','query_receipt')]
        for r,phase in [(train,'train'),(cal,'cal'),(query,'query')]:
            if (r.get('phase')!=phase or r.get('status')!='COMPLETED' or r.get('condition')!=d['condition']
                    or r.get('seed')!=2025 or r.get('parent_source_checkpoint_sha256')!=SOURCE or r.get('labels_read') is not False):
                raise PermissionError('Actual matched-U phase/arm/source/permission chain')
        if train.get('scientific_training_status')!='VALID_ALL_ATTEMPTS_UPDATED' or train.get('updates')!=200 or train.get('attempts')!=200:
            raise ValueError('Only completed200 fixed endpoint updates')
        if cal.get('records')!=512 or query.get('points')!=len(index) or query.get('state_unchanged') is not True:
            raise ValueError('CAL512/whole QUERY/unchanged readout required')
        if train['checkpoint_sha256']!=cal['checkpoint_sha256'] or train['checkpoint_sha256']!=query['checkpoint_sha256']:
            raise PermissionError('Same arm trained checkpoint chain')
        c=read(d['calibration']);validate_calibration(c)
        if cal['calibration_sha256']!=query['calibration_sha256'] or cal['calibration_sha256']!=d['calibration']['sha256']:
            raise PermissionError('Own frozen calibration chain')
        m=read(d['manifest']);identity=m['contract']['identity']
        if (not m['complete'] or m['committed_records']!=len(index) or m['contract']['ranges']!=index_ranges(index)
                or query['bank_manifest_sha256']!=d['manifest']['sha256']
                or identity!=d['expected_identity'] or identity['checkpoint_sha256']!=train['checkpoint_sha256']
                or identity['calibration_sha256']!=d['calibration']['sha256'] or m['contract']['threshold']!=c['threshold']):
            raise PermissionError('Exact actual bank five identities/ranges/own threshold')
        calibrations['U_'+d['condition']]=c
    return calibrations

def raw_bank_scalars(descriptor,index,calibration):
    manifest=read(descriptor['manifest']);directory=bound(descriptor['manifest']).parent
    keys=('domain','entity','run','index','available_index','score','valid','r_tf','r_ms','r_pred')
    values={k:[] for k in keys[3:]};position=0;maxerr=0
    for chunk in manifest['chunks']:
        p=directory/chunk['file']
        if p.parent!=directory or sha(p)!=chunk['sha256'] or chunk['start_position']!=position:
            raise ValueError('Unchanged bounded raw chunks required')
        with np.load(p,allow_pickle=False) as a:
            trees=json.loads(a['tree_utf8'].tobytes().decode('utf-8'))
            pools={k:a[k] for k in a.files if k!='tree_utf8'}
        if len(trees)!=chunk['records']:raise ValueError('Chunk count')
        for tree in trees:
            node=_field(tree,'raw');r={k:_scalar(_field(node,k),pools) for k in keys}
            if position>=len(index) or (r['domain'],r['entity'],r['run'],r['index'],r['available_index'])!=('PSM','anonymous_PSM','test_original',int(index[position]),int(index[position])):
                raise ValueError('Original native index/current availability changed')
            replay=scoring.apply_calibration(r,calibration)
            if bool(r['valid'])!=replay['valid']:raise ValueError('Original valid/NA mismatch')
            if replay['valid']:
                if not np.isclose(r['score'],replay['score'],rtol=1e-12,atol=1e-12):raise ValueError('R3 verification failed')
                maxerr=max(maxerr,abs(r['score']-replay['score']))
            for k in values:values[k].append(r[k])
            position+=1
    if position!=len(index):raise ValueError('Incomplete raw bank')
    result={k:np.asarray(v,dtype=bool if k=='valid' else np.int64 if k in ('index','available_index') else np.float64) for k,v in values.items()};result['R3max']=maxerr
    return result

def save_arrays(output,name,index,score,valid):
    p=output/(name+'_scores.npz')
    with p.open('xb') as f:np.savez_compressed(f,index=index,score=score,valid=valid,available_index=index)
    return dict(path=str(p),sha256=sha(p))

def complete_steady_all_missed(steady):
    """UNKNOWN-only/partial events are not observed full-event misses."""
    return sum(bool(q['eligible']) and q['scored_points']>0 and q['unknown_points']==0
               and q['detected_points']==0 for q in steady)

def run(config):
    began=time.monotonic()
    if config.get('status')!='ROOT_BOUND' or config.get('seed')!=2025:raise PermissionError('Pending template not executable')
    for p,h in config['source_sha256'].items():
        if sha(p)!=h:raise PermissionError('Frozen shared evaluator changed')
    require_metric_dependencies()
    index=query_index(read(config['query_plan']))
    calibrations=admit_new_banks(config,index)
    zcal=read(config['ZT_calibration']);validate_calibration(zcal);calibrations['Z_T']=zcal
    lineage=read(config['ZT_lineage_reference'])['ZT_old_CAL_reuse']
    if lineage['source_checkpoint_sha256']!=SOURCE or lineage['calibration_sha256']!=config['ZT_calibration']['sha256'] or lineage['index_order_exact'] is not True:
        raise PermissionError('Accepted c00 same-CAL512 ZT lineage required')
    original=read(config['original_source_bank_manifest'])
    if not original['complete'] or original['committed_records']!=87329 or original['contract']['identity']['checkpoint_sha256']!=SOURCE:raise PermissionError('Original complete c00 source only')
    with np.load(bound(config['source_scalar_cache']),allow_pickle=False) as a:
        at=crop(a['index'],index);source=np.stack([a[k][at] for k in scoring.COMPONENTS],axis=1)
    arrays={};arrays['Z_T']=score_components(source,zcal)
    rawreports=[]
    for d in config['new_banks']:
        name='U_'+d['condition'];raw=raw_bank_scalars(d,index,calibrations[name])
        arrays[name]=(raw['score'],raw['valid'].astype(bool))
        residual=np.stack([raw[k] for k in scoring.COMPONENTS],axis=1)
        arrays[name+'_ZT_FIXED']=score_components(residual,zcal)
        rawreports.append(dict(condition=d['condition'],manifest=d['manifest'],R3max=raw['R3max'],canonical_scores_preserved=True))
    mean_prov=read(config['Mean512']['provenance'])
    mean_gate=next(x for x in mean_prov['provenance']['gates'] if x['id']=='history_mean512')
    if mean_prov['calculation_inputs_sha256']!=config['Mean512']['inputs']['sha256'] or mean_gate['threshold']!=config['Mean512']['threshold']:
        raise PermissionError('Accepted Mean512 canonical inputs/original source threshold')
    with np.load(bound(config['Mean512']['inputs']),allow_pickle=False) as a:
        at=crop(a['index'],index);arrays['Mean512_SOURCECAL']=(a['score'][at].copy(),a['valid'][at].copy())
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    sealed={name:save_arrays(output,name,index,*values) for name,values in arrays.items()}
    write(output/'ALL_RAW_AND_VIEWS_BEFORE_LABELS.json',dict(status='SEALED_3RAW_7HYBRID_VIEWS_AND_MEAN',labels_read=0,
        views=sealed,raw_reports=rawreports,calibrations=calibrations,source_config=config))
    # Only after all newly produced raw and every fixed view are sealed.
    with np.load(bound(config['truth']),allow_pickle=False) as a:labels=a['full_labels'].copy()
    if labels.shape!=(87841,):raise ValueError('Full native truth axis required')
    results={};rows=[];phases=[]
    for name,(score,valid) in arrays.items():
        tau=(config['Mean512']['threshold'] if name=='Mean512_SOURCECAL' else zcal['threshold'] if name.endswith('_ZT_FIXED') or name=='Z_T' else calibrations[name]['threshold'])
        reasons=[None if v else 'MISSING_REQUIRED_COMPONENT' for v in valid]
        r=evaluation.evaluate_arrays(index=index,score=score,valid=valid,available_index=index,invalid_reasons=reasons,
            full_labels=labels,threshold=tau,bank_complete=True,original_length=87841,warmup_points=512,
            affiliation_identity=bound(config['affiliation_identity']))
        reject_dependency_na(r);results[name]=r
        evaluation.save_evaluation(output/name,r,index=index,score=score,valid=valid,available_index=index,
            invalid_reasons=reasons,full_labels=labels,provenance=dict(config=config,view=name,score_asset=sealed[name],no_QUERY_fitting=True))
        p=event_diagnostics(r);phases.extend(dict(method=name,**v) for v in p)
        row=dict(method=name,AP=r['ranking'].get('AP'),ROC_AUC=r['ranking'].get('ROC_AUC'),
            points=r['ranking']['evaluated_points'],positive_points=r['ranking'].get('positive_points'),
            threshold=tau,permission='cached sourceCAL baseline' if name=='Mean512_SOURCECAL' else 'same targetCAL512 plan; ownCAL or fixed ZT',
            complete_events=r['coverage']['main_eligible_events'],original_events=r['coverage']['original_events'])
        for prefix,stream in [('strict','strict_point'),('confirmed3','confirmed3_current')]:
            st=r['streams'][stream];point=st['point'];ev=st['main_events']['metrics'];normal=st['normal']
            for k in ['TP','FP','Precision','Recall','F1']:row[prefix+'_'+k]=point[k]
            for k in ['M','Ntrue','one_to_one_eventPrecision','one_to_one_eventRecall','delay_right_censored_events']:row[prefix+'_'+k]=ev[k]
            for k in ['normal_exposure_points','normal_FPR','normal_alarm_fragments','normal_alarm_fragments_per_1000_points']:row[prefix+'_'+k]=normal[k]
            row[prefix+'_positive_fraction']=r['ranking'].get('positive_fraction')
            for k in ['Aff_P','Aff_R','Aff_F1','reason']:row[prefix+'_'+k]=st['affiliation']['global_result'].get(k)
            delays=[e['delay_points'] for e in st['main_events']['events'] if e['eligible'] and e['delay_points'] is not None]
            row[prefix+'_hit_delay_median']=float(np.median(delays)) if delays else None
            whole=[q for q in p if q['stream']==stream and q['phase']=='all' and q['eligible']]
            row[prefix+'_max_main_event_observed_miss']=max((q['event_longest_miss'] for q in whole),default=None)
            steady=[q for q in p if q['stream']==stream and q['phase']=='steady_after512' and q['duration']>512]
            row[prefix+'_steady_points']=sum(q['total_points'] for q in steady)
            row[prefix+'_steady_scored_points']=sum(q['scored_points'] for q in steady)
            row[prefix+'_steady_unknown_points']=sum(q['unknown_points'] for q in steady)
            row[prefix+'_steady_detected_points']=sum(q['detected_points'] for q in steady)
            row[prefix+'_steady_all_missed_events']=complete_steady_all_missed(steady)
        rows.append(row)
    paired=[]
    by={r['method']:r for r in rows}
    for reference,candidate in PAIRS:
        for metric in ['AP','ROC_AUC','strict_TP','strict_FP','confirmed3_TP','confirmed3_FP','confirmed3_M','confirmed3_normal_FPR','confirmed3_max_main_event_observed_miss']:
            a,b=by[reference][metric],by[candidate][metric]
            paired.append(dict(reference=reference,candidate=candidate,metric=metric,delta=None if a is None or b is None else b-a))
    write(output/'RESULT.json',dict(status='COMPLETED_FIXED_PSM_U_ABC',seconds=time.monotonic()-began,rows=rows,paired=paired,
        source_config=config,main_views=list(OWN),auxiliary_views=[x+'_ZT_FIXED' for x in OWN[1:]],
        interpretation='One seed/synthetic family/screened PSM development subset. Cross-CAL FPR not matched alarm cost. No robustness confirmation.'))
    for filename,table in [('metrics.csv',rows),('fixed_comparisons.csv',paired),('event_phases.csv',phases)]:
        fields=list(dict.fromkeys(k for row in table for k in row))
        with (output/filename).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(table)
    write(output/'EXTERNAL_FULL_PSM_REFERENCE.json',config['external_full_axis_reference'])
    return dict(status='COMPLETED_FIXED_PSM_U_ABC',views=len(rows),new_GPUh=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True)
    a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Exact fixed config required')
    print(json.dumps(run(json.loads(Path(a.config).read_text()))))

if __name__=='__main__':main()
