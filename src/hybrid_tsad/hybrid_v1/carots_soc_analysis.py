"""Fixed two-arm/two-score SOC-control evaluation; no fitting or score search."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from . import carots_soc_control as control
from . import r2_learnability_analysis as old

MODELS=('LAG_U','CAROTS_SOC_LAG_U')
VIEWS=('A_RESIDUAL','B_REPRESENTATION')
NORMAL=('CROSS_N','NORMAL_DRIFT','HARD_NORMAL','AUGMENT_NORMAL')
NULL='NO_DRIVER0_COUPLING'
bound,sha,dump,csvwrite=old.bound,old.sha,old.dump,old.csvwrite


def representation_inputs(records,calibration):
    rows=list(records);index=np.array([r['index'] for r in rows],dtype=np.int64)
    if not np.array_equal(index,np.arange(512,2560)):
        raise ValueError('Full2048 original feature axis required')
    features=np.stack([np.asarray(r['representation']['feature']) for r in rows])
    score=np.array([r['representation']['score'] for r in rows],dtype=np.float64)
    expected=control.representation_scores(features,calibration)
    if not np.allclose(score,expected,rtol=0,atol=1e-12):
        raise ValueError('Archived feature score differs from normalCAL cosine replay')
    for r in rows:
        p=r['representation'];t=r['index']
        if (p['feature_identity']!=control.FEATURE_ID or p['available_index']!=t or
                p['input_bounds']!=[t-511,t+1] or p['forecast_target_unused']!=t+1 or
                p['threshold']!=calibration['threshold'] or
                p['alarm']!=bool(p['score']>calibration['threshold'])):
            raise PermissionError('Representation feature/score time or CAL identity mismatch')
    return dict(index=index,score=score,valid=np.ones(2048,dtype=bool),
        available_index=index.copy(),invalid_reasons=[None]*2048),features


def phase_diagnostics(array,threshold,case):
    from ..candidate_experiment import confirmed_alarm
    index=array['index'];score=array['score'];valid=array['valid']
    if not valid.all():raise ValueError('Controlled fixtures require complete finite scores')
    alarms={'strict_point':score>threshold,
            'confirmed3_current':confirmed_alarm(index,score,threshold)}
    result=[]
    for phase,(start,end) in dict(pre=(512,768),event=(768,1536),steady=(1280,1536),recovery=(1536,2560)).items():
        mask=(index>=start)&(index<end)
        positive=case in ('CROSS_LAG12_20','CROSS_BREAK') and phase in ('event','steady')
        for stream,alarm in alarms.items():
            result.append(dict(phase=phase,stream=stream,start=start,end_exclusive=end,
                points=int(mask.sum()),alarm_points=int(alarm[mask].sum()),
                anomaly_point_recall=float(alarm[mask].mean()) if positive else None,
                normal_control_FP=int(alarm[mask].sum()) if case in NORMAL else None,
                mean_score=float(score[mask].mean()),threshold=float(threshold),
                classification='NULL_NA' if case==NULL else 'FIXED_RELATION_EVENT' if positive else
                    'NORMAL_CONTROL' if case in NORMAL else 'EVENT_OUTSIDE_AUXILIARY'))
    return result


def sealed(cfg,out):
    from .chunked_bank import read_records
    from .stage2_u_evaluation import reject_truth_aliases
    reject_truth_aliases(cfg)
    freeze=bound(cfg['science_freeze'])
    if sha(freeze)!=control.SCIENCE_FREEZE_SHA:raise PermissionError('Root science identity required')
    seal=json.loads(bound(cfg['raw_seal']).read_text())
    if seal.get('all_raw_complete') is not True or seal.get('banks')!=cfg['banks']:
        raise PermissionError('Full14 residual+feature seal required before truth')
    if [(b['method'],b['case']) for b in cfg['banks']]!=[(m,c) for m in MODELS for c in control.CASES]:
        raise ValueError('Fixed two arms by seven unique cases; no repeated SHAM')
    if len({str(Path(b['directory']).absolute()) for b in cfg['banks']})!=14:
        raise ValueError('Distinct raw banks required')
    widths={}
    for name in MODELS:
        desc=cfg['models'][name]
        report=json.loads(bound(desc['training']).read_text())
        arm=json.loads(bound(desc['arm_config']).read_text())
        if (report.get('attempts')!=1000 or report.get('updates')!=1000 or
                report.get('endpoint_attempt')!=999 or
                report.get('lag_update_evidence',{}).get('update_exposure_gate') is not True or
                arm['model_config'].get('self_lag_enabled') is not True or
                arm['model_config'].get('cross_lag_enabled') is not True):
            raise PermissionError('Fixed1000 original signed-lag endpoint identity required')
        if name=='CAROTS_SOC_LAG_U' and (report.get('SOC_nonzero_gradient_steps',0)<1 or
                arm.get('SOC')!={'weight':.1,'temperature':.1,'similarity':.5}):
            raise PermissionError('No actual fixed SOC training exposure')
        bound(desc['checkpoint'])
        widths[name]=6*arm['model_config']['width']
    # Native input bytes, not labels, are lawful before the sole truth opener.
    # Recompute the exact producer input hash; do not require an absent new key.
    manifest=json.loads(bound(cfg['query_manifest']).read_text());data={};inputs={}
    for entry in manifest['runs']:
        if entry.get('role')!='QUERY' or entry.get('domain')!=control.NAMESPACE:
            raise PermissionError('Native input role/namespace mismatch')
        with np.load(bound(entry),allow_pickle=False) as z:
            if set(z.files)!={'values','observed'}:raise PermissionError('Native input must not contain labels')
            x,mask=z['values'],z['observed']
        if (x.dtype!=np.float32 or mask.dtype!=np.bool_ or x.shape!=(2560,3) or
                mask.shape!=x.shape or not mask.all() or not np.isfinite(x).all()):
            raise ValueError('Original finite fully observed native C3 input required')
        data[entry['run']]=x.astype(float)
        inputs[entry['run']]=dict(input_sha256=hashlib.sha256(x.tobytes()+mask.tobytes()).hexdigest(),
            values_float32_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            observed_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    if set(data)!=set(control.CASES) or len(manifest['runs'])!=7:
        raise ValueError('Seven distinct fixed native query inputs required')
    arrays={};predictions={};thresholds={};metadata=[]
    for b in cfg['banks']:
        method,case=b['method'],b['case'];p=Path(b['directory'])
        m=json.loads(bound(dict(path=str(p/'manifest.json'),sha256=b['manifest_sha256'])).read_text())
        residual=json.loads(bound(b['calibration']).read_text())
        rep=json.loads(bound(b['representation_calibration']).read_text());c=m['contract']
        if (m.get('complete') is not True or m.get('committed_records')!=2048 or
                c['identity']!=b['identity'] or c['threshold']!=residual['threshold'] or
                b['identity']['checkpoint_sha256']!=cfg['models'][method]['checkpoint']['sha256'] or
                b['identity']['calibration_sha256']!=b['calibration']['sha256'] or
                b['identity']['representation_calibration_sha256']!=b['representation_calibration']['sha256'] or
                b['identity']['input_sha256']!=inputs[case]['input_sha256'] or
                residual.get('active_components')!=['r_tf','r_ms','r_pred'] or
                residual.get('valid_windows')!=512 or rep.get('valid_windows')!=512 or
                residual.get('domains')!=[old.experiment.NAMESPACE] or
                rep.get('feature_identity')!=control.FEATURE_ID or c.get('chunk_size')!=64 or
                c.get('confirmation_points')!=3 or len(c['ranges'])!=1 or
                [c['ranges'][0]['start'],c['ranges'][0]['end']]!=[512,2560]):
            raise PermissionError('Raw/CAL/weights/feature dimensions or coverage differ')
        for chunk in m['chunks']:
            if sha(p/chunk['file'])!=chunk['sha256']:raise ValueError('Changed raw chunk bytes')
        rows=list(read_records(p))
        if any(r['run']!=case or r['domain']!=control.NAMESPACE for r in rows):
            raise ValueError('Case/namespace identity mismatch')
        if ('query_manifest_sha256' in b['identity'] and
                b['identity']['query_manifest_sha256']!=cfg['query_manifest']['sha256']):
            raise PermissionError('Existing query-manifest identity differs')
        for row in rows:
            native=row['replay']['input_identity']
            if any(native.get(k)!=inputs[case][k] for k in ('values_float32_sha256','observed_sha256')):
                raise PermissionError('Raw record native-input replay identity differs')
        a=old.array_inputs(rows,residual);z,features=representation_inputs(rows,rep)
        if features.shape[1]!=widths[method]:raise ValueError('Feature dimension differs from model fixed6*width')
        arrays[method,'A_RESIDUAL',case]=a;arrays[method,'B_REPRESENTATION',case]=z
        predictions[method,case]=old.prediction_inputs(rows)
        thresholds[method,'A_RESIDUAL',case]=residual['threshold']
        thresholds[method,'B_REPRESENTATION',case]=rep['threshold']
        with (out/(method+'__'+case+'__RAW_FEATURE_SCORES.npz')).open('xb') as f:
            np.savez_compressed(f,index=a['index'],available_index=a['available_index'],
                features=features,score_A=a['score'],score_B=z['score'],
                r_tf=np.array([r['r_tf'] for r in rows]),r_ms=np.array([r['r_ms'] for r in rows]),
                r_pred=np.array([r['r_pred'] for r in rows]))
        metadata.append(dict(method=method,case=case,manifest_sha256=b['manifest_sha256'],
                             feature_dimensions=list(features.shape),records=len(rows)))
    dump(out/'PRELABEL_ALL14_RAW_GATES.json',dict(banks=metadata,all_complete=True,
        all_features_replayed=True,native_inputs=inputs,truth_opened=False,
        optional_query_manifest_key='Checked when producer recorded it; not newly required',
        SHAM='CROSS_N exact alias, not another forward'))
    return arrays,predictions,thresholds,data


def run(cfg):
    from .controlled_wave_evaluation import require_metric_dependencies,reject_dependency_na
    from ..engine.official_affiliation import load_official_affiliation
    from .evaluation import evaluate_arrays,save_evaluation
    from .stage2_u_evaluation import long_miss
    from .named_evaluate_cli import point_pair,paired_events
    from .cross_centered_analysis import frozen_ar
    require_metric_dependencies();aff=str(bound(cfg['affiliation_identity']));load_official_affiliation(aff)
    for p,h in cfg['source_sha256'].items():
        if sha(p)!=h:raise PermissionError('Pinned analysis source changed')
    if cfg['source_sha256'].get(str(Path(__file__).resolve()))!=sha(__file__):
        raise PermissionError('Actual analysis caller not pinned')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    arrays,predictions,thresholds,data=sealed(cfg,out)
    # Sole truth opener is after all14 raw residual/feature identity gates.
    with np.load(bound(cfg['truth']),allow_pickle=False) as z:truth={k:z[k] for k in z.files}
    if set(truth)!=set(control.CASES):raise ValueError('Fixed seven-case truth required')
    for case,y in truth.items():
        expected=np.zeros(2560,np.int8)
        if case in ('CROSS_LAG12_20','CROSS_BREAK'):expected[768:1536]=1
        if not np.array_equal(y,expected):raise ValueError('Original event boundary changed')
    ar,ar_pred,taus=frozen_ar(cfg,data,out)
    predictions.update(ar_pred)
    for (method,case),a in ar.items():
        arrays[method,'FIXED_AR',case]=a;thresholds[method,'FIXED_AR',case]=taus[method]
    metrics={};summary=[];phases=[];misses=[];mse=[];pairs=[];prediction_pairs=[]
    for (method,case),p in predictions.items():
        mse+=old.mse_rows(p,method,case)
        np.savez_compressed(out/(method+'__'+case+'__PREDICTION_INPUTS.npz'),**p)
    for (method,view,case),a in arrays.items():
        key=method,view,case;tau=thresholds[key]
        phases += [dict(method=method,view=view,case=case,**r) for r in phase_diagnostics(a,tau,case)]
        if case==NULL:
            summary.append(dict(method=method,view=view,case=case,status='NULL_MECHANISM_CLASSIFICATION_EVENT_HEALTHY_FPR_NA',
                ranking=None,streams=None,threshold=tau,mean_score=float(a['score'].mean())))
            continue
        result=evaluate_arrays(**a,full_labels=truth[case],threshold=tau,bank_complete=True,
            original_length=2560,warmup_points=512,affiliation_identity=aff)
        reject_dependency_na(result);metrics[key]=result
        save_evaluation(out/'metrics'/(method+'__'+view+'__'+case),result,**a,
            full_labels=truth[case],provenance=dict(config=cfg,method=method,view=view,case=case))
        summary.append(dict(method=method,view=view,case=case,ranking=result['ranking'],coverage=result['coverage'],
            streams={s:dict(point=v['point'],events=v['main_events']['metrics'],affiliation=v['affiliation'],
                normal=v['normal'] if case in NORMAL else dict(status='EVENT_OUTSIDE_NORMAL_AUXILIARY'))
                for s,v in result['streams'].items()}))
        misses += [dict(method=method,view=view,case=case,**r) for r in long_miss(result)]
    for case in control.CASES:
        prediction_pairs += [dict(reference=MODELS[0],candidate=MODELS[1],case=case,**r)
            for r in old.paired_mse(predictions[MODELS[0],case],predictions[MODELS[1],case])]
        if case==NULL:continue
        for view in VIEWS:
            a=(MODELS[0],view,case);b=(MODELS[1],view,case)
            delta,_=point_pair(arrays[a],arrays[b],truth[case],0)
            pairs.append(dict(reference=MODELS[0],candidate=MODELS[1],view=view,case=case,
                point=delta,events=paired_events(metrics[a],metrics[b])))
    csvwrite(out/'H1_MSE_ALL_METHODS_CASES.csv',mse)
    csvwrite(out/'PAIRED_H1_MSE_DIFFERENCES.csv',prediction_pairs)
    csvwrite(out/'ALL_FIXED_PHASE_DETECTION.csv',phases)
    if misses:csvwrite(out/'ORIGINAL_EVENT_LONG_MISS.csv',misses)
    dump(out/'ALL_FIXED_DETECTION_METRICS.json',summary);dump(out/'PAIRED_ORIGINAL_METRICS.json',pairs)
    dump(out/'RECEIPT.json',dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT',neural_models=2,
        neural_score_views=2,CPU_controls=2,physical_cases=7,all_score_case_rows=len(summary),
        primary='B_REPRESENTATION matched objective contrast, not query-selected',
        query_latent_run=9504,model_seed=2025,score_axis=[512,2560],
        statistical_confirmation=False,new_fit_AR=False,new_neural_forward=False,
        normal_FPR_cases=list(NORMAL),null='NA_NO_HEALTH_OR_TIMED_EVENT',SHAM='CROSS_N alias not independent',
        decision_requirement='Sustained unseen LAG benefit beyond normal/negative controls and simple AR; no new losses if absent'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Unpinned config')
    run(json.loads(Path(a.config).read_text()))


if __name__=='__main__':main()
