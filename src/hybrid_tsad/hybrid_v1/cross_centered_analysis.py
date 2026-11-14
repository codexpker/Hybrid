"""One centered-Cross candidate comparison; existing metric mathematics reused."""
import argparse,json
from pathlib import Path
import numpy as np
from . import r2_learnability as experiment
from . import r2_learnability_analysis as old

MODELS=('LAG_U','CENTERED_MAG','CROSS_OFF_LAG_U')
METHODS=MODELS+('AR_SELF','ARX12')
CASES=experiment.CASES
PAIRS=(('LAG_U','CENTERED_MAG'),('CROSS_OFF_LAG_U','CENTERED_MAG'))
POLICY='CROSS_CENTERED_QK_MAGNITUDE_V1'
bound,sha,dump,csvwrite=old.bound,old.sha,old.dump,old.csvwrite


def policy_gate(spec,source):
    p=json.loads(bound(spec).read_text())
    if p.get('policy')!=POLICY or p.get('source_sha256')!=source['sha256'] or p.get('new_parameters')!=0:
        raise PermissionError('Candidate routing policy omitted or changed')
    bound(source)
    return p


def sealed(cfg,out):
    from .chunked_bank import read_records
    seal=json.loads(bound(cfg['raw_seal']).read_text())
    if seal.get('banks')!=cfg['banks'] or seal.get('all_raw_complete') is not True:raise PermissionError('Full15 seal required')
    if [(b['method'],b['case']) for b in cfg['banks']]!=[(m,c) for m in MODELS for c in CASES]:raise ValueError('Fixed3x5 grid')
    if len({str(Path(b['directory']).absolute()) for b in cfg['banks']})!=15:raise ValueError('Distinct banks')
    policy_gate(cfg['models']['CENTERED_MAG']['routing_policy'],cfg['candidate_policy_source'])
    for name in MODELS:
        desc=cfg['models'][name];report=json.loads(bound(desc['training']).read_text());arm=json.loads(bound(desc['arm_config']).read_text())
        expected='CROSS_OFF_LAG_U' if name=='CROSS_OFF_LAG_U' else 'LAG_U'
        if (report.get('attempts')!=1000 or report.get('updates')!=1000 or report.get('endpoint_attempt')!=999
                or report.get('selection')!='FIXED_ATTEMPT_ENDPOINT_NO_BEST_CHECKPOINT'
                or report.get('lag_update_evidence',{}).get('update_exposure_gate') is not True
                or arm.get('arm')!=expected or arm['model_config'].get('self_lag_enabled') is not True
                or arm['model_config'].get('cross_lag_enabled')!=(name!='CROSS_OFF_LAG_U')):
            raise PermissionError('Invalid training/arm identity')
        for key in ('QK_before','QK_after'):bound(desc[key])
        for route in desc['routes'].values():bound(route);bound(dict(path=route['npz_path'],sha256=route['npz_sha256']))
    arrays={};predictions={};cals={};meta=[]
    for b in cfg['banks']:
        k=b['method'],b['case'];p=Path(b['directory']);m=json.loads(bound(dict(path=str(p/'manifest.json'),sha256=b['manifest_sha256'])).read_text())
        cal=json.loads(bound(b['calibration']).read_text());c=m['contract']
        if (m.get('complete') is not True or m.get('committed_records')!=2048 or c['identity']!=b['identity']
                or c['threshold']!=cal['threshold'] or b['identity']['calibration_sha256']!=b['calibration']['sha256']
                or cal.get('active_components')!=['r_tf','r_ms','r_pred'] or cal.get('valid_windows')!=512
                or cal.get('domains')!=[experiment.NAMESPACE] or cal.get('source_domain_equal_weight') is not False
                or c.get('confirmation_points')!=3 or c.get('chunk_size')!=64 or len(c['ranges'])!=1
                or [c['ranges'][0]['start'],c['ranges'][0]['end']]!=[512,2560]):raise PermissionError('OwnCAL/complete grid identity')
        for chunk in m['chunks']:
            if sha(p/chunk['file'])!=chunk['sha256']:raise ValueError('Raw chunk bytes changed')
        rows=list(read_records(p))
        if any(r['run']!=b['case'] for r in rows):raise ValueError('Case mismatch')
        arrays[k]=old.array_inputs(rows,cal);predictions[k]=old.prediction_inputs(rows);cals[k]=cal
        meta.append(dict(method=k[0],case=k[1],manifest_sha256=b['manifest_sha256'],records=len(rows)))
    dump(out/'PRELABEL_ALL15_RAW_GATES.json',dict(banks=meta,all_complete=True,truth_opened=False,new_candidates=1))
    return arrays,predictions,cals


def frozen_ar(cfg,data,out):
    """Reuse old coefficients, target scales, and thresholds. No fitting/CAL pass."""
    helper=experiment.load_original_controls(bound(cfg['original_controls']))
    saved=json.loads(bound(cfg['frozen_ar_controls']).read_text());scores={};pred={}
    if set(saved['models'])!={'AR_SELF','ARX12'} or saved.get('lambda_penalty')!=1.:raise ValueError('Fixed AR controls')
    for name,by in saved['models'].items():
        models={int(k):v for k,v in by.items()};scale=np.array([models[i]['target_scale'] for i in (1,2)])
        for case,x in data.items():
            t=np.arange(512,2560,dtype=np.int64);p=experiment.predict_controls(helper,{name:models},x,t)[name]
            error=((p-x[t,1:])/scale)**2
            scores[name,case]=dict(index=t,score=error.mean(1),valid=np.ones(len(t),bool),available_index=t.copy(),invalid_reasons=[None]*len(t))
            full=np.full((len(t),3),np.nan);full[:,1:]=p;z=full.copy();z[:,1:]=error
            pred[name,case]=dict(index=t,prediction=full,target=x[t].astype(float),normalized_squared_error=z,physical_squared_error=(full-x[t])**2,valid=np.isfinite(full))
    dump(out/'REUSED_AR_PARAMETERS.json',dict(source=cfg['frozen_ar_controls'],new_fit=False,new_calibration=False,thresholds=saved['thresholds']))
    return scores,pred,saved['thresholds']


def routes_with_mapping(entries,method,case):
    selected,grids,neighbours=old.route_rows(entries,method,case)
    lookup={}
    for e in entries:
        meta=e['meta'];key=e['index'],e['pass_name']
        if method=='CENTERED_MAG':
            if meta.get('cross_routing_policy')!=POLICY:raise PermissionError('Raw route candidate policy missing')
            raw=np.asarray(meta['correlation']);mapped=np.asarray(meta['cross_routing_score']);center=np.asarray(meta['cross_centering_term'])
            if mapped.shape!=(1,3,3,33) or not np.allclose(mapped,np.abs(raw-center),atol=1e-6,rtol=1e-6):raise ValueError('Mapped score not centered magnitude')
            lookup[key]=(mapped,center)
    for row in grids:
        i,j,l=row['target'],row['source'],row['lag']
        if method=='CENTERED_MAG':
            mapped,center=lookup[row['point'],row['pass_name']]
            row['applied_cross_routing_score']=float(mapped[0,i,j,l]);row['centering_term']=float(center[0,i,j,l]);row['routing_policy']=POLICY
        else:
            row['applied_cross_routing_score']=row['signed_QK_correlation'];row['centering_term']=None;row['routing_policy']='ORIGINAL_SIGNED'
    return selected,grids,neighbours


def run(cfg):
    from .controlled_wave_evaluation import require_metric_dependencies,reject_dependency_na
    from ..engine.official_affiliation import load_official_affiliation
    from .evaluation import evaluate_arrays,save_evaluation
    from .stage2_u_evaluation import long_miss
    from .named_evaluate_cli import point_pair,paired_events
    from .controlled_wave_recovery import load_bundle
    require_metric_dependencies();identity=str(bound(cfg['affiliation_identity']));load_official_affiliation(identity)
    for p,h in cfg['source_sha256'].items():
        if sha(p)!=h:raise PermissionError('Pinned analysis source changed')
    if cfg['source_sha256'].get(str(Path(__file__).resolve()))!=sha(__file__):raise PermissionError('Caller not pinned')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    arrays,predictions,cals=sealed(cfg,out)
    with np.load(bound(cfg['truth']),allow_pickle=False) as z:truth={k:z[k] for k in z.files}
    if set(truth)!=set(CASES):raise ValueError('Five-case truth')
    for case,y in truth.items():
        expected=np.zeros(2560,np.int8)
        if case in ('CROSS_LAG12_20','CROSS_BREAK'):expected[768:1536]=1
        if not np.array_equal(y,expected):raise ValueError('Original event boundaries')
    data={}
    for e in json.loads(bound(cfg['query_manifest']).read_text())['runs']:
        with np.load(bound(e),allow_pickle=False) as z:data[e['run']]=z['values'].astype(float)
    if set(data)!=set(CASES):raise ValueError('Five-case query data')
    extra,p,taus=frozen_ar(cfg,data,out);arrays.update(extra);predictions.update(p)
    metrics={};summary=[];mse=[];miss=[];pairs=[];metricpairs=[]
    for method in METHODS:
        for case in CASES:
            key=method,case;a=arrays[key];pred=predictions[key];tau=cals[key]['threshold'] if method in MODELS else taus[method]
            np.savez_compressed(out/(method+'__'+case+'__PREDICTION_INPUTS.npz'),**pred);mse+=old.mse_rows(pred,method,case)
            if case==old.NULL:
                summary.append(dict(method=method,case=case,status='NULL_MECHANISM_CONTROL_EVENT_AND_NORMAL_FPR_NA',ranking=None,normal_FPR=None,event=None,threshold=tau,score_mean=float(a['score'].mean())))
                np.savez_compressed(out/(method+'__NULL_RAW_SCORES.npz'),**{k:v for k,v in a.items() if k!='invalid_reasons'});continue
            r=evaluate_arrays(**a,full_labels=truth[case],threshold=tau,bank_complete=True,original_length=2560,warmup_points=512,affiliation_identity=identity)
            reject_dependency_na(r);metrics[key]=r;save_evaluation(out/'metrics'/(method+'__'+case),r,**a,full_labels=truth[case],provenance=dict(config=cfg,method=method,case=case))
            summary.append(dict(method=method,case=case,ranking=r['ranking'],coverage=r['coverage'],streams={s:dict(point=v['point'],events=v['main_events']['metrics'],normal=v['normal'],affiliation=v['affiliation']) for s,v in r['streams'].items()}))
            miss += [dict(method=method,case=case,**row) for row in long_miss(r)]
    for reference,candidate in PAIRS:
        for case in CASES:
            pairs += [dict(reference=reference,candidate=candidate,case=case,**r) for r in old.paired_mse(predictions[reference,case],predictions[candidate,case])]
            if case!=old.NULL:
                delta,_=point_pair(arrays[reference,case],arrays[candidate,case],truth[case],0)
                metricpairs.append(dict(reference=reference,candidate=candidate,case=case,point=delta,events=paired_events(metrics[reference,case],metrics[candidate,case])))
    selected=[];grids=[];near=[];trainmeta=[]
    for method in MODELS:
        desc=cfg['models'][method]
        for case in CASES:
            spec=desc['routes'][case];entries=load_bundle(bound(spec),bound(dict(path=spec['npz_path'],sha256=spec['npz_sha256'])))
            s,g,n=routes_with_mapping(entries,method,case);selected+=s;grids+=g;near+=n
        trainmeta.append(dict(method=method,training=json.loads(bound(desc['training']).read_text()),QK_before=json.loads(bound(desc['QK_before']).read_text()),QK_after=json.loads(bound(desc['QK_after']).read_text())))
    csvwrite(out/'H1_MSE_ALL_METHODS_CASES.csv',mse);csvwrite(out/'PAIRED_H1_MSE_DIFFERENCES.csv',pairs);csvwrite(out/'ORIGINAL_EVENT_LONG_MISS.csv',miss)
    csvwrite(out/'ROUTE_ALL_SELECTED_SLOTS.csv',selected);csvwrite(out/'ROUTE_FULL_LAG_GRID.csv',grids);csvwrite(out/'ROUTE_SOURCE0_LAG12_NEIGHBOURHOOD.csv',near)
    dump(out/'ALL_FIXED_DETECTION_METRICS.json',summary);dump(out/'PAIRED_ORIGINAL_METRICS.json',metricpairs);dump(out/'TRAINING_PARAMETER_AND_QK_EVIDENCE.json',trainmeta)
    dump(out/'RECEIPT.json',dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT',neural_models=3,CPU_controls=2,physical_cases=5,classifications=20,null_controls=5,
        query_latent_run=9304,model_seed=2025,score_axis=[512,2560],positive_fraction_event_cases=.375,new_fit_AR=False,new_neural_forward=False,
        one_candidate_only=True,statistical_confirmation=False,normal_FPR_cases=['CROSS_N','NORMAL_DRIFT'],decision_requirement='Routing improvement alone is insufficient; if normal prediction and anomaly detection gains are absent this specific revision stops.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Unpinned config')
    run(json.loads(Path(a.config).read_text()))

if __name__=='__main__':main()
