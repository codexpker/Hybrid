"""Fixed CPU comparison of completed matched-process R2 banks.

No Torch, neural forward, score search or threshold choice. Existing metric,
event, confirmation, long-miss and official Affiliation implementations are used.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from . import r2_learnability as experiment

MODELS=experiment.ALL_MODELS
METHODS=MODELS+('AR_SELF','ARX12')
CASES=experiment.CASES
PAIRS=(('CROSS_OFF_LAG_U','LAG_U'),('U','LAG_U'),('SOURCE','LAG_U'))
NULL='NO_DRIVER0_COUPLING'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bound(asset):
    if not asset.get('sha256') or sha(asset['path'])!=asset['sha256']:raise PermissionError('Changed or unsealed asset')
    return Path(asset['path'])
def dump(p,x):experiment.dump(p,x)
def csvwrite(p,rows):
    if not rows:raise ValueError('Empty required table')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def prediction_inputs(records):
    """Original physical h1 outputs, observed targets and original r_pred values."""
    rows=list(records);index=np.asarray([r['index'] for r in rows],np.int64)
    p=np.asarray([r['prediction'][0] for r in rows],np.float64)
    y=np.asarray([r['replay']['current_target'][0] for r in rows],np.float64)
    z=np.asarray([r['per_channel']['r_pred'] for r in rows],np.float64)
    if p.shape!=y.shape or p.shape!=z.shape or p.shape!=(len(rows),3):raise ValueError('Original allC h1 archive required')
    return dict(index=index,prediction=p,target=y,normalized_squared_error=z,
        physical_squared_error=(p-y)**2,valid=np.isfinite(p)&np.isfinite(y)&np.isfinite(z))


def mse_rows(inputs,method,case):
    out=[]
    for name,channels in (('ALL_C',[0,1,2]),('DRIVER_0',[0]),('RESPONSE_1',[1]),('RESPONSE_2',[2]),('RESPONSE_MEAN',[1,2])):
        valid=inputs['valid'][:,channels];physical=inputs['physical_squared_error'][:,channels]
        normalized=inputs['normalized_squared_error'][:,channels]
        n=int(valid.sum());complete_group=bool(valid.any(0).all())
        out.append(dict(method=method,case=case,target_group=name,channels=channels,valid_channel_points=n,
            original_scale_MSE=float(physical[valid].mean()) if n and complete_group else None,
            normalized_MSE=float(normalized[valid].mean()) if n and complete_group else None,
            normalization_scope='FIT_target_std_per_response' if method in ('AR_SELF','ARX12') else 'Original_visible_local_forecast_scale',
            availability='h1 arrived t, prediction issued from history ending t-1',
            interpretation='Normal prediction test' if case=='CROSS_N' else 'Mechanism/operating-change descriptive prediction error',
            NA_reason=None if n and complete_group else 'GROUP_NOT_FULLY_PREDICTED_OR_NO_VALID_OBSERVATION'))
    return out


def paired_mse(reference,candidate):
    if not np.array_equal(reference['index'],candidate['index']) or not np.array_equal(reference['target'],candidate['target'],equal_nan=True):
        raise ValueError('Prediction pair original axis or targets differ')
    good=reference['valid']&candidate['valid'];rows=[]
    for name,ix in (('ALL_C',[0,1,2]),('RESPONSE_1',[1]),('RESPONSE_2',[2]),('RESPONSE_MEAN',[1,2])):
        mask=good[:,ix];n=int(mask.sum())
        for field in ('physical_squared_error','normalized_squared_error'):
            a,b=reference[field][:,ix],candidate[field][:,ix]
            rows.append(dict(target_group=name,error_field=field,n=n,reference_MSE=float(a[mask].mean()) if n else None,
                candidate_MSE=float(b[mask].mean()) if n else None,
                candidate_minus_reference=float((b-a)[mask].mean()) if n else None,CI=None,
                statistics='One training seed and one independent query latent run; descriptive paired differences only'))
    return rows


def route_rows(entries,method,case):
    """Every saved route slot; no new inference, best-lag selection or reweighting."""
    selected=[];grids=[];neighbour=[]
    keys=set()
    for entry in entries:
        t=int(entry['index'] if 'index' in entry else entry['point']);mode=entry['pass_name'];meta=entry['meta']
        if 'index' in entry and 'point' in entry and entry['index']!=entry['point']:
            raise ValueError('Ambiguous route time fields')
        if t not in experiment.ROUTE_ANCHORS or (t,mode) in keys:raise ValueError('Unexpected or repeated route anchor')
        keys.add((t,mode));forecast=mode=='forecast'
        cross_active=method!='CROSS_OFF_LAG_U'
        corr=np.asarray(meta['correlation'],np.float64);candidate=np.asarray(meta['cross_candidate_valid'],bool)
        if corr.shape!=(1,3,3,33) or candidate.shape!=corr.shape:raise ValueError('Original full i/j/lag grid required')
        mass=np.zeros((3,3,33),np.float64)
        route=meta['cross']
        source=np.asarray(route['source']);lag=np.asarray(route['lag']);weight=np.asarray(route['weights'],np.float64);valid=np.asarray(route['selected_valid'],bool)
        if not (source.shape==lag.shape==weight.shape==valid.shape) or source.shape[:2]!=(1,3):raise ValueError('Typed selected route axes differ')
        for i in range(3):
            for slot in range(source.shape[-1]):
                j,l,w,ok=int(source[0,i,slot]),int(lag[0,i,slot]),float(weight[0,i,slot]),bool(valid[0,i,slot])
                if ok and (i==j or not 0<=j<3 or not 0<=l<=32 or not candidate[0,i,j,l] or not np.isfinite(w)):
                    raise ValueError('Illegal valid cross route')
                if ok:mass[i,j,l]+=w
                selected.append(dict(method=method,case=case,point=t,pass_name=mode,target=i,source=j,slot=slot,
                    selected_lag=l,weight=w,valid=ok,cross_output_active=cross_active,
                    physical_source_index=t-(1 if forecast else 0)-l if ok else None,
                    physical_formula='t-1-lag' if forecast else 't-lag',
                    interpretation='Selection is not causal identification; inactive Cross selection has no output effect' if not cross_active else 'Selected aggregation value, not proof of a causal edge'))
        for i in range(3):
            for j in range(3):
                for l in range(33):
                    value=float(corr[0,i,j,l])
                    grids.append(dict(method=method,case=case,point=t,pass_name=mode,target=i,source=j,lag=l,
                        candidate_valid=bool(candidate[0,i,j,l]),signed_QK_correlation=value if np.isfinite(value) else None,
                        selected_weight=float(mass[i,j,l]),cross_output_active=cross_active))
        for i in (1,2):
            neighbour.append(dict(method=method,case=case,point=t,pass_name=mode,target=i,source=0,
                response_sign='positive' if i==1 else 'negative',weight_at_historical_lag12=float(mass[i,0,12]),
                weight_in_predefined_lag10_14=float(mass[i,0,10:15].sum()),total_source0_weight=float(mass[i,0].sum()),
                signed_QK_at_lag12=float(corr[0,i,0,12]) if candidate[0,i,0,12] else None,
                cross_output_active=cross_active,forecast_physical_time_caution=forecast,
                caution='Historical Q/K l12 and forecast value time t-1-l differ; no requirement that forecast must select12'))
    expected={(t,mode) for t in experiment.ROUTE_ANCHORS for mode in ('forecast','reconstruct_group0','reconstruct_group1')}
    if keys!=expected:raise ValueError('All15 same-forward anchors and all3 passes required')
    return selected,grids,neighbour


def array_inputs(records,calibration):
    from .controlled_wave_evaluation import scalar_record
    values=[scalar_record(r,calibration) for r in records]
    if [r[0] for r in values]!=list(range(512,2560)):raise ValueError('Exactly full original [512,2560) score axis required')
    return dict(index=np.asarray([r[0] for r in values],np.int64),score=np.asarray([r[1] for r in values],np.float64),
        valid=np.asarray([r[2] for r in values],bool),available_index=np.asarray([r[3] for r in values],np.int64),invalid_reasons=[r[4] for r in values])


def sealed_banks(cfg,out):
    """All20 raw banks, own CAL and byte identities checked BEFORE truth opener."""
    from .chunked_bank import read_records
    seal=json.loads(bound(cfg['raw_seal']).read_text(encoding='utf-8'))
    if seal.get('banks')!=cfg['banks'] or seal.get('all_raw_complete') is not True:
        raise PermissionError('Completion seal must enumerate exact20 bank descriptors')
    if [(b['method'],b['case']) for b in cfg['banks']]!=[(m,c) for m in MODELS for c in CASES]:
        raise ValueError('All four fixed models and five physical cases required, no SHAM duplication')
    if len({str(Path(b['directory']).absolute()) for b in cfg['banks']})!=20:raise ValueError('Distinct raw banks required')
    arrays={};predictions={};calibrations={};metadata=[]
    for name in MODELS[1:]:
        report=json.loads(bound(cfg['models'][name]['training']).read_text())
        arm=json.loads(bound(cfg['models'][name]['arm_config']).read_text())
        if (report.get('attempts')!=1000 or report.get('updates')!=1000 or report.get('endpoint_attempt')!=999
                or report.get('selection')!='FIXED_ATTEMPT_ENDPOINT_NO_BEST_CHECKPOINT'
                or report.get('lag_update_evidence',{}).get('update_exposure_gate') is not True):
            raise PermissionError('Unfinished/INCOMPLETE trained arm cannot silently enter formal evaluation')
        if (arm.get('arm')!=name or arm['model_config'].get('self_lag_enabled') is not True
                or arm['model_config'].get('cross_lag_enabled')!=(name!='CROSS_OFF_LAG_U')):
            raise PermissionError('Saved model flags differ from the declared comparison arm')
    for model in MODELS:
        if model=='SOURCE':bound(cfg['models'][model]['QK_source'])
        else:
            bound(cfg['models'][model]['QK_before']);bound(cfg['models'][model]['QK_after'])
        for case in CASES:
            route=cfg['models'][model]['routes'][case]
            bound(route);bound(dict(path=route['npz_path'],sha256=route['npz_sha256']))
    for desc in cfg['banks']:
        key=desc['method'],desc['case'];folder=Path(desc['directory']);manifest=json.loads(bound(dict(path=str(folder/'manifest.json'),sha256=desc['manifest_sha256'])).read_text())
        cal=json.loads(bound(desc['calibration']).read_text());contract=manifest['contract']
        if (manifest.get('complete') is not True or manifest.get('committed_records')!=2048
                or contract['threshold']!=cal['threshold'] or contract['identity']!=desc['identity']
                or desc['identity']['calibration_sha256']!=desc['calibration']['sha256']
                or cal.get('active_components')!=['r_tf','r_ms','r_pred']
                or cal.get('source_domain_equal_weight') is not False
                or cal.get('domains')!=[experiment.NAMESPACE] or cal.get('valid_windows')!=512
                or contract.get('confirmation_points')!=3 or contract.get('chunk_size')!=64
                or len(contract.get('ranges',[]))!=1
                or [contract['ranges'][0]['start'],contract['ranges'][0]['end']]!=[512,2560]):
            raise PermissionError('Complete2048 bank or own normal CAL identity differs')
        for chunk in manifest['chunks']:
            if sha(folder/chunk['file'])!=chunk['sha256']:raise ValueError('Changed raw chunk')
        rows=list(read_records(folder))
        if any(r['run']!=desc['case'] for r in rows):raise ValueError('Case stream mismatch')
        arrays[key]=array_inputs(rows,cal);predictions[key]=prediction_inputs(rows);calibrations[key]=cal
        metadata.append(dict(method=key[0],case=key[1],manifest_sha256=desc['manifest_sha256'],calibration=desc['calibration'],records=len(rows)))
    dump(out/'PRELABEL_ALL20_RAW_GATES.json',dict(all_complete=True,banks=metadata,truth_opened=False,scorer_recomputed=True))
    return arrays,predictions,calibrations


def baseline_outputs(cfg,data,out):
    """Original fixed AR_SELF/ARX12 fit, target scale, and CAL quantile99 math."""
    from .stage2_u_matched import load_roles
    helper=experiment.load_original_controls(bound(cfg['original_controls']))
    normal=load_roles(str(bound(cfg['normal_manifest'])),{'FIT','CAL'},512,32);runs={r.role:r for r in normal}
    plan=json.loads(bound(cfg['plan']).read_text());fit_t=np.asarray([e['index'] for e in plan['FIT']],np.int64)
    cal_t=np.asarray([e['index'] for e in plan['CAL']],np.int64);models=experiment.fit_controls(helper,runs['FIT'].values,fit_t)
    cal_pred=experiment.predict_controls(helper,models,runs['CAL'].values,cal_t)
    score_by={};pred_by={};thresholds={}
    for name,by in models.items():
        scale=np.asarray([by[c]['target_scale'] for c in (1,2)])
        cal_error=((cal_pred[name]-runs['CAL'].values[cal_t,1:])/scale)**2
        cal_score=cal_error.mean(1);thresholds[name]=float(np.quantile(cal_score,.99))
        np.savez_compressed(out/(name+'_CAL_INPUTS.npz'),index=cal_t,prediction=cal_pred[name],target=runs['CAL'].values[cal_t,1:],scale=scale,score=cal_score)
        for case,x in data.items():
            t=np.arange(512,2560,dtype=np.int64);p=experiment.predict_controls(helper,{name:by},x,t)[name]
            error=((p-x[t,1:])/scale)**2;score_by[name,case]=dict(index=t,score=error.mean(1),valid=np.ones(len(t),bool),available_index=t.copy(),invalid_reasons=[None]*len(t))
            full=np.full((len(t),3),np.nan);full[:,1:]=p;z=full.copy();z[:,1:]=error
            pred_by[name,case]=dict(index=t,prediction=full,target=x[t].astype(np.float64),normalized_squared_error=z,
                physical_squared_error=(full-x[t])**2,valid=np.isfinite(full))
    dump(out/'FITTED_AR_CONTROLS.json',dict(models=models,thresholds=thresholds,methods=['AR_SELF','ARX12'],lambda_penalty=1.,
        normalization='Original per-response target std fitted on FIT, not neural visible-local scale',
        calibration='Original np.quantile(CAL score,.99) default linear; CAL independent run',known_lag_reference_not_discovery=True,
        original_helper=cfg['original_controls'],plan=cfg['plan']))
    return score_by,pred_by,thresholds


def run(cfg):
    from .controlled_wave_evaluation import require_metric_dependencies,reject_dependency_na
    from ..engine.official_affiliation import load_official_affiliation
    from .evaluation import evaluate_arrays,save_evaluation
    from .stage2_u_evaluation import long_miss
    from .named_evaluate_cli import point_pair,paired_events
    require_metric_dependencies();identity=str(bound(cfg['affiliation_identity']));load_official_affiliation(identity)
    for path,digest in cfg['source_sha256'].items():
        if sha(path)!=digest:raise PermissionError('Changed pinned analysis/backend source')
    if cfg['source_sha256'].get(str(Path(__file__).resolve()))!=sha(__file__):
        raise PermissionError('Actual analysis caller must be in the frozen closure')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    arrays,predictions,cals=sealed_banks(cfg,out)
    # No QUERY semantic labels were opened until all20 bank gates passed.
    with np.load(bound(cfg['truth']),allow_pickle=False) as z:truth={k:z[k] for k in z.files}
    if set(truth)!=set(CASES):raise ValueError('Exact five-case complete original truth required')
    for case,y in truth.items():
        expected=np.zeros(2560,np.int8)
        if case in ('CROSS_LAG12_20','CROSS_BREAK'):expected[768:1536]=1
        if not np.array_equal(y,expected):raise ValueError('Frozen original event boundary differs')
    manifest=json.loads(bound(cfg['query_manifest']).read_text());data={}
    for e in manifest['runs']:
        with np.load(bound(e),allow_pickle=False) as z:data[e['run']]=z['values'].astype(np.float64)
    if set(data)!=set(CASES):raise ValueError('Exact query data case closure')
    extra,predextra,taus=baseline_outputs(cfg,data,out);arrays.update(extra);predictions.update(predextra)
    metrics={};summary=[];mse=[];misses=[];pred_pairs=[];metric_pairs=[]
    for method in METHODS:
        for case in CASES:
            key=method,case;a=arrays[key];pred=predictions[key]
            np.savez_compressed(out/(method+'__'+case+'__PREDICTION_INPUTS.npz'),**pred)
            mse+=mse_rows(pred,method,case)
            tau=cals[key]['threshold'] if method in MODELS else taus[method]
            if case==NULL:
                summary.append(dict(method=method,case=case,status='NULL_MECHANISM_CONTROL_EVENT_AND_NORMAL_FPR_NA',ranking=None,normal_FPR=None,event=None,
                    threshold=tau,score_mean=float(a['score'][a['valid']].mean()),score_min=float(a['score'][a['valid']].min()),score_max=float(a['score'][a['valid']].max())))
                np.savez_compressed(out/(method+'__NULL_RAW_SCORES.npz'),**{k:v for k,v in a.items() if k!='invalid_reasons'})
                continue
            result=evaluate_arrays(**a,full_labels=truth[case],threshold=tau,bank_complete=True,original_length=2560,warmup_points=512,affiliation_identity=identity)
            reject_dependency_na(result);metrics[key]=result
            save_evaluation(out/'metrics'/(method+'__'+case),result,**a,full_labels=truth[case],provenance=dict(config=cfg,method=method,case=case,SHAM_alias_only=True))
            summary.append(dict(method=method,case=case,status='EVALUATED',ranking=result['ranking'],coverage=result['coverage'],
                streams={s:dict(point=v['point'],events=v['main_events']['metrics'],affiliation=v['affiliation'],
                    normal=v['normal'] if case in ('CROSS_N','NORMAL_DRIFT') else dict(status='AUXILIARY_EVENT_OUTSIDE_REGION_NOT_PRIMARY_NORMAL_FPR')) for s,v in result['streams'].items()}))
            misses += [dict(method=method,case=case,**r) for r in long_miss(result)]
    for reference,candidate in PAIRS:
        for case in CASES:
            pred_pairs += [dict(reference=reference,candidate=candidate,case=case,**r) for r in paired_mse(predictions[reference,case],predictions[candidate,case])]
            if case!=NULL:
                effect,_=point_pair(arrays[reference,case],arrays[candidate,case],truth[case],0)
                metric_pairs.append(dict(reference=reference,candidate=candidate,case=case,point=effect,events=paired_events(metrics[reference,case],metrics[candidate,case])))
    csvwrite(out/'H1_MSE_ALL_METHODS_CASES.csv',mse);csvwrite(out/'PAIRED_H1_MSE_DIFFERENCES.csv',pred_pairs)
    if misses:csvwrite(out/'ORIGINAL_EVENT_LONG_MISS.csv',misses)
    dump(out/'ALL_FIXED_DETECTION_METRICS.json',summary);dump(out/'PAIRED_ORIGINAL_METRICS.json',metric_pairs)
    routes=[];grids=[];neighbours=[];curves=[];trainmeta=[]
    for model in MODELS:
        desc=cfg['models'][model]
        for case in CASES:
            from .controlled_wave_recovery import load_bundle
            spec=desc['routes'][case]
            entry=load_bundle(bound(spec),bound(dict(path=spec['npz_path'],sha256=spec['npz_sha256'])))
            r,g,n=route_rows(entry,model,case);routes+=r;grids+=g;neighbours+=n
        if model=='SOURCE':
            trainmeta.append(dict(method=model,updates=0,QK_source=json.loads(bound(desc['QK_source']).read_text())))
            continue
        report=json.loads(bound(desc['training']).read_text());before=json.loads(bound(desc['QK_before']).read_text());after=json.loads(bound(desc['QK_after']).read_text())
        trainmeta.append(dict(method=model,attempts=report['attempts'],updates=report['updates'],selection=report['selection'],
            state_change=report.get('state_change'),lag_update_evidence=report.get('lag_update_evidence'),lag_parameter_max_abs_changes=report.get('lag_parameter_max_abs_changes'),QK_before=before,QK_after=after))
        for r in report['trace']:
            curves.append(dict(method=model,attempt=r['attempt'],updated=r['updated'],loss=r.get('loss'),gradient_norm=r.get('gradient_norm'),
                learning_rate=r.get('learning_rate',report['learning_rate']),SELECT_diagnostic=r.get('SELECT_diagnostic'),
                mask_seed=r.get('mask_seed',r.get('actual_mask_seed')),lag_gradient_norms=json.dumps(r.get('lag_gradient_norms',{}))))
    csvwrite(out/'ROUTE_ALL_SELECTED_SLOTS.csv',routes);csvwrite(out/'ROUTE_FULL_LAG_GRID.csv',grids);csvwrite(out/'ROUTE_SOURCE0_LAG12_NEIGHBOURHOOD.csv',neighbours)
    csvwrite(out/'TRAINING_AND_SELECT_CURVES.csv',curves);dump(out/'TRAINING_PARAMETER_AND_QK_EVIDENCE.json',trainmeta)
    dump(out/'RECEIPT.json',dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT',neural_models=4,CPU_controls=2,physical_cases=5,
        classifications=24,null_controls=6,SHAM='CROSS_N logical alias; no repeated forward/independence claim',
        score_axis=[512,2560],positive_fraction_event_cases=.375,model_seed=2025,query_latent_run=9204,
        statistical_confirmation=False,new_multihorizon_detector=False,new_neural_forward=False,normal_FPR_cases=['CROSS_N','NORMAL_DRIFT']))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Exact fixed analysis config required')
    c=json.loads(Path(a.config).read_text());run(c)
if __name__=='__main__':main()
