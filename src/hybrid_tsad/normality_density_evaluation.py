"""Fixed common-axis evidence for the complete joint-density experiment.

All thresholds are sealed from legitimate CAL before QUERY label evaluation.
Historical controls remain immutable and training permissions are explicit.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .terminal_aligned_experiment import evaluator, load, read, sha, dump
from .normality_density_runner import RoleReader
from .alarm_complementarity import confirmed, longest_miss, point_counts
from .hybrid_v1 import evaluation as metrics

HIST='hybrid_native_predictive_cycle_20261113_v1'


def common_cal_bank(z,domain,role):
    """Retain the pre-existing mature CAL axis, irrespective of new history.

    SWaT's frozen native control starts at CAL+512. New H192 inference
    legitimately covers earlier points too, but those are not substituted for
    any of the 512 pre-fixed calibration endpoints. Raw banks remain intact.
    """
    if domain!='SWAT' or role!='CAL':
        return z
    keep=z['index'][0]>=446173
    assert np.array_equal(z['index'][0,keep],np.arange(446173,495000))
    length=z['index'].shape[1]
    return {k:(v[:,keep] if v.ndim>=2 and v.shape[:2]==(1,length) else v)
            for k,v in z.items()}


def table(file, rows):
    import csv
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(file).open('x',newline='',encoding='utf-8')as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def past_mean_and_level(values,loc,H):
    """Simple matched controls; the current observation is excluded from mean."""
    assert (loc>=H).all() and (loc<values.shape[1]).all()
    cs=np.concatenate((np.zeros((len(values),1,values.shape[2])),np.cumsum(values,axis=1,dtype=np.float64)),axis=1)
    batch=np.arange(len(values))[:,None]
    mean=(cs[batch,loc]-cs[batch,loc-H])/H
    current=values[batch,loc]
    return np.mean(np.abs(current-mean),axis=-1),np.mean(np.abs(current),axis=-1)


def new_bank(output,domain,variant,role):
    folder=output/domain/variant/'score'
    receipt=read(folder/'SCORE_RECEIPT.json')
    file=folder/f'{role}.npz'
    assert sha(file)==next(e['sha256']for e in receipt['banks']if e['role']==role)
    z=load(file)
    assert np.array_equal(z['available_index'],z['index']) and np.array_equal(z['max_input_index'],z['index'])
    assert np.all(np.diff(z['index'],axis=1)==1)
    assert np.array_equal(z['raw_score'],z['channel_score'].mean(-1))
    return common_cal_bank(z,domain,role),dict(path=str(file),sha256=sha(file))


def seal(workspace,output,config):
    cfg=read(config);root=output/'evaluation';root.mkdir(exist_ok=False)
    allentries=[]
    for domain in ('TEP','PSM','SWAT'):
        folder=root/domain;folder.mkdir();(folder/'banks').mkdir()
        roles=('CAL','QUERY','NORMAL_TEST')if domain=='TEP'else('CAL','QUERY')
        entries={};sources=[];axes={};ids={}
        for variant,name in [('full','Hybrid_full'),('time_only','Hybrid_time_only'),('local','Hybrid_local'),('identity','Student_identity')]:
            e={}
            for role in roles:
                z,source=new_bank(output,domain,variant,role)
                if role not in axes:
                    axes[role]=z['index'].copy();ids[role]=np.column_stack([z['fault'][:,0],z['run'][:,0]])
                assert np.array_equal(axes[role],z['index'])
                assert np.array_equal(ids[role],np.column_stack([z['fault'][:,0],z['run'][:,0]]))
                e[role]=z['raw_score'] if domain=='TEP'else z['raw_score'][0]
                sources.append(source)
            entries[name]=e
        if domain in ('TEP','PSM'):
            for name,key in [('LEFT_native','all_channel_mean')]+([('LEFT_target9','raw_score')]if domain=='TEP'else []):
                e={}
                for role in roles:
                    f=workspace/'results'/HIST/f'{domain}_native'/'score'/f'{role}.npz';z=load(f)
                    rc=read(f.parent/'SCORE_RECEIPT.json')
                    assert sha(f)==next(r['sha256']for r in rc['outputs']if r['role']==role)
                    assert np.array_equal(z['index'],axes[role])
                    e[role]=z[key]if domain=='TEP'else z[key][0]
                    sources.append(dict(path=str(f),sha256=sha(f),field=key))
                entries[name]=e
        else:
            e={}
            for role in roles:
                f=workspace/'results/left_multievent_mixed_u_20261111_v1/left_score'/f'{role}.npz';z=load(f)
                ix=axes[role][0];positions=ix-z['index'][0]
                assert np.array_equal(z['index'][positions],ix)
                e[role]=z['score'][positions];sources.append(dict(path=str(f),sha256=sha(f)))
            entries['LEFT_native']=e
        if domain=='TEP':
            previous=workspace/'results/hybrid_terminal_aligned_20261115_v1/evaluation/TEP'
            for name in ('Ridge_H','Ridge_C','Simple_native_RidgeC_half','P0','MeanFIT'):
                file=previous/'banks'/f'{name}.npz';z=load(file)
                desc=next(d for d in read(previous/'RAW_SEAL.json')['methods']if d['method']==name)
                assert sha(file)==desc['sha256']
                for role in roles:assert np.array_equal(z[role+'_index'],axes[role][0])
                entries[name]={role:z[role]for role in roles}
                sources.append(dict(path=str(file),sha256=sha(file),permission='target9 residual; same3 observable channels but a different prediction target'))
        # Strong simple current-level and strictly-past mean baselines under
        # exactly this domain's scaler and physical roles, no QUERY fitting.
        rr=RoleReader(workspace,domain,roles)
        mean_history={};fit_level={}
        H=cfg['domains'][domain]['history']
        for role in roles:
            values,runs,faults,offsets=rr.load(role)
            loc=axes[role]-offsets[:,None]-(1 if domain=='TEP'else 0)
            score,level=past_mean_and_level(values,loc,H)
            mean_history[role]=score if domain=='TEP'else score[0]
            fit_level[role]=level if domain=='TEP'else level[0]
        entries[f'HistoryMean{H}_matched']=mean_history
        entries['FIT_level_matched']=fit_level
        selected=None
        if domain=='PSM':selected=np.linspace(106720,132480,512,dtype=np.int64)
        if domain=='SWAT':selected=np.linspace(446173,494999,512,dtype=np.int64)
        descriptors=[]
        for name,e in entries.items():
            cal=e['CAL']if domain=='TEP'else e['CAL'][selected-axes['CAL'][0,0]]
            tau=float(np.quantile(cal,.99,method='higher'))
            file=folder/'banks'/f'{name}.npz'
            extra={role+'_index':ix[0]for role,ix in axes.items()}
            if domain=='TEP':extra.update(QUERY_ids=ids['QUERY'],NORMAL_TEST_ids=ids['NORMAL_TEST'])
            if selected is not None:extra['selected_CAL']=selected
            np.savez_compressed(file,**e,**extra)
            descriptors.append(dict(method=name,path=str(file),sha256=sha(file),threshold=tau))
        protocol=dict(primary='all channels mean joint window NLL for Hybrid; native endpoint scores unchanged',
            normalizer='own original physicalFITonly',source_selection='normal SELECT only; domain max budget fixed',
            baseline_permissions='LEFT target-normal; Ridge_H/C and old simple are target9-specific controls, LEFT_native TEP mean3 primary, LEFT_target9 secondary',
            current_scope='historically exposed development; not blind confirmation',
            threshold='own legitimateCAL q99higher, strict>',selected_CAL=None if selected is None else selected.tolist(),
            score_axis='current endpoint assigned trailing-windowNLL; no observed future',
            long_steady='TEP561..960; PSM/SWAT after512 from original event start',
            source_sha256=sha(__file__))
        dump(folder/'metric_definition.json',protocol)
        dump(folder/'RAW_SEAL.json',dict(status='SEALED_BEFORE_LABEL_EVALUATION',methods=descriptors,sources=sources,
            config_sha256=sha(config),implementation_sha256=sha(__file__)))
        allentries.extend(dict(domain=domain,**d)for d in descriptors)
    dump(root/'ALL_DOMAINS_SEALED.json',dict(status='ALL_FROZEN_BEFORE_LABEL_EVALUATION',methods=allentries,config_sha256=sha(config),code_sha256=sha(__file__)))


def labels(workspace,domain):
    if domain=='SWAT':file=workspace/'preparations/left_multievent_mixed_u_20261111_v1/SWAT_LABEL_METADATA.npz';key='label'
    else:
        original=read(workspace/'preparations/hybrid_left_joint_cycle_20261109_v2/EVALUATION.ACTUAL.json')['truth']['path']
        name=str(original).replace('\\','/')
        anchor='preparations/'if 'preparations/'in name else'results/'
        file=workspace/name[name.index(anchor):];key='full_labels'
    return load(file)[key].astype(np.int8),dict(path=str(file),sha256=sha(file),field=key)


def evaluate_series(workspace,output,domain):
    from sklearn.metrics import average_precision_score,roc_auc_score
    folder=output/'evaluation'/domain;sealed=read(folder/'RAW_SEAL.json')
    truth,labelmeta=labels(workspace,domain)
    rows=[];event_rows=[];alarm_rows=[]
    for desc in sealed['methods']:
        assert sha(desc['path'])==desc['sha256']
        z=load(desc['path']);ix=z['QUERY_index'];score=z['QUERY']
        expected=np.arange(512,87841)if domain=='PSM'else np.arange(608535,944919)
        assert np.array_equal(ix,expected)
        scored=np.zeros(len(truth),bool);scored[ix]=True
        events=metrics._coverage(truth,scored,0)
        strict=score>desc['threshold']
        for stream,alarm in [('strict',strict),('confirmed3',confirmed(ix,strict))]:
            full=np.zeros(len(truth),bool);full[ix]=alarm
            ev,_=metrics._event_view(truth,scored,full,np.arange(len(truth)),events,0,'main_eligible')
            norm,_=metrics._normal(truth,scored,full,0)
            eligible=[e for e in events if e['main_eligible']]
            cov=[];steady=[];tp=n=0;miss=[];delays=[]
            lookup={e['event_id']:e for e in ev['events']}
            for e in events:
                lo,hi=e['local_start'],e['local_end'];bits=full[lo:hi];info=lookup[e['event_id']]
                valid=e['main_eligible'];st=None
                if valid:
                    cov.append(float(bits.mean()));miss.append(longest_miss(bits))
                    if info['delay_points']is not None:delays.append(info['delay_points'])
                    if hi-lo>512:
                        segment=bits[512:];st=float(segment.mean());steady.append(st);tp+=int(segment.sum());n+=len(segment)
                event_rows.append(dict(domain=domain,method=desc['method'],stream=stream,**e,
                    **{k:v for k,v in info.items()if k!='event_id'},
                    coverage=float(bits.mean())if valid else None,
                    longest_miss=longest_miss(bits)if valid else None,steady_coverage=st))
            alarm_rows.extend(dict(method=desc['method'],stream=stream,**a)for a in ev['alarms'])
            rows.append(dict(domain=domain,method=desc['method'],stream=stream,threshold=desc['threshold'],
                AP=float(average_precision_score(truth[ix],score)),ROC=float(roc_auc_score(truth[ix],score)),
                **point_counts(truth,scored,full),**ev['metrics'],**{k:v for k,v in norm.items()if not isinstance(v,list)},
                event_macro_coverage=float(np.mean(cov))if cov else None,steady_macro_coverage=float(np.mean(steady))if steady else None,
                steady_pooled_coverage=tp/n if n else None,long_event_count=len(steady),steady_points=n,
                max_event_miss=max(miss)if miss else None,hit_only_delay_median=float(np.median(delays))if delays else None,
                uncertainty='single seed; historical development; no independent cross-domain confirmation'))
    table(folder/'summary.csv',rows);table(folder/'per_event.csv',event_rows);table(folder/'alarms.csv',alarm_rows)
    dump(folder/'EVALUATION_RECEIPT.json',dict(status='COMPLETED',label=labelmeta,raw_seal_sha256=sha(folder/'RAW_SEAL.json'),methods=len(sealed['methods'])))


def evaluate(workspace,output):
    started=time.time();read(output/'evaluation/ALL_DOMAINS_SEALED.json')
    # Reuse the accepted TEP definitions and run-cluster sensitivity code,
    # including all20faults and the prespecified4/11/14subset.
    m=evaluator(workspace,output);m.evaluate_tep()
    evaluate_series(workspace,output,'PSM');evaluate_series(workspace,output,'SWAT')
    external_psm(workspace,output)
    dump(output/'evaluation/FINAL_EVALUATION_RECEIPT.json',dict(status='COMPLETED',seconds=time.time()-started,
          all_results_retained=True,no_QUERY_parameter_selection=True,implementation_sha256=sha(__file__)))


def external_psm(workspace,output):
    """Ranking on a common axis does not equalize training or offline rights."""
    from sklearn.metrics import average_precision_score,roc_auc_score
    truth,_=labels(workspace,'PSM');folder=output/'evaluation/PSM'
    idx=np.arange(512,87744);rows=[]
    for d in read(folder/'RAW_SEAL.json')['methods']:
        z=load(d['path']);assert np.array_equal(z['QUERY_index'][:len(idx)],idx)
        score=z['QUERY'][:len(idx)]
        rows.append(dict(method=d['method'],AP=float(average_precision_score(truth[idx],score)),ROC=float(roc_auc_score(truth[idx],score)),
            points=len(idx),inference='causal endpoint; Hybrid uses observed trailing-windowjointNLL',
            training='targetFITnormal; ownSELECT/CAL'if d['method']!='Student_identity'else'no model training; targetFITscaler',
            alarm_comparison='own independent targetCAL in full-axis table',source_sha256=d['sha256']))
    source=workspace/'results/hybrid_v1_psm_U_scope2000_fixed_evaluation_20261107_v1'
    for name,permission in [('TimeRadar_official_native','official source pretrained; target pre-test normalization statistics; native offline windows include future relative to some scored timestamps'),
                            ('HistoryMean512_ZS','no target fitting; historical source scales/calibration, strictpast512 mean')]:
        file=source/name/'common_axis_inputs.npz';z=load(file)
        assert np.array_equal(z['index'],idx)and z['valid'].all()
        score=z['score']
        rows.append(dict(method=name,AP=float(average_precision_score(truth[idx],score)),ROC=float(roc_auc_score(truth[idx],score)),
            points=len(idx),inference=permission,training=permission,
            alarm_comparison='ranking only here; distinct historical calibration; TimeRadar excluded from online alarm comparison',source_sha256=sha(file)))
    table(folder/'external_common_axis.csv',rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['seal','evaluate'])
    ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--config',type=Path)
    a=ap.parse_args()
    if a.phase=='seal':
        if a.config is None:ap.error('--config required')
        seal(a.workspace.resolve(),a.output.resolve(),a.config)
    else:evaluate(a.workspace.resolve(),a.output.resolve())


if __name__=='__main__':main()
