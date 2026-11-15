"""Frozen-score normal-state transfer validation, with separate CAL/check roles.

No model training, weight selection, score modification or QUERY-based fitting.
Historical raw scores and original evaluation outputs remain immutable.
"""
import argparse
from pathlib import Path
import time
import numpy as np

from .normality_density_evaluation import read,load,dump,sha,table,evaluate_series,evaluator
from .alarm_complementarity import confirmed
from .engine.events import binary_events

HISTORY='hybrid_normality_density_20261115_v1'


def subsample(indices,n=512):
    assert len(indices)>=n
    return np.asarray(indices)[np.linspace(0,len(indices)-1,n,dtype=np.int64)]


def normal_metrics(index,score,tau):
    index=np.asarray(index);score=np.asarray(score)
    if index.ndim==1:index=index[None];score=score[None]
    assert index.shape==score.shape and np.isfinite(score).all()
    result=[]
    for stream in ('strict','confirmed3'):
        fp=frag=total=0;per=[]
        for i,(ix,s)in enumerate(zip(index,score)):
            alarm=s>tau
            if stream=='confirmed3':alarm=confirmed(ix,alarm)
            # Count alarm fragments within genuinely contiguous observed blocks.
            cuts=np.r_[0,np.flatnonzero(np.diff(ix)!=1)+1,len(ix)]
            nfrag=sum(len(binary_events(alarm[a:b]))for a,b in zip(cuts[:-1],cuts[1:]))
            fp+=int(alarm.sum());frag+=nfrag;total+=len(ix)
            per.append(dict(run_row=i,points=len(ix),FP=int(alarm.sum()),fragments=nfrag,FPR=float(alarm.mean())))
        result.append(dict(stream=stream,normal_points=total,normal_FP=fp,normal_FPR=fp/total,
            normal_fragments=frag,normal_fragments_per_1000=1000*frag/total,
            empirical_budget_met=bool(fp/total<=.01),per_run=per))
    return result


def seal(workspace,output,config):
    cfg=read(config);root=output/'evaluation';root.mkdir(exist_ok=False)
    checks=[];descriptors=[]
    oldroot=workspace/'results'/HISTORY/'evaluation'
    for domain in ('PSM','SWAT','TEP'):
        dest=root/domain;dest.mkdir();(dest/'banks').mkdir()
        old=read(oldroot/domain/'RAW_SEAL.json');entries=[]
        for method in cfg['methods']:
            src=next(r for r in old['methods']if r['method']==method)
            assert sha(src['path'])==src['sha256'];z=load(src['path'])
            if domain=='PSM':
                calix=subsample(np.arange(106720,119504));checkix=np.arange(120240,132481)
                assert calix[-1]<119504 and checkix[0]-191>=120048
                cal=z['CAL'][calix-z['CAL_index'][0]];check=z['CAL'][checkix-z['CAL_index'][0]]
            elif domain=='TEP':
                assert z['CAL'].shape==(20,400)
                cal=z['CAL'][:10];check=z['CAL'][10:]
                calix=checkix=np.arange(101,501)
            else:
                # Executor provides only preregistered normal DONOR endpoints.
                f=output/'SWAT'/method/'NORMAL_BANK.npz';donor=load(f)
                receipt=read(f.parent/'RECEIPT.json');assert sha(f)==receipt['bank_sha256']
                manifest=load(output/'SWAT/NORMAL_ROLE_INDEX.npz')
                assert np.array_equal(donor['CAL_index'],manifest['CAL_index'])
                assert np.array_equal(donor['CHECK_index'],manifest['CHECK_index'])
                calix=subsample(donor['CAL_index']);checkix=donor['CHECK_index']
                cal=donor['CAL_score'][np.searchsorted(donor['CAL_index'],calix)];check=donor['CHECK_score']
            tau=float(np.quantile(cal,.99,method='higher'))
            file=dest/'banks'/f'{method}.npz'
            fields={k:v for k,v in z.items()if not k.startswith('CAL')and k!='selected_CAL'}
            np.savez_compressed(file,**fields,CAL=cal,CAL_index=calix,NORMAL_CHECK=check,NORMAL_CHECK_index=checkix)
            entry=dict(method=method,path=str(file),sha256=sha(file),threshold=tau,
                       historical_threshold=src['threshold'],historical_bank_sha256=src['sha256'])
            entries.append(entry);descriptors.append(dict(domain=domain,**entry))
            for threshold_name,threshold in [('new_CAL',tau),('historical_CAL',src['threshold'])]:
                for row in normal_metrics(np.broadcast_to(checkix,check.shape),check,threshold):
                    per=row.pop('per_run');checks.append(dict(domain=domain,method=method,calibration=threshold_name,threshold=threshold,**row))
                    table(dest/f'{method}_{threshold_name}_normal_runs_{row["stream"]}.csv',per)
            assert np.array_equal(load(file)['QUERY'],z['QUERY'])
        dump(dest/'RAW_SEAL.json',dict(status='NEW_CAL_SEALED_WITHOUT_QUERY_LABELS',methods=entries,
            source_seal_sha256=sha(oldroot/domain/'RAW_SEAL.json'),config_sha256=sha(config)))
    table(root/'normal_check.csv',checks)
    dump(root/'ALL_DOMAINS_SEALED.json',dict(methods=descriptors,config_sha256=sha(config),source_sha256=sha(__file__),QUERY_scores_identical=True))


def evaluate(workspace,output):
    import pandas as pd
    started=time.time();read(output/'evaluation/ALL_DOMAINS_SEALED.json')
    m=evaluator(workspace,output);m.evaluate_tep()
    for domain in ('PSM','SWAT'):evaluate_series(workspace,output,domain)
    checks=pd.read_csv(output/'evaluation/normal_check.csv');gates=[]
    cfg=read(workspace/'preparations/hybrid_normal_cal_transfer_20261115_v1/FIXED_VALIDATION.json')
    for domain in ('PSM','SWAT','TEP'):
        frame=pd.read_csv(output/'evaluation'/domain/'summary.csv')
        if domain=='TEP':frame=frame[frame.scope=='all_faults']
        for method in cfg['methods']:
            ck=checks[(checks.domain==domain)&(checks.method==method)&(checks.stream=='strict')&(checks.calibration=='new_CAL')].iloc[0]
            qr=frame[(frame.method==method)&(frame.stream=='strict')].iloc[0]
            gates.append(dict(domain=domain,method=method,normal_CHECK_FPR=ck.normal_FPR,
                downstream_normal_FPR=qr.normal_FPR,CHECK_budget_met=bool(ck.normal_FPR<=.01),
                downstream_budget_met=bool(qr.normal_FPR<=.01),both_budget_met=bool(ck.normal_FPR<=.01 and qr.normal_FPR<=.01),
                interpretation='empirical held-normal budget, not independent population certification; QUERY normals used only posthoc'))
    table(output/'evaluation/budget_gate.csv',gates)
    dump(output/'evaluation/FINAL_RECEIPT.json',dict(status='COMPLETED',CPU_seconds=time.time()-started,
         model_updates=0,threshold_search=False,QUERY_rescoring=False,source_sha256=sha(__file__)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['seal','evaluate'])
    ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--config',type=Path)
    a=ap.parse_args()
    if a.phase=='seal':seal(a.workspace.resolve(),a.output.resolve(),a.config.resolve())
    else:evaluate(a.workspace.resolve(),a.output.resolve())


if __name__=='__main__':main()
