"""Label-free CPU production of the fixed existing HistoryMean512 formula.

The raw formula is reused from hybrid_v1.baselines.baseline_record. Historical
source median/log1p units are deliberately not inherited; new own-role CAL is
used later. No model, fitting, label decoding, or performance selection here.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .timeradar_cross_variable_evaluation import (sha, _dump, _load_bank, cal_threshold,
    evaluate_arrays, _save_method, _flat, _write_csv, ARMS)


def history_mean512(values):
    values=np.asarray(values)
    if values.ndim!=2 or len(values)<=512 or not np.isfinite(values).all():
        raise ValueError('Finite [time,channels], with >512 role-local points required')
    values=values.astype(np.float64)
    windows=np.lib.stride_tricks.sliding_window_view(values,512,axis=0)
    result=np.empty(len(values)-512,np.float64)
    for begin in range(0,len(result),128):
        stop=min(begin+128,len(result))
        past=windows[begin:stop]
        mean=past.mean(axis=-1)
        variance=np.square(past-mean[...,None]).mean(axis=-1)
        current=values[begin+512:stop+512]
        result[begin:stop]=np.square((mean-current)/np.sqrt(variance+1e-5)).mean(axis=-1)
    if not np.isfinite(result).all():raise ValueError('Nonfinite fixed baseline score')
    return result


def produce(workspace,output):
    w,output=Path(workspace),Path(output)
    binding=w/'preparations/timeradar_cross_variable_20261116_v1/data_binding.json'
    meta=json.loads(binding.read_text(encoding='utf8'))
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();receipts=[]
    for record in meta['records']:
        folder=output/record['id']/'history_mean512'/'score';folder.mkdir(parents=True)
        roles=[];banks=[]
        for spec in record['roles']:
            if spec['role'] not in ('CAL','QUERY'):continue
            source=w/spec['path']
            if sha(source)!=spec['sha256']:raise PermissionError('Fixed role changed')
            with np.load(source,allow_pickle=False) as z:
                if set(z.files)!={'x','index'}:raise ValueError('x/index-only role required')
                x,index=z['x'].copy(),z['index'].copy()
            score=history_mean512(x);index=index[512:]
            path=folder/(spec['role']+'.npz')
            np.savez_compressed(path,index=index,score=score,available_index=index,max_input_index=index)
            banks.append(dict(role=spec['role'],path=str(path),sha256=sha(path),points=len(index)))
            roles.append(dict(role=spec['role'],path=str(source),sha256=sha(source),fields_read=['x','index']))
        receipt=dict(status='COMPLETED',record=record['id'],arm='history_mean512',banks=banks,roles=roles,
            data_binding_sha256=sha(binding),source_sha256=sha(__file__),
            formula='mean_c(((mean(x[t-512:t])-x[t])/sqrt(population_var(x[t-512:t])+1e-5))**2)',
            formula_reference='hybrid_v1/baselines.py::baseline_record(method=history_mean512)',
            history=512,score_units='raw_r_pred; no historical source median/log1p mapping',
            scoring='strictly past512 prediction, arrived current residual; role-local warmup512 UNKNOWN',
            no_labels_decoded=True,new_training=False,model_forward=False,GPU_hours=0)
        _dump(folder/'SCORE_RECEIPT.json',receipt);receipts.append(receipt)
    _dump(output/'SEAL.json',dict(status='COMPLETED_SCORE_ONLY_PENDING_MODEL_BANK_SEAL',
        records=len(receipts),banks=sum(len(r['banks'])for r in receipts),seconds=time.monotonic()-started,
        labels_read=False,GPU_hours=0,source_sha256=sha(__file__),receipts=receipts))
    return receipts


def compare_common(workspace,model_results,baseline_results,output):
    """Full-seal evaluation on fixed per-record [512, native-scored-end)."""
    w,mroot,broot,output=map(Path,(workspace,model_results,baseline_results,output))
    binding=w/'preparations/timeradar_cross_variable_20261116_v1/data_binding.json'
    meta=json.loads(binding.read_text(encoding='utf8'))
    evalpath=w/'data/timeradar_cross_variable_20261116_v1/evaluation/evaluation_manifest.json'
    evalmeta={r['id']:r for r in json.loads(evalpath.read_text(encoding='utf8'))['records']}
    # Gate all method-record score seals before reading any new QUERY labels.
    for r in meta['records']:
        for arm in ARMS:
            folder=mroot/r['id']/arm/'score'
            receipt=json.loads((folder/'SCORE_RECEIPT.json').read_text())
            if receipt['status']!='COMPLETED' or not all((folder/f'{role}.npz').is_file()for role in ('CAL','QUERY')):
                raise FileNotFoundError('All 32 model score banks must be sealed first')
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for record in meta['records']:
        rid=record['id'];end=record['query_points']//100*100
        common=np.arange(512,end,dtype=np.int64)
        label_spec=evalmeta[rid];label_path=w/label_spec['path']
        if sha(label_path)!=label_spec['sha256']:raise PermissionError('Fixed evaluation labels changed')
        with np.load(label_path,allow_pickle=False)as z:
            labels=z['label'].copy()
            if len(labels)!=record['query_points'] or not np.array_equal(z['index'],np.arange(len(labels))):
                raise ValueError('Complete original label axis required')
        rolemap={r['role']:r for r in record['roles']}
        for name in ARMS+('history_mean512',):
            root=broot if name=='history_mean512'else mroot
            folder=root/rid/name/'score';rp=folder/'SCORE_RECEIPT.json'
            receipt=json.loads(rp.read_text())
            if receipt['record']!=rid or receipt['arm']!=name or receipt['status']!='COMPLETED':
                raise ValueError('Completed score receipt record/arm identity required')
            receipt_roles={r['role']:r for r in receipt['roles']}
            for role in ('CAL','QUERY'):
                if receipt_roles[role]['sha256']!=rolemap[role]['sha256']:
                    raise PermissionError('Score receipt uses different physical data roles')
            banks={Path(x['path']).name:x['sha256']for x in receipt['banks']}
            for role in ('CAL','QUERY'):
                if sha(folder/f'{role}.npz')!=banks[f'{role}.npz']:raise PermissionError('Score receipt hash mismatch')
            _,cal,_=_load_bank(folder/'CAL.npz');ix,score,available=_load_bank(folder/'QUERY.npz')
            pos=np.searchsorted(ix,common)
            if np.any(pos>=len(ix))or not np.array_equal(ix[pos],common):raise ValueError('Incomplete fixed comparison axis')
            online=name=='history_mean512';tau=cal_threshold(cal)
            result=evaluate_arrays(common,score[pos],labels,tau,available_index=available[pos],online=online)
            result['CAL_points']=len(cal);result['CAL_exceedance_rate']=float(np.mean(cal>tau))
            result['permissions']=('FIT-only data scaler; own independent CAL; strictly past512'if online else
                                   'FIT-only data scaler; own independent CAL; native offline full-window/batch scorer')
            sources={str(p):sha(p)for p in (folder/'CAL.npz',folder/'QUERY.npz',rp,label_path,binding,evalpath)}
            _save_method(output/rid/name,result,common,score[pos],labels,available[pos],sources)
            row=_flat(rid,name,result);row['common_start']=512;row['common_end_exclusive']=end;rows.append(row)
    _write_csv(output/'metrics.csv',rows)
    _dump(output/'RESULT.json',dict(status='COMPLETED',rows=rows,GPU_hours=0,
        fixed_common_rule='per-record [512,floor(original_query_length/100)*100)',
        warnings=['TimeRadar/Hybrid offline; H512 causal; no equal online delay claim',
                  'Own-role CAL .99 higher with strict>; no QUERY threshold fitting']))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluate-common',action='store_true');p.add_argument('--model-results',type=Path)
    p.add_argument('--baseline-results',type=Path)
    a=p.parse_args()
    if a.evaluate_common:
        if a.model_results is None or a.baseline_results is None:p.error('Both score roots required')
        compare_common(a.workspace,a.model_results,a.baseline_results,a.output)
    else:produce(a.workspace,a.output)
