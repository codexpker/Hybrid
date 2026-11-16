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

from .timeradar_cross_variable_evaluation import sha, _dump


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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();produce(a.workspace,a.output)
