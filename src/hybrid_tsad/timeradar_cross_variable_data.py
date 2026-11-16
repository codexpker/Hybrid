"""Fixed public PSM/SMD roles for the native TimeRadar adaptation comparison.

CPU data preparation only. No model, score or QUERY-driven data selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def ident(path):
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def spans(mask):
    delta = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(delta == 1), np.flatnonzero(delta == -1)))


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def run(w):
    start = time.perf_counter()
    out = w/'data/timeradar_cross_variable_20261116_v1'
    out.mkdir(parents=True, exist_ok=False)
    evaldir = out/'evaluation'
    evaldir.mkdir()
    psm = w/'data/PSM.csv'
    assert sha(psm) == '38e446209e82df94b82c11aa43d87e3671da86167ed794c1b7be5eb91fc0e8ac'
    df = pd.read_csv(psm)
    names = df['cols'].unique().tolist()
    assert names == ['feature_'+str(i) for i in range(25)]+['label']
    length = 132481+87841
    assert len(df) == length*26
    assert all(np.all(df['cols'].to_numpy()[i*length:(i+1)*length] == n)
               for i,n in enumerate(names))
    allvalues = np.column_stack([df['data'].to_numpy()[i*length:(i+1)*length] for i in range(25)])
    all_labels = df['data'].to_numpy()[25*length:]
    sources = [('PSM', allvalues[:132481], allvalues[132481:], all_labels[132481:],
                [ident(psm)], dict(train_label_audit_zero=bool((all_labels[:132481] == 0).all()),
                                  source_kind='author processed finite PSM; no new imputation',
                                  mapping='feature_0..feature_24; target train132481, test87841',
                                  source_version='author hosted processed CSV, SHA pinned'))]
    del df, allvalues, all_labels
    # IDs were frozen by lexical completeness rule before any method results.
    for machine in ['machine-1-1', 'machine-1-2', 'machine-1-3']:
        folder = w/'data/event_smd_20261030_v1'
        paths = [folder/(machine+'_'+r+'.txt') for r in ['train','test','test_label']]
        if not paths[0].exists() and machine == 'machine-1-1':
            paths[0] = w/'data/smd_machine_1_1_train_20261029/machine-1-1_train.txt'
        x,y,l = [np.loadtxt(p, delimiter=',') for p in paths]
        assert x.ndim == y.ndim == 2 and x.shape[1] == y.shape[1] == 38
        sources.append((machine,x,y,l,[ident(p) for p in paths],
            dict(source_kind='native ServerMachineDataset public text records',
                 source_version='OmniAnomaly original asset cache; train1-1 pinned7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4',
                 normality='Dataset training-side assumption, not independently certified clean operation')))
    binding = dict(id='timeradar_cross_variable_20261116_v1',seed=2024,window=100,stride=100,copies=10,
        L=1,L_semantics='native L*copy variance +(1-L)*reconstruction criterion; L1 variance only',
        track='strict adaptation comparison: official model/score unchanged, StandardScaler FIT only',
        selection_rule='SMD lexicographically first three complete cached native train/test/label entities; no performance filtering',
        records=[],learner_labels=False,query_selection_by_label=False,
        historical_exposure='All are public development assets with prior exposure; not blind confirmation',
        offline='Full-window point scores and shared batch FRFT order use future observations relative to interior points; not strict online',
        guard_points=100,source_sha256=sha(Path(__file__)))
    evaluation = dict(records=[])
    for name,train,test,label,source,notes in sources:
        assert np.isfinite(train).all() and np.isfinite(test).all()
        assert label.shape == (len(test),) and np.isin(label,[0,1]).all()
        f, s = 3*len(train)//5,4*len(train)//5
        bounds = dict(FIT=[0,f],SELECT=[f+100,s],CAL=[s+100,len(train)])
        scaler = StandardScaler().fit(train[:f])
        folder=out/name
        folder.mkdir()
        np.savez(folder/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,
                 fit_bounds=np.array([0,f]))
        roles=[]
        for role,(lo,hi) in list(bounds.items())+[('QUERY',(0,len(test)))]:
            original = test if role=='QUERY' else train
            x=scaler.transform(original[lo:hi]).astype(np.float32)
            assert np.isfinite(x).all() and len(x)>=100
            path=folder/(role+'.npz')
            np.savez_compressed(path,x=x,index=np.arange(lo,hi,dtype=np.int64))
            roles.append(dict(dataset='PSM' if name=='PSM' else 'SMD',entity=name,
                run='test' if role=='QUERY' else 'train',role=role,
                path=str(path.relative_to(w)).replace('\\','/'),sha256=sha(path),shape=list(x.shape),
                bounds_half_open=[lo,hi],n_windows=len(x)//100,scored_points=(len(x)//100)*100,
                unscored_tail=len(x)%100,fields=['x','index'],format='npz'))
        binding['records'].append(dict(id=name,dataset='PSM' if name=='PSM' else 'SMD',
            channels=train.shape[1],channel_order=['feature_'+str(i) for i in range(train.shape[1])],
            train_points=len(train),query_points=len(test),source=source,notes=notes,
            scaler=dict(path=str((folder/'scaler.npz').relative_to(w)).replace('\\','/'),
                        sha256=sha(folder/'scaler.npz'),fit_bounds=[0,f]),
            score_config=dict(batch_size=64 if name=='PSM' else 32,norm=0 if name=='PSM' else 1,L=1,copies=10),roles=roles))
        covered=len(test)//100*100
        events=[]
        for i,(a,b) in enumerate(spans(label.astype(bool))):
            scored=max(0,min(b,covered)-a)
            events.append(dict(event_id=name+':test:'+str(a)+':'+str(b),start=int(a),end_exclusive=int(b),
                total_points=int(b-a),scored_points=int(scored),coverage_ratio=scored/(b-a),
                eligible_for_complete_event_metric=bool(b<=covered),missing_indices=list(range(max(a,covered),b))))
        lp=evaldir/(name+'_labels.npz')
        np.savez_compressed(lp,index=np.arange(len(label)),label=label.astype(np.int8))
        evaluation['records'].append(dict(id=name,path=str(lp.relative_to(w)).replace('\\','/'),
            sha256=sha(lp),n=len(label),positive_points=int(label.sum()),events=events,
            complete_events=sum(e['eligible_for_complete_event_metric'] for e in events),
            original_event_count=len(events),scored_points=covered,unscored_tail=len(test)-covered,
            label_source='PSM processed label column' if name=='PSM' else 'native test_label.txt'))
    dump(out/'learner_manifest.json',binding)
    dump(evaldir/'evaluation_manifest.json',evaluation)
    prep=w/'preparations/timeradar_cross_variable_20261116_v1'
    prep.mkdir(parents=True,exist_ok=True)
    dump(prep/'data_binding.json',binding)
    dump(prep/'DATA_PREPARATION_RECEIPT.json',dict(status='COMPLETE',CPU_seconds=time.perf_counter()-start,
        GPU_hours=0,model_forward=0,source_file=sha(Path(__file__)),
        learner_manifest_sha256=sha(out/'learner_manifest.json'),
        evaluation_manifest_sha256=sha(evaldir/'evaluation_manifest.json'),
        no_labels_in_learner_files=True,no_query_or_label_based_selection=True))
    print(json.dumps(dict(status='COMPLETE',records=[dict(id=r['id'],roles=r['roles']) for r in binding['records']]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    run(p.parse_args().workspace.resolve())
