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


SMD_COMMIT='7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4'
SMD_OLD=('machine-1-1','machine-1-2','machine-1-3')
SMD_CACHED_NEW=('machine-1-4','machine-1-5','machine-2-1','machine-3-1','machine-3-2','machine-3-3')
SMD_TREE_SHA=dict(train='a5e3ac1a28ea06e9b17bc8ec533d5d37ce6cdb981228be29bc603cc377a00fcd',
    test='c4d436f648bea5ddb9a290f08157bcc190afc33e9a68cd7d2d7ac6d9d1162899',
    test_label='4592c42458186a958c381e166101d51aa755bececf93aed3fecbf5a7f2a6496d')


def git_blob(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def smd_official_inventory(w):
    """Use the previously acquired exact official Git trees, not model scores."""
    cache=w/'data/stage2_u_machine31_original_20261104'
    root=json.loads((cache/'ROOT.API.json').read_text(encoding='utf8'))
    dataset=json.loads((cache/'DATASET.API.json').read_text(encoding='utf8'))
    assert next(r['sha']for r in root if r['name']=='ServerMachineDataset')==dataset['sha']=='c2eb46aa3b754177c2312f3a0d6bd85cfec7a2b4'
    trees={}
    for role,expected in SMD_TREE_SHA.items():
        path=cache/f'{role}.API.json';assert sha(path)==expected
        tree=json.loads(path.read_text(encoding='utf8'))
        assert next(r['sha']for r in dataset['tree']if r['path']==role)==tree['sha']
        assert not tree['truncated']and len(tree['tree'])==28
        trees[role]={r['path'][:-4]:r for r in tree['tree']}
    names=sorted(trees['train'])
    assert all(sorted(t)==names for t in trees.values())
    assert [sum(n.startswith(f'machine-{g}-')for n in names)for g in (1,2,3)]==[8,9,11]
    # Cached public raw files only. Processed arrays/aggregate CSV cannot qualify.
    candidates={}
    for path in sorted((w/'data').glob('**/machine-*_*.txt')):
        for role in ('test_label','train','test'):
            suffix='_'+role+'.txt'
            if path.name.endswith(suffix):
                name=path.name[:-len(suffix)]
                if name in trees[role]:candidates.setdefault((name,role),[]).append(path)
                break
    records=[]
    for name in names:
        sources=[]
        for role in ('train','test','test_label'):
            expected=trees[role][name];found=None
            for candidate in candidates.get((name,role),[]):
                if candidate.stat().st_size==expected['size']and git_blob(candidate)==expected['sha']:
                    found=candidate;break
            sources.append(dict(role=role,tree_path=f'ServerMachineDataset/{role}/{name}.txt',
                expected_bytes=expected['size'],expected_git_blob_sha=expected['sha'],
                path=str(found)if found else None,sha256=sha(found)if found else None,
                status='CACHED_VERIFIED'if found else'MISSING',
                url=f'https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/{SMD_COMMIT}/ServerMachineDataset/{role}/{name}.txt'))
        records.append(dict(id=name,sources=sources,status='READY'if all(s['path']for s in sources)else'MISSING',
            historical_exposure='SMD aggregate was previously evaluated; not a blind independent domain',
            prior_native_raw_cached=all(s['path']for s in sources)))
    return dict(source_repo='https://github.com/NetManAIOps/OmniAnomaly',commit=SMD_COMMIT,
        entity_rule='All 28 complete official train/test/test_label names, no result-based selection',
        group_counts=[8,9,11],records=records,
        tree_inputs=[dict(role=r,path=str(cache/f'{r}.API.json'),sha256=h)for r,h in SMD_TREE_SHA.items()])


def acquire_smd_missing(w,inventory):
    """Bounded same-pin missing-text acquisition, preserving every failure."""
    from concurrent.futures import ThreadPoolExecutor
    import urllib.request
    destination=w/'data/smd_full_original_7fb0_20261116';destination.mkdir(exist_ok=True)
    missing=[(r['id'],s)for r in inventory['records']for s in r['sources']if s['path']is None]
    total=sum(s['expected_bytes']for _,s in missing)
    if total>332933646:raise PermissionError('Missing-source bytes exceed the approved fixed 57-file scope')
    def one(item):
        name,s=item;p=destination/f"{name}_{s['role']}.txt";part=p.with_suffix('.txt.part')
        began=time.monotonic();row=dict(entity=name,**s)
        try:
            if not p.exists():
                request=urllib.request.Request(s['url'],headers={'User-Agent':'Hybrid-fixed-public-SMD-data'})
                with urllib.request.urlopen(request,timeout=60)as response,part.open('xb')as handle:
                    count=0
                    while block:=response.read(1024*1024):
                        count+=len(block)
                        if count>s['expected_bytes']:raise ValueError('Source exceeds pinned per-file byte bound')
                        handle.write(block)
                if count!=s['expected_bytes']:raise ValueError('Truncated source')
                if git_blob(part)!=s['expected_git_blob_sha']:raise ValueError('Official Git blob identity mismatch')
                part.rename(p)
            if p.stat().st_size!=s['expected_bytes']or git_blob(p)!=s['expected_git_blob_sha']:raise ValueError('Existing source is not the pinned blob')
            row.update(status='ACQUIRED_VERIFIED',path=str(p),sha256=sha(p),actual_bytes=p.stat().st_size)
            s.update(path=str(p),sha256=sha(p),status='ACQUIRED_VERIFIED')
        except Exception as error:row.update(status='FAILED_MISSING',error=repr(error),partial_path=str(part)if part.exists()else None)
        row['seconds']=time.monotonic()-began
        return row
    with ThreadPoolExecutor(max_workers=4)as pool:rows=list(pool.map(one,missing))
    dump(destination/'ACQUISITION.json',dict(commit=SMD_COMMIT,expected_files=len(missing),expected_bytes=total,
        per_file_limit='exact official Git tree size',total_limit_bytes=332933646,records=rows,
        no_fallback_source=True,no_other_dataset=True))
    for record in inventory['records']:record['status']='READY'if all(s['path']for s in record['sources'])else'MISSING'
    return inventory


def _smd_record(w,source,out):
    name=source['id'];paths={r['role']:Path(r['path'])for r in source['sources']}
    train=np.loadtxt(paths['train'],delimiter=',');test=np.loadtxt(paths['test'],delimiter=',')
    label=np.loadtxt(paths['test_label'],delimiter=',')
    if (train.ndim!=2 or test.ndim!=2 or train.shape[1]!=38 or test.shape[1]!=38
            or not np.isfinite(train).all()or not np.isfinite(test).all()
            or label.shape!=(len(test),)or not np.isin(label,[0,1]).all()):raise ValueError('Native array shape/finite/label audit failed')
    f,s=3*len(train)//5,4*len(train)//5;scaler=StandardScaler().fit(train[:f]);folder=out/name
    folder.mkdir(exist_ok=False)
    np.savez(folder/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,fit_bounds=np.array([0,f]))
    roles=[]
    for role,(lo,hi)in dict(FIT=(0,f),SELECT=(f+100,s),CAL=(s+100,len(train)),QUERY=(0,len(test))).items():
        x=scaler.transform((test if role=='QUERY'else train)[lo:hi]).astype(np.float32)
        if len(x)<100 or not np.isfinite(x).all():raise ValueError('Insufficient/invalid role')
        p=folder/f'{role}.npz';np.savez_compressed(p,x=x,index=np.arange(lo,hi,dtype=np.int64))
        roles.append(dict(dataset='SMD',entity=name,run='test'if role=='QUERY'else'train',role=role,
            path=p.relative_to(w).as_posix(),sha256=sha(p),shape=list(x.shape),bounds_half_open=[lo,hi],
            n_windows=len(x)//100,scored_points=len(x)//100*100,unscored_tail=len(x)%100,fields=['x','index'],format='npz'))
    record=dict(id=name,dataset='SMD',channels=38,channel_order=[f'feature_{i}'for i in range(38)],
        train_points=len(train),query_points=len(test),source=[ident(paths[r])for r in ('train','test','test_label')],
        notes=dict(source_kind='original official native public SMD',source_commit=SMD_COMMIT,
            normality='Dataset training-side assumption, not independently certified clean operation'),
        scaler=dict(path=(folder/'scaler.npz').relative_to(w).as_posix(),sha256=sha(folder/'scaler.npz'),fit_bounds=[0,f]),
        score_config=dict(batch_size=32,norm=1,L=1,copies=10),roles=roles)
    covered=len(test)//100*100;events=[]
    for a,b in spans(label.astype(bool)):
        count=max(0,min(b,covered)-a)
        events.append(dict(event_id=f'{name}:test:{a}:{b}',start=int(a),end_exclusive=int(b),total_points=int(b-a),
            scored_points=int(count),coverage_ratio=count/(b-a),eligible_for_complete_event_metric=bool(b<=covered),
            missing_indices=list(range(max(a,covered),b))))
    lp=out/'evaluation'/f'{name}_labels.npz';np.savez_compressed(lp,index=np.arange(len(label)),label=label.astype(np.int8))
    evaluation=dict(id=name,path=lp.relative_to(w).as_posix(),sha256=sha(lp),n=len(label),positive_points=int(label.sum()),
        events=events,complete_events=sum(e['eligible_for_complete_event_metric']for e in events),original_event_count=len(events),
        scored_points=covered,unscored_tail=len(test)-covered,label_source='official native test_label text, pinned Git blob')
    return record,evaluation


def full_smd(w,*,wave,acquire_missing=False):
    """Two immutable acquisition waves; original three physical identities reused."""
    started=time.monotonic();prep=w/'preparations/timeradar_smd_full_20261116_v1';prep.mkdir(parents=True,exist_ok=True)
    out=w/'data/timeradar_smd_full_20261116_v1';out.mkdir(exist_ok=True);(out/'evaluation').mkdir(exist_ok=True)
    inventory=smd_official_inventory(w)
    if acquire_missing:inventory=acquire_smd_missing(w,inventory)
    basepath=w/'preparations/timeradar_cross_variable_20261116_v1/data_binding.json'
    assert sha(basepath)=='d10cd11eca48f570f0e4a122f133cd8be60792c743ba68e0554f293ec4e82149'
    base=json.loads(basepath.read_text(encoding='utf8'))
    chosen=list(SMD_CACHED_NEW)if wave=='cached'else[r['id']for r in inventory['records']if r['id']not in SMD_OLD+SMD_CACHED_NEW]
    records=[];evaluations=[]
    for source in inventory['records']:
        if source['id']not in chosen:continue
        if source['status']!='READY':continue
        try:
            r,e=_smd_record(w,source,out);records.append(r);evaluations.append(e)
            source['role_status']='READY'
        except Exception as error:source.update(role_status='INVALID',role_error=repr(error))
    name='wave1'if wave=='cached'else'wave2'
    binding={k:v for k,v in base.items()if k not in ('records','id','selection_rule','source_sha256')}
    binding.update(id=f'timeradar_smd_full_20261116_v1_{name}',records=records,
        selection_rule='Official complete 28 entities; immutable cached6/missing19 acquisition waves, no performance selection',
        source_sha256=sha(__file__),original_three_binding_sha256=sha(basepath),official_commit=SMD_COMMIT)
    bp=prep/f'data_binding_{name}.json';dump(bp,binding)
    ep=prep/f'evaluation_manifest_{name}.json';dump(ep,dict(records=evaluations))
    dump(prep/f'official_inventory_{name}.json',inventory)
    dump(prep/f'DATA_RECEIPT_{name}.json',dict(status='READY'if len(records)==len(chosen)else'INCOMPLETE',
        expected_records=chosen,ready_records=[r['id']for r in records],binding_sha256=sha(bp),evaluation_manifest_sha256=sha(ep),
        seconds=time.monotonic()-started,GPU_hours=0,no_model_forward=True,no_query_performance_used=True))
    if wave=='missing':
        oldeval=json.loads((w/'data/timeradar_cross_variable_20261116_v1/evaluation/evaluation_manifest.json').read_text(encoding='utf8'))
        previous=json.loads((prep/'data_binding_wave1.json').read_text(encoding='utf8'))
        preveval=json.loads((prep/'evaluation_manifest_wave1.json').read_text(encoding='utf8'))
        allrecords=[r for r in base['records']if r['id']in SMD_OLD]+previous['records']+records
        alleval=[r for r in oldeval['records']if r['id']in SMD_OLD]+preveval['records']+evaluations
        dump(prep/'data_binding_full28.json',dict(binding,id='timeradar_smd_full_20261116_v1',records=sorted(allrecords,key=lambda r:r['id'])))
        dump(prep/'evaluation_manifest_full28.json',dict(records=sorted(alleval,key=lambda r:r['id'])))
        lookup={r['id']:r for r in allrecords};registry=[]
        for source in inventory['records']:
            rid=source['id'];r=lookup.get(rid)
            registry.append(dict(id=rid,status='READY'if r else source.get('role_status',source['status']),
                wave='reuse_old3'if rid in SMD_OLD else'wave1'if rid in SMD_CACHED_NEW else'wave2',
                data_record=r,source_audit=source))
        dump(prep/'full28_registry.json',dict(expected_entities=28,ready_entities=len(allrecords),records=registry,
            original_three_reused_without_reprocessing=True,not_blind_confirmation=True))
    print(json.dumps(dict(wave=name,ready=[r['id']for r in records],binding=str(bp),sha256=sha(bp))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--full-smd',action='store_true');p.add_argument('--wave',choices=['cached','missing'],default='cached')
    p.add_argument('--acquire-missing',action='store_true')
    a=p.parse_args()
    if a.full_smd:full_smd(a.workspace.resolve(),wave=a.wave,acquire_missing=a.acquire_missing)
    else:run(a.workspace.resolve())
