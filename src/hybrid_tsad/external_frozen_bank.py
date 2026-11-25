"""Pinned zero-update external detector banks for the admitted public records."""

import argparse
from pathlib import Path
import time

import numpy as np

from . import native_event_space as nasa
from .left_nasa_author import OUT as NASA_LEFT, raw_paths
from .left_smd28_author import checked_data
from .left_swat_author import dump, read, sha
from .native_event_smd28 import records
from .timercd_fixed import ARMS, detector_from_config, predict_with_cache
from .timercd_fixed_runner import RoleReader, state_hash

OUT = Path('results/native_endpoint_external_20261125_v1')
DADA_ROOT = Path('literature/dada_official_ea172389/source')
DADA_REV = 'ea172389acb466a11a7caac2d18b5ead754aaa43'
DADA_SHA = {
    'config.json': 'f59ba5fbfb8f84be1122058bd6d376c7290962bd6cfce35ad04fd45a1c354f53',
    'configuration_DADA.py': 'af07aa40bd7cab9571c59a613b7fd99498c6dce5a88036cbf23ce4ef83e34be6',
    'modeling_DADA.py': '9c71040d57e52c26613c4a91bcda414dc5ef5533b54acaf12570b94c224c58f4',
    'pytorch_model.bin': '66db3cc568d3df550bdb0aab7cf40c191acc1acd4bfee62b2b63af843369d850',
}
# Window (100), norm flag and batch size are from the pinned author scripts.
DADA_SETUP = {'PSM': (1,32), 'SMD': (0,32), 'MSL': (0,128), 'SMAP': (1,32)}
RCD_PRIOR = Path('results/timercd_fixed_20261117_v1')
RCD_CONFIG = Path('preparations/timercd_fixed_20261117_v1/runtime_v1/configs/timercd-fixed-20261117.json')
RCD_UNI = 'be8906c6a62f16cedd8fe0fe48ef11c4363099633ed6ba8780af819337f7ffaa'
RCD_FILE = 'pretrain_checkpoint_best_uni.pth'
PSM_RAW = Path('data/common_psm_raw_20261029_v1')
PSM_ASSETS = ('f722bce0b19324d85c7b5691efe73453eb56022e867ccc1ba47bb3406787df38',
              '6592518615d690ab153592b6dd0220c424441a0d4ee7bac87c3f89fe942893f5')


def raw_record(w, domain, entity, record):
    if domain == 'SMD':
        train, test = (checked_data(w,record,role) for role in ('train','test'))
        return train,test,[r['sha256'] for r in record['source'][:2]]
    if domain == 'PSM':
        paths = [w/PSM_RAW/name for name in ('prefix.npy','test.npy')]
        assert [sha(path) for path in paths] == list(PSM_ASSETS)
        train,test = (np.load(path,allow_pickle=False) for path in paths)
        assert train.shape==(132481,25) and test.shape==(87841,25)
        return train,test,list(PSM_ASSETS)
    paths = [raw_paths(w,domain,role,entity) for role in ('train','test')]
    ids = [sha(path) for path in paths]
    train,test = (np.load(path,allow_pickle=False)[:,0:1].astype(np.float64) for path in paths)
    old_train = read(w/NASA_LEFT/domain/'TRAIN.json')
    old_score = read(w/NASA_LEFT/domain/'SCORES.json')
    train_entry, = (r for r in old_train['inputs'] if r['entity']==entity)
    test_entry, = (r for r in old_score['banks'] if r['entity']==entity)
    assert ids == [train_entry['train_sha256'], test_entry['input_sha256']]
    if domain == 'SMAP':
        normalized_source = w/nasa.SMAP_QUERY/f'{entity}.npz'
        np.testing.assert_array_equal(test[:,0].astype(np.float32),
                                      nasa.load_npz(normalized_source).astype(np.float32))
    return train,test,ids


def rcd_fit(w, domain, entity, train):
    if domain == 'MSL':
        stop = int(len(train)*.6)
        return train[:stop].astype(np.float32).astype(np.float64),[0,stop],None
    path = w/nasa.SMAP/entity/'FIT.npz'
    fit = nasa.load_npz(path)[:,None]
    np.testing.assert_array_equal(fit.astype(np.float32),train[:len(fit)].astype(np.float32))
    return fit,[0,len(fit)],sha(path)


def cached_rcd(w, entity, count, seal=None):
    if seal is None:
        seal = read(w/RCD_PRIOR/'RAW_BANK_SEAL.json')
    assert seal['status']=='COMPLETED'
    result = {}
    for arm in ARMS:
        key = f'{entity}/{arm}/QUERY.npz'
        item, = (r for r in seal['banks'] if r['relative']==key)
        path = w/RCD_PRIOR/key
        assert sha(path)==item['sha256']
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['index'],np.arange(count))
            np.testing.assert_array_equal(z['logit'],z['head_logits'][:,1]-z['head_logits'][:,0])
            assert z['valid'].all()
            values = {'score':z['logit'],'index':z['index'],'available_index':z['available_index']}
        result[arm] = (values,item['sha256'])
    return result


def prepare(w):
    start = time.perf_counter()
    root = w/OUT
    root.mkdir(parents=True,exist_ok=True)
    rcd_config = read(w/RCD_CONFIG)
    assert rcd_config.get('variant','multi')=='multi'
    assert all(sha(w/DADA_ROOT/'DADA'/f)==digest for f,digest in DADA_SHA.items())
    rcd_uni = (Path(rcd_config['checkpoint']).parent/RCD_FILE).as_posix()
    assert sha(w/rcd_uni)==RCD_UNI
    seal = read(w/RCD_PRIOR/'RAW_BANK_SEAL.json')
    official = {r['id']:r for r in records(w)}
    jobs = [('PSM','PSM',None)] + [('SMD',e,r) for e,r in sorted(official.items())]
    jobs += [(d,e,None) for d in ('MSL','SMAP') for e in nasa.entity_list(w,d)]
    rows = []
    for domain,entity,record in jobs:
        train,test,ids = raw_record(w,domain,entity,record)
        assert np.isfinite(train).all() and np.isfinite(test).all()
        left_w = 96 if domain=='MSL' else 192
        n = len(test)
        row = dict(domain=domain,entity=entity,channels=test.shape[1],normal_points=len(train),test_points=n,
                   raw_assets_sha256=ids,causal_axis=[99,n],
                   offline_axis=[99,min(n//100*100,n//left_w*left_w)])
        if domain in ('PSM','SMD'):
            raw_old, old_index = RoleReader(w,rcd_config,entity,['QUERY']).load('QUERY')
            np.testing.assert_array_equal(raw_old,test)
            np.testing.assert_array_equal(old_index,np.arange(n))
            banks = cached_rcd(w,entity,n,seal)
            row['TimeRCD_QUERY_sha256'] = {name:digest for name,(_,digest) in banks.items()}
        else:
            fit,bounds,fit_hash = rcd_fit(w,domain,entity,train)
            mean,std = fit.mean(axis=0),fit.std(axis=0,ddof=0)
            row.update(TimeRCD_fit=bounds, TimeRCD_fit_asset_sha256=fit_hash,
                       TimeRCD_fit_mean=mean.tolist(),TimeRCD_fit_std=np.where(std==0,1e-8,std).tolist())
        if domain in ('PSM','SMAP'):
            row['history_axis'] = [512,87744 if domain=='PSM' else n]
        assert row['offline_axis'][1]>99
        rows.append(row)
    assert len(rows)==109
    dump(root/'AXES_AND_INPUTS.json',dict(records=rows,DADA=dict(code=DADA_REV,sha256=DADA_SHA,
         window=100,stride=100,setup=DADA_SETUP,mask_seed_per_entity=2024,
         scaler='per-entity StandardScaler over full released normal array, including validation if applicable',
         score='native infer, ten complementary copies, variance across copies averaged over input channels'),
         TimeRCD=dict(code=rcd_config['author_commit'],multi_sha256=rcd_config['checkpoint_sha256'],
         uni_sha256=RCD_UNI,uni_checkpoint=rcd_uni,hf_revision=rcd_config['checkpoint_HF_revision'],
         config_sha256=sha(w/RCD_CONFIG),window=5000,batch_size=1,
         score='raw abnormal-class minus normal-class channel-mean logit',
         main_statistics='existing raw FIT only, std==0 replaced by 1e-8 as in the fixed TimeRCD protocol',
         diagnostic_statistics='separate full QUERY role as in official API'),
         metric=dict(VUS_windowSize=100,VUS_thre=250),seed_alignment='candidate original seeds 2025/26/27',
         labels_read=False,source_sha256=sha(__file__),preparation_seconds=time.perf_counter()-start))


def dada_model(w,device):
    import transformers
    from transformers import AutoModel
    location = w/DADA_ROOT/'DADA'
    assert all(sha(location/f)==digest for f,digest in DADA_SHA.items())
    model, info = AutoModel.from_pretrained(location,trust_remote_code=True,local_files_only=True,
                                           output_loading_info=True)
    assert not any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),info
    return model.to(device).eval(),transformers.__version__


def dada_scores(model, train, test, policy, device, *, parity=False):
    import torch
    from sklearn.preprocessing import StandardScaler
    norm,batch = policy
    scaler = StandardScaler().fit(train)
    stop = len(test)//100*100
    x = scaler.transform(test[:stop]).astype(np.float32).reshape(-1,100,test.shape[1])
    np.random.seed(2024)
    result = []
    with torch.inference_mode():
        for i in range(0,len(x),batch):
            t = torch.from_numpy(x[i:i+batch]).to(device)
            if parity and i==0:
                rng = np.random.get_state()
                official = model.infer(t,norm=norm)
                np.random.set_state(rng)
                copies = model.model.infer(t,norm=norm)
                torch.testing.assert_close(official,copies.var(dim=0,correction=1).mean(dim=-1),rtol=0,atol=0)
                np.random.set_state(rng)
            result.append(model.infer(t,norm=norm).cpu().numpy())
    score = np.concatenate(result).reshape(-1)
    assert score.shape==(stop,) and np.isfinite(score).all()
    index = np.arange(stop)
    return dict(score=score,index=index,available_index=index//100*100+99,
                window_start_index=index//100*100),dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist())


def save_bank(root,name,values,receipt):
    root.mkdir(parents=True,exist_ok=True)
    p = root/f'{name}.npz'
    assert not p.exists()
    np.savez_compressed(p,**values)
    dump(p.with_suffix('.json'), {**receipt,'bank_sha256':sha(p)})


def score(w, detector, lane, lanes):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = w/OUT
    path = root/'AXES_AND_INPUTS.json'
    binding = read(path)
    assert sha(__file__)==binding['source_sha256']
    assert not binding['labels_read']
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    if detector == 'DADA':
        model, version = dada_model(w,device)
        base = state_hash(model)
        all_rows = sorted(binding['records'],key=lambda row:row['test_points']*row['channels'],reverse=True)
    else:
        cfg = read(w/RCD_CONFIG)
        assert sha(w/RCD_CONFIG)==binding['TimeRCD']['config_sha256']
        cfg.update(variant='uni',checkpoint=binding['TimeRCD']['uni_checkpoint'],checkpoint_sha256=RCD_UNI)
        model = detector_from_config(w,cfg,1,device)
        base = state_hash(model._tester.model)
        all_rows = [r for r in binding['records'] if r['domain'] in ('MSL','SMAP')]
    smd = {r['id']:r for r in records(w)}
    items = []
    for idx,row in enumerate(all_rows[lane::lanes]):
        d,e = row['domain'],row['entity']
        train,test,ids = raw_record(w,d,e,smd.get(e))
        assert ids==row['raw_assets_sha256'] and len(test)==row['test_points']
        target = root/d/e
        torch.cuda.synchronize()
        start = time.perf_counter()
        meta = dict(domain=d, entity=e, raw_assets_sha256=ids, protocol_sha256=sha(path), labels_read=False)
        if detector=='DADA':
            if idx==0:
                dada_scores(model,train,train[:100],DADA_SETUP[d],device,parity=True)
            values,statistics = dada_scores(model,train,test,DADA_SETUP[d],device)
            torch.cuda.synchronize()
            save_bank(target,'DADA',values,dict(**meta,seconds=time.perf_counter()-start,stats=statistics,
                     checkpoint_sha256=DADA_SHA['pytorch_model.bin'],normal_witness=idx==0))
        else:
            fit,bounds,fit_hash = rcd_fit(w,d,e,train)
            assert bounds==row['TimeRCD_fit'] and fit_hash==row['TimeRCD_fit_asset_sha256']
            mean = np.asarray(row['TimeRCD_fit_mean'])
            std = np.asarray(row['TimeRCD_fit_std'])
            assert np.array_equal(mean,fit.mean(axis=0))
            if idx==0:
                sample = fit[:min(len(fit),5000)]
                prob,logit = model.predict(sample,return_logits=True)
                actual,_ = predict_with_cache(model,sample,np.arange(len(sample)),ARMS[0])
                np.testing.assert_array_equal(actual['prob'],prob)
                np.testing.assert_array_equal(actual['logit'],logit)
            for arm in ARMS:
                values,stats = predict_with_cache(model,test,np.arange(len(test)),arm,mean,std)
                torch.cuda.synchronize()
                save_bank(target,'TimeRCD_'+arm,values,dict(**meta,seconds=time.perf_counter()-start,
                     stats=stats,checkpoint_sha256=RCD_UNI,normal_witness=idx==0))
                start = time.perf_counter()
        items.append([d,e])
        print(detector,lane,d,e,round(time.perf_counter()-start,2),flush=True)
    assert base == state_hash(model if detector=='DADA' else model._tester.model)
    dump(root/f'{detector}_LANE_{lane}.json',dict(records=items,model_unchanged=True,zero_updates=True,
        seconds=time.perf_counter()-begin, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        library=version if detector=='DADA' else 'pinned TimeRCD', protocol_sha256=sha(path)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','dada','timercd'))
    parser.add_argument('--lane',type=int,default=0)
    parser.add_argument('--lanes',type=int,default=1)
    args = parser.parse_args()
    w = Path.cwd()
    if args.action=='prepare':
        prepare(w)
    else:
        score(w,'DADA' if args.action=='dada' else 'TimeRCD',args.lane,args.lanes)
