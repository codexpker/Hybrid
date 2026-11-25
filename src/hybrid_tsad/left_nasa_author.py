"""Shared native LEFT models over entity-isolated NASA normal training arrays."""
import argparse
import os
import sys
import time
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .left_swat_author import read, dump, sha
from . import native_event_space as old

OUT=Path('results/left_nasa_author_20261123_v1')
PREP=Path('preparations/left_nasa_author_20261123_v1')
REV='3fadb4811797075e233076b9faceb0d8ec59b697'


def raw_paths(w,domain,split,entity):
    return w/(old.MSL if domain=='MSL' else PREP/'raw_smap')/split/(entity+'.npy')


def execute(w, domain, arm=None, *, check_only=False):
    import torch
    from torch.utils.data import Dataset,DataLoader
    from sklearn.preprocessing import StandardScaler
    t0 = time.monotonic()
    if arm is not None:
        from .left_decoder_context_run import (PUBLIC, fingerprint, freeze_public_detection,
                                               public_kind, public_model_choices, sampler_fingerprint)
        check_dir = w/PUBLIC/domain/'observation_check'
        output = check_dir if check_only else w/PUBLIC/domain/arm
    else:
        output = w/OUT/domain
    output.mkdir(parents=True, exist_ok=False)
    random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed_all(2025)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    upstream=w/'preparations/left_baseline_build_20261101_v1/upstream'
    sys.path[:0]=[str(upstream),str(upstream.parent/'deps_thop_transport2_v1/site')]
    import exp.exp_anomaly_detection as official
    window=96 if domain=='MSL' else 192
    ids=old.entity_list(w,domain)
    normal={};inputs=[]
    for e in ids:
        path=raw_paths(w,domain,'train',e);a=np.load(path,allow_pickle=False)
        assert a.ndim==2 and np.isfinite(a[:,0]).all()
        normal[e]=a[:,0].astype(np.float64)[:,None]
        inputs.append(dict(entity=e,train_sha256=sha(path),normal_length=len(a),fit_stop=int(len(a)*.8),validation_stop=len(a)))
    scaler=StandardScaler().fit(np.concatenate(list(normal.values())))
    np.savez(output/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_)
    scaled={e:scaler.transform(x).astype(np.float32) for e,x in normal.items()}
    configs=w/PREP/'configs'
    config=read(configs/domain/'model_configs_Left.json')
    assert config['seq_len']==window and config['enc_in']==1
    compatibility={}
    if domain=='MSL':
        for a,b in [('alpha_casd_l','alpha_ms_l'),('alpha_casd_s','alpha_ms_s')]:
            assert a in config and b not in config
            compatibility[a]=dict(mapped_to=b,value=config[a]);config[b]=config[a]
        # Separate derived config; preserve upstream originals.
        import shutil
        derived=output/'configs'/domain;derived.mkdir(parents=True)
        for name in ['train_configs.json','model_configs_0.json']:
            shutil.copyfile(configs/domain/name,derived/name)
        dump(derived/'model_configs_Left.json',config);configs=derived.parent
    train_config=read(configs/domain/'train_configs.json')
    assert train_config==dict(train_epochs=20,batch_size=128,optim='adam',learning_rate=1e-4,lradj='type1',patience=3)
    if arm is not None:
        previous = w/OUT/domain
        ref = read(previous/'TRAIN.json')
        assert ref['upstream'] == REV and ref['seed'] == 2025 and ref['window'] == window
        assert inputs == ref['inputs'] and compatibility == ref['compatibility']
        assert {n:sha(configs/domain/n) for n in ref['configs']} == ref['configs']
        with np.load(previous/'scaler.npz', allow_pickle=False) as prior:
            for key in ('mean', 'scale', 'var'):
                np.testing.assert_array_equal(getattr(scaler, key+'_'), prior[key])
        kind = public_kind(w, official, arm)
    counts={}
    class Windows(Dataset):
        def __init__(self,role,stride):
            self.windows=[]
            self.roles=[]
            for e,x in scaled.items():
                cut=int(len(x)*.8);start,stop=(0,cut) if role=='train' else (cut,len(x))
                ends=np.arange(start,max(start,stop-window+1),stride,dtype=np.int64)
                self.roles.append(dict(entity=e,start=start,stop=stop,windows=len(ends)))
                self.windows.extend((e,int(a)) for a in ends)
            counts[role]=self.roles
        def __len__(self):return len(self.windows)
        def __getitem__(self,i):
            e,a=self.windows[i]
            return scaled[e][a:a+window],np.zeros((window,1),np.float32)
    losses=[];select=[];updates=0;epoch_lrs=[]
    sampler = []
    base_stop=official.EarlyStopping
    class Stop(base_stop):
        def save_checkpoint(self,val_loss,model,path):
            super().save_checkpoint(val_loss,model,path)
            select.append(dict(epoch=len(losses),val=float(val_loss),step=model.step))
    official.EarlyStopping=Stop
    class Experiment(official.Exp_Anomaly_Detection):
        def _acquire_device(self):return torch.device('cuda:0')
        def _build_model(self):
            return super()._build_model() if arm is None else kind(self.hyper_args)

        def _get_data(self,flag,step=None):
            nonlocal updates
            ds=Windows(flag,1 if flag=='train' else window)
            assert len(ds)>0
            loader=DataLoader(ds,batch_size=128,shuffle=flag=='train',drop_last=False,num_workers=0)
            if flag=='train':updates=len(loader)
            return ds,loader
        def vali(self,loader):
            value = float(super().vali(loader))
            assert np.isfinite(value)
            losses.append(value)
            if arm is not None:
                sampler.append(sampler_fingerprint())
                dump(output/f'epoch{len(losses):02d}.json', dict(epoch=len(losses), validation_loss=value,
                     updates=updates*len(losses), cpu_rng_sha256=sampler[-1]))
            return value
        def _adjust_learning_rate(self,opt,epoch,conf,**kwargs):
            super()._adjust_learning_rate(opt,epoch,conf,**kwargs)
            epoch_lrs.append(dict(epoch=epoch,lr=opt.param_groups[0]['lr']))
    exp=Experiment(SimpleNamespace(data=domain,data_origin='DADA',model_class='Left',gpu=0,use_gpu=True,gpu_type='cuda',use_multi_gpu=False,configs_path=str(configs),model_save_path=str(output/'model'),results_save_path=str(output/'unused')),id=0)
    metadata = {}
    if arm is not None:
        first = next(x for x in scaled.values() if int(len(x)*.8) >= 2*window)
        sample = first[:2*window].reshape(2, window, 1)
        choices = public_model_choices(exp, arm, output, check_dir/'CHECKS.json', sample,
                                       check_only=check_only)
        if check_only:
            return
        metadata.update(choices, sampler_cpu_rng_sha256=sampler, hardware=torch.cuda.get_device_name())
    else:
        # Necessary single normal-window shape/gradient check without state mutation.
        from copy import deepcopy
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
        witness=deepcopy(exp.model)
        witness.train();sample=torch.from_numpy(next(iter(scaled.values()))[:window][None]).cuda()
        loss,_=witness(sample,None,None,None);assert loss.ndim==0 and torch.isfinite(loss)
        loss.backward();assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in witness.parameters())
        del witness
        # Restore the training RNG consumed by the isolated check to the fixed state.
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
    exp.train()
    pt=Path(exp.model_save_path)/'checkpoint.pth'
    assert pt.is_file() and min(losses)==select[-1]['val']
    train_seconds=time.monotonic()-t0
    dump(output/'TRAIN.json',dict(domain=domain,seed=2025,upstream=REV,compatibility=compatibility,project_compatible_implementation=bool(compatibility),entities=len(ids),window=window,inputs=inputs,scaler_sha256=sha(output/'scaler.npz'),scaler_permissions='domain statistics across all admitted official normal training arrays, including normal validation; test excluded',roles=counts,loss_history=losses,selections=select,selected=select[-1],epochs=len(losses),updates=updates*len(losses),updates_per_epoch=updates,final_step=exp.model.step,warmup_steps=exp.model.warmup_steps,ramp_steps=exp.model.ramp_steps,selected_step=select[-1]['step'],ramp_fraction=min(1,max(0,(exp.model.step-exp.model.warmup_steps)/exp.model.ramp_steps)),checkpoint_sha256=sha(pt),worker_sha256=sha(__file__),configs={n:sha(configs/domain/n) for n in ['train_configs.json','model_configs_0.json','model_configs_Left.json']},epoch_lrs=epoch_lrs,train_seconds=train_seconds,device=os.environ.get('CUDA_VISIBLE_DEVICES'),peak_reserved=torch.cuda.max_memory_reserved(),test_values_or_labels_in_training=False,target_official_normal_training_used=True,**metadata))
    tic = time.monotonic()
    banks = []
    scoring_metadata = {}
    if arm is not None:
        lam, state, step = freeze_public_detection(exp.model, arm)
        scoring_metadata.update(detection_modulation=metadata['detection_modulation'], native_lambda=lam)
    else:
        exp.model.eval()
    for entity in ids:
        path=raw_paths(w,domain,'test',entity);raw=np.load(path,allow_pickle=False)
        test=scaler.transform(raw[:,0].astype(np.float64)[:,None]).astype(np.float32)
        stop=len(test)//window*window
        blocks=torch.from_numpy(test[:stop].reshape(-1,window,1));scores=[]
        with torch.inference_mode():
            for a in range(0,len(blocks),128):
                x=blocks[a:a+128].cuda();s=exp.model.infer(x,None,None,None)[0]
                assert s.shape==x.shape and torch.isfinite(s).all();scores.append(s.cpu().numpy())
        values=np.concatenate(scores).reshape(-1)
        bank=output/(entity+'.npz');ix=np.arange(stop)
        np.savez_compressed(bank,score=values,index=ix,available_index=(ix//window+1)*window-1,window_start=ix//window*window)
        banks.append(dict(entity=entity,input_sha256=sha(path),bank_sha256=sha(bank),input_points=len(test),scored_points=stop,omitted_tail=len(test)-stop,window=window,mode='offline full-position nonoverlap native LEFT',CAL_operations='NA: scaler and normal validation may overlap historical project CAL'))
    if arm is not None:
        assert fingerprint(exp.model.state_dict()) == state and exp.model.step == step
        assert exp.model._curriculum_lambda() == lam
        scoring_metadata['state_unchanged'] = True
    dump(output/'SCORES.json',dict(domain=domain,banks=banks,selected_checkpoint_sha256=sha(pt),score_seconds=time.monotonic()-tic,train_seconds=train_seconds,total_seconds=time.monotonic()-t0,device=os.environ.get('CUDA_VISIBLE_DEVICES'),frozen_inference=True,test_labels_read=False,**scoring_metadata))

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('domain', choices=['MSL', 'SMAP'])
    p.add_argument('--arm', choices=['native', 'observation'])
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    if a.check and a.arm != 'observation':
        p.error('--check requires the observation arm')
    execute(Path.cwd(), a.domain, a.arm, check_only=a.check)
