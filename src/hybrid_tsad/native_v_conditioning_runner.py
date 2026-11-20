"""Role-isolated shared native-V cache and six fixed source learners.

All native calls are serial B1 within a process. No target adaptation or labels.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from .native_v_conditioning import (SOURCE_SHA,WEIGHT_SHA,load_native,fixed_native_rng,actual_mask_bits,
    state_hashes,build_predictor,pinball_per_sample,conditional_score,raw_residual_score,sha)

EXPERIMENT='native_v_conditioning_20261119_v1'
def read(p):return json.loads(Path(p).read_text())
def dump(p,r):
    with Path(p).open('x')as f:json.dump(r,f,indent=2,allow_nan=False)
def seed(n=2025):
    random.seed(n);np.random.seed(n);torch.manual_seed(n)
    if torch.cuda.is_initialized():torch.cuda.manual_seed(n)
def setup(cap=20):
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(cap/45,0);seed();return torch.device('cuda:0')
def source_path(w):return w/'preparations/timeradar_cross_variable_20261116_v1/runtime/official/TimeRadar.py'


def prepare(w,root):
    root.mkdir(parents=True,exist_ok=False)
    for domain,channels in [('SWAT',31),('PSM',25)]:
        d=root/domain;d.mkdir();oldroot=w/'results/tf_conditional_energy_20261119_v1'/domain
        old=read(oldroot/'TASK.json');scaler=oldroot/'SCALER.npz';assert sha(scaler)==old['scaler_sha256']
        shutil.copyfile(scaler,d/'SCALER.npz');roles={k:old['roles'][k]for k in ['SOURCE_FIT','SOURCE_SELECT','CAL','QUERY']}
        if domain=='SWAT':
            mapping=w/'preparations/left_multievent_mixed_u_20261111_v1/OFFICIAL_31_CHANNEL_MAPPING.json'
            columns=sorted([x for x in read(mapping)['mapping']if not x['removed']],key=lambda x:x['retained_index'])
            names=[x['raw_name']for x in columns];order_source=dict(relative=str(mapping.relative_to(w)),sha256=sha(mapping))
        else:names=[f'original_array_column_{i}'for i in range(channels)];order_source=dict(description='Published25 numeric array columns in original stored order; no invented semantic aliases')
        index={}
        for role in ['SOURCE_FIT','SOURCE_SELECT']:
            lo,hi=roles[role];index[role]=np.arange(lo+512,hi,16,dtype=np.int64)
        index['CAL']=np.asarray(old['cal_points'],dtype=np.int64)
        index['QUERY']=np.arange(*old['score_axes']['QUERY'],dtype=np.int64)
        expected=[24718,6156,512,336321]if domain=='SWAT'else[6592,1625,512,64728]
        assert [len(index[k])for k in ['SOURCE_FIT','SOURCE_SELECT','CAL','QUERY']]==expected
        for role,times in index.items():
            lo,hi=roles[role];assert np.all(np.diff(times)>0)and int(times.min())-512>=lo and int(times.max())<hi
            np.save(d/f'{role}_INDEX.npy',times,allow_pickle=False)
        # Same physical record must have disjoint role footprints; role never
        # enters the identity key and cannot conceal real overlap.
        recs={role:old['data']['source_train'if role.startswith('SOURCE')else'target_test']for role in roles}
        for i,r1 in enumerate(roles):
            for r2 in list(roles)[i+1:]:
                if recs[r1]['record_id']==recs[r2]['record_id']:
                    assert roles[r1][1]<=roles[r2][0]or roles[r2][1]<=roles[r1][0]
        fit=len(index['SOURCE_FIT']);epochs=[np.random.default_rng(2025)]
        plan=np.stack([epochs[0].permutation(fit)for _ in range(20)])
        np.savez_compressed(d/'SOURCE_PERMUTATIONS.npz',positions=plan)
        dump(d/'TASK.json',dict(experiment=EXPERIMENT,domain=domain,entity=domain+f'_{channels}_channel_benchmark',channels=channels,
            data=old['data'],roles=roles,channel_order=names,channel_order_source=order_source,scaler_sha256=sha(d/'SCALER.npz'),
            indexes={k:dict(relative=str((d/f'{k}_INDEX.npy').relative_to(w)),sha256=sha(d/f'{k}_INDEX.npy'),count=len(v),
                record_id=recs[k]['record_id'],first=int(v[0]),last=int(v[-1]))for k,v in index.items()},
            cache_identity=['domain','entity','record_id','absolute_time_index'],role_is_permission_not_identity=True,
            source_permutation_sha256=sha(d/'SOURCE_PERMUTATIONS.npz'),seed=2025,history=512,native_window=100,
            source_quantile_method='linear',source_quantiles=[.05,.5,.95],source_sV_floor=1e-12,score_width_floor=1e-3,
            head_init_B_C='zero final weight; global FIT V/sV quantile bias FP32; stable sort permits ties',
            D_head_init='native nn.Linear.reset_parameters default, fixed seed2025; only encoder matches C tensorwise',
            D_score_precision='savedFP32 prediction/actual promoted FP64 before per-channel square and channel mean',
            native=dict(source_commit='4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2',source_sha256=SOURCE_SHA,checkpoint_sha256=WEIGHT_SHA,
                norm=0,copies=10,mask_mode='c',mask_seed=84551,B1_only=True,criterion=None,L=1,terminal=99,correction=1),
            training=dict(methods=['B','C','D'],batch=128,Adam_lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,
                precision='FP32',epochs=20,max_updates=4000,actual_max_updates=min(20*((fit+127)//128),4000),
                include_step0=True,SELECT_every_epoch=True,selection='earliest strictly minimum full normal SOURCE_SELECT mean task loss',
                sampler='all FIT once/epoch without replacement, same fixed20 permutations for B/C/D, retain last batch'),
            cal_rule=dict(quantile=.99,method='higher',strictly_greater=True,exact_original512=True),
            labels_read=False,raw_record_integrity='reuse verified immutable source bindings; numerical access isolated to explicitly materialized roles'))
    dump(root/'CONFIG.json',dict(experiment=EXPERIMENT,base_commit='1be5e0001ae80b787302c2fc62f58860e101cd11',
        methods=['A','B','C','D'],domains=['SWAT','PSM'],seed=2025,no_R=True,no_extra_domains_or_seeds=True,
        roles=['SOURCE_FIT','SOURCE_SELECT','CAL','QUERY'],source_only_training=True,normal_fixture_reuse=True))


def role_values(w,d,task,role):
    if role not in task['roles']:raise PermissionError(role)
    rec=task['data']['source_train'if role.startswith('SOURCE')else'target_test']
    raw=np.load(w/rec['relative'],mmap_mode='r');assert list(raw.shape)==rec['shape']and raw.shape[1]==task['channels']
    lo,hi=task['roles'][role]
    actual=np.array(raw[lo:hi],dtype=np.float64,copy=True)
    with np.load(d/'SCALER.npz',allow_pickle=False)as z:values=np.asarray((actual-z['mean'])/z['std'],dtype=np.float32)
    assert np.isfinite(values).all()and sha(d/'SCALER.npz')==task['scaler_sha256']
    return values,actual,lo,rec


def role_index(w,d,task,role):
    rec=task['indexes'][role];path=w/rec['relative'];assert sha(path)==rec['sha256']
    return np.load(path,allow_pickle=False)


def cache(w,root,domain,role,part=0,parts=1,native_checkpoint=None,freeze_name='ALL_SIX_MODELS_FROZEN.json',metadata=None):
    # CAL/QUERY values are unavailable to this entry until all6 models freeze.
    if role in ['CAL','QUERY']:assert read(root/freeze_name)['status']=='COMPLETED'
    d=root/domain;task=read(d/'TASK.json');out=root/'cache'/domain/role/f'part{part}';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();device=setup(30 if role in ['CAL','QUERY'] else 20)
    native=load_native(w,device)
    actual_checkpoint_sha=WEIGHT_SHA
    checkpoint_binding={}
    if native_checkpoint is not None:
        official_state=state_hashes(native)
        saved=torch.load(native_checkpoint,map_location=device,weights_only=True)
        native.load_state_dict(saved['model'],strict=True);native.requires_grad_(False);native.eval()
        actual_checkpoint_sha=sha(native_checkpoint)
        loaded=state_hashes(native)
        assert all(loaded[k]==value for k,value in official_state.items()if not k.startswith('decoder.'))
        checkpoint_binding=dict(actual_endpoint_sha256=actual_checkpoint_sha,official_initial_checkpoint_sha256=WEIGHT_SHA,
            frozen_except_decoder_verified=True,native_state_hashes=loaded)
    before=state_hashes(native);values,actual,lo,record=role_values(w,d,task,role)
    rng_before=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone())
    index=np.array_split(role_index(w,d,task,role),parts)[part];n=len(index);c=task['channels']
    V=np.empty(n,dtype=np.float32);reconstruction=np.empty((n,10,c),dtype=np.float32);orders=np.empty(n,dtype=np.float32)
    raw_current=np.empty((n,c),dtype=np.float64);normalized_current=np.empty((n,c),dtype=np.float32)
    bits=actual_mask_bits(c,device);np.savez_compressed(out/'MASK_BITS.npz',mask=bits)
    captured=[]
    hook=native.net.register_forward_hook(lambda module,args,result:captured.append(result.detach()))
    reused=0;full=0;fixtures=[];old=w/'results/timeradar_native_v_protocol_20261119_v1';gate=read(old/'NATIVE_V_NORMAL_GATE.json')
    assert gate['source_sha256']==SOURCE_SHA and gate['checkpoint_sha256']==WEIGHT_SHA
    with torch.no_grad():
        for i,t in enumerate(index):
            relative=int(t)-lo;assert relative>=512 and relative<len(values)
            x=values[relative-99:relative+1];input=torch.from_numpy(x.copy()).to(device)[None];captured.clear()
            reusable=native_checkpoint is None and role=='SOURCE_FIT'and int(t)in [512,1024,2048]
            if reusable:
                entry=next(v for v in gate['domains'][domain]['points']if v['t']==int(t));p=old/entry['fixture_relative']
                assert sha(p)==entry['fixture_sha256']and gate['domains'][domain]['scaler_sha256']==task['scaler_sha256']
                with np.load(p,allow_pickle=False)as z:
                    assert np.array_equal(z['input'],x);V[i]=z['native_V_terminal'];reconstruction[i]=z['copies'][:,99]
                # Complete only the missing scalar order, never redo cached V.
                native.net(input.permute(0,2,1).reshape(c,100).mean(0));reused+=1
                fixtures.append(dict(t=int(t),fixture_sha256=sha(p),reused=['V','10_terminal_copies'],completed=['native_net_order_only']))
            else:
                with fixed_native_rng(device):
                    _,copies=native(input,norm=0,mask_mode='c',copies=10)
                    score=native.cal_anomaly_score(input,copies,anomaly_criterion=None,L=1)
                V[i]=float(score[0,99]);reconstruction[i]=copies[:,0,99].cpu().numpy();full+=1
            assert len(captured)==1;orders[i]=float(captured[0].reshape(-1)[0])
            raw_current[i]=actual[relative];normalized_current[i]=values[relative]
            if(i+1)%10000==0:print(json.dumps(dict(event='CACHE_PROGRESS',domain=domain,role=role,part=part,done=i+1,total=n)),flush=True)
    hook.remove();assert before==state_hashes(native)and all(p.grad is None and not p.requires_grad for p in native.parameters())
    assert torch.equal(rng_before[0],torch.get_rng_state())and torch.equal(rng_before[1],torch.cuda.get_rng_state(device))
    assert np.isfinite(V).all()and np.all(V>=0)and np.isfinite(reconstruction).all()
    reference=reconstruction.astype(np.float64).var(1,ddof=1).mean(-1)
    assert np.allclose(V,reference,rtol=1e-5,atol=1e-7)
    canonical=dict(canonical_score=V.astype(np.float64),feature_start=index-99,feature_end_exclusive=index+1)if native_checkpoint is not None else{}
    legacy_history=dict(history_start=index-512,history_end_exclusive=index)if native_checkpoint is None else{}
    path=out/'BANK.npz';np.savez_compressed(path,absolute_time_index=index,available_index=index,V=V,reconstruction_terminal=reconstruction,
        actual_order=orders,current_raw=raw_current,current_normalized=normalized_current,
        window_start=index-99,window_end_exclusive=index+1,**legacy_history,**canonical)
    dump(out/'RECEIPT.json',dict(status='COMPLETED',domain=domain,entity=task['entity'],record_id=record['record_id'],role=role,part=part,parts=parts,
        relative=str(path.relative_to(w)),sha256=sha(path),count=n,index_sha256=hashlib.sha256(index.tobytes()).hexdigest(),
        scaler_sha256=task['scaler_sha256'],source_sha256=SOURCE_SHA,checkpoint_sha256=actual_checkpoint_sha,channel_order=task['channel_order'],
        mask_seed=84551,mask_shape=list(bits.shape),mask_bits_sha256=hashlib.sha256(bits.tobytes()).hexdigest(),mask_file_sha256=sha(out/'MASK_BITS.npz'),
        mask_version='torch.randCUDA84551 first5C20 +complement; reshape10,C,20; torch2.8.0cu128 A40',
        full_native_forward_count=full,reused_native_V_count=reused,partial_fixture_completion=fixtures,
        native_immutable=True,CPU_CUDA_RNG_restored=True,labels_read=False,B1_only=True,epoch_or_optimizer=False,
        elapsed_GPU_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),**checkpoint_binding,**(metadata or {})))
    print(json.dumps(dict(event='CACHE_COMPLETED',domain=domain,role=role,part=part,count=n,reused=reused)),flush=True)


def merge_cache(w,root,domain,role,metadata=None):
    d=root/domain;task=read(d/'TASK.json');folder=root/'cache'/domain/role
    paths=sorted(folder.glob('part*/RECEIPT.json'),key=lambda p:int(p.parent.name[4:]));receipts=[read(p)for p in paths]
    assert receipts and len(receipts)==receipts[0]['parts']
    arrays={};mask_hash=receipts[0]['mask_bits_sha256']
    for r in receipts:
        assert r['status']=='COMPLETED'and r['record_id']==task['indexes'][role]['record_id']and r['mask_bits_sha256']==mask_hash
        p=w/r['relative'];assert sha(p)==r['sha256']
        with np.load(p,allow_pickle=False)as z:
            for k in z.files:arrays.setdefault(k,[]).append(z[k].copy())
    combined={k:np.concatenate(v)for k,v in arrays.items()};assert np.array_equal(combined['absolute_time_index'],role_index(w,d,task,role))
    bank=folder/'BANK.npz';assert not bank.exists();np.savez_compressed(bank,**combined)
    dump(folder/'SEALED.json',dict(status='COMPLETED',domain=domain,entity=task['entity'],record_id=receipts[0]['record_id'],role=role,
        relative=str(bank.relative_to(w)),sha256=sha(bank),count=len(combined['V']),parts=[dict(relative=str(p.relative_to(w)),sha256=sha(p))for p in paths],
        mask_bits_sha256=mask_hash,scaler_sha256=task['scaler_sha256'],labels_read=False,**(metadata or {})))
    if role=='SOURCE_FIT':
        physical=np.quantile(combined['V'].astype(np.float64),[.05,.5,.95],method='linear');sV=max(float(physical[2]),1e-12)
        bias=(physical/sV).astype(np.float32)
        dump(d/'QUANTILE_SCALE.json',dict(s_V=sV,FIT_quantiles_physical=physical.tolist(),q_bias_f32=bias.tolist(),source_quantile_method='linear',
            source_fit_cache_sha256=sha(bank),point_count=len(combined['V']),floor=1e-12))


def load_cache(w,root,domain,role):
    seal=read(root/'cache'/domain/role/'SEALED.json');path=w/seal['relative'];assert sha(path)==seal['sha256']
    with np.load(path,allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
    return data,seal


def histories(values,lo,endpoints,device):
    indexes=np.asarray(endpoints,dtype=np.int64)-lo
    assert indexes.min()>=512 and indexes.max()<len(values)
    return torch.from_numpy(np.stack([values[i-512:i]for i in indexes])).to(device)


def model_task_loss(model,method,h,cache,positions,sV,device):
    prediction=model(h)
    if method in ['B','C']:
        target=torch.tensor(cache['V'][positions].astype(np.float64)/sV,dtype=torch.float32,device=device)
        per_sample=pinball_per_sample(prediction,target)
    else:
        target=torch.from_numpy(cache['current_normalized'][positions]).to(device)
        per_sample=(prediction-target).square().mean(-1)
    return per_sample,prediction,target


def validate(model,method,values,lo,cache,sV,device):
    modes=[(m,m.training)for m in model.modules()];model.eval();total=0.;count=0
    with torch.no_grad():
        for start in range(0,len(cache['V']),128):
            pos=np.arange(start,min(start+128,len(cache['V'])),dtype=np.int64)
            h=histories(values,lo,cache['absolute_time_index'][pos],device)
            losses,_,_=model_task_loss(model,method,h,cache,pos,sV,device)
            assert torch.isfinite(losses).all();total+=float(losses.double().sum());count+=len(pos)
    for m,mode in modes:m.training=mode
    return total/count


def gates(w,root,domain):
    out=root/domain/'ENGINEERING_GATE';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();device=setup()
    d=root/domain;task=read(d/'TASK.json');scale=read(d/'QUANTILE_SCALE.json');cache_data,_=load_cache(w,root,domain,'SOURCE_FIT')
    values,actual,lo,_=role_values(w,d,task,'SOURCE_FIT');pos=np.arange(128,dtype=np.int64)
    h=histories(values,lo,cache_data['absolute_time_index'][pos],device)
    target=torch.from_numpy(cache_data['current_normalized'][pos]).to(device)
    for j,t in enumerate(cache_data['absolute_time_index'][pos]):
        assert np.array_equal(h[j].cpu().numpy(),values[int(t)-lo-512:int(t)-lo])
        assert np.array_equal(target[j].cpu().numpy(),values[int(t)-lo])
    rows={};encoder_hashes={}
    for method in ['B','C','D']:
        seed();model=build_predictor(method,task['channels'],w,scale['q_bias_f32']).to(device)
        if method in ['C','D']:encoder_hashes[method]=state_hashes(model.encoder)
        torch.cuda.synchronize();tt=time.monotonic();losses,pred,yt=model_task_loss(model,method,h,cache_data,pos,scale['s_V'],device)
        loss=losses.mean();loss.backward();torch.cuda.synchronize();seconds=time.monotonic()-tt
        assert torch.isfinite(loss)and all(p.grad is None or torch.isfinite(p.grad).all()for p in model.parameters())
        record=dict(parameters=sum(p.numel()for p in model.parameters()),B128_forward_backward_seconds=seconds,loss=float(loss.detach()))
        if method in ['B','C']:
            expected=torch.tensor(scale['q_bias_f32'],device=device).expand(128,3)
            assert torch.equal(pred,expected)
            body=model.body if method=='B'else model.encoder
            assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in body.parameters())
            record['zero_head_step0_encoder_zero_gradient_expected']=True
            model.zero_grad(set_to_none=True)
            # Engineering-only nonzero head confirms the real encoder gradient
            # path; this mutated/discarded gate model is never used for training.
            with torch.no_grad():model.head.weight.normal_(0,.01)
            newloss,_,_=model_task_loss(model,method,h,cache_data,pos,scale['s_V'],device);newloss.mean().backward()
            assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in body.parameters())
            record['nonzero_head_encoder_gradient_nonzero']=True
        else:
            assert torch.equal(yt,target)
            manual=(pred-target).square().mean(-1);torch.testing.assert_close(losses,manual,rtol=0,atol=0)
            record['target_is_actual_current_normalized']=True
        record['peak_allocated_bytes']=torch.cuda.max_memory_allocated();rows[method]=record
        del model
    assert encoder_hashes['C']==encoder_hashes['D']
    bias=np.asarray(scale['q_bias_f32'],dtype=np.float32)
    constant=np.tile(bias,(len(cache_data['V']),1));score=conditional_score(cache_data['V'],constant,scale['s_V'])['canonical_score']
    assert np.array_equal(np.argsort(cache_data['V'],kind='stable'),np.argsort(score,kind='stable'))
    assert np.all(np.isfinite(score))
    dump(out/'GATE.json',dict(status='PASS_ENGINEERING_ONLY',models=rows,source_only=True,no_CAL_QUERY_opened=True,
        C_D_encoder_initial_state_equal=True,C_D_encoder_initial_hashes=encoder_hashes['C'],
        source_step0_monotone_order_equal_A=True,q_bias_f32=scale['q_bias_f32'],s_V=scale['s_V'],
        current_excluded_from_history=True,discarded_gate_models_not_training=True,
        elapsed_GPU_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    print(json.dumps(dict(event='GATES_COMPLETED',domain=domain,status='PASS_ENGINEERING_ONLY')),flush=True)


def train(w,root,domain,method):
    d=root/domain;task=read(d/'TASK.json');assert read(d/'ENGINEERING_GATE/GATE.json')['status']=='PASS_ENGINEERING_ONLY'
    out=root/'train'/domain/method;out.mkdir(parents=True,exist_ok=False);started=time.monotonic();device=setup();scale=read(d/'QUANTILE_SCALE.json')
    fit,_=load_cache(w,root,domain,'SOURCE_FIT');select,_=load_cache(w,root,domain,'SOURCE_SELECT')
    fit_values,_,fit_lo,_=role_values(w,d,task,'SOURCE_FIT');val_values,_,val_lo,_=role_values(w,d,task,'SOURCE_SELECT')
    permutation_path=d/'SOURCE_PERMUTATIONS.npz';assert sha(permutation_path)==task['source_permutation_sha256']
    with np.load(permutation_path)as z:permutations=z['positions'].copy()
    assert permutations.shape==(20,len(fit['V']))and all(np.array_equal(np.sort(p),np.arange(len(p)))for p in permutations)
    seed();model=build_predictor(method,task['channels'],w,scale['q_bias_f32']).to(device);initial=state_hashes(model)
    encoder_initial=state_hashes(model.encoder)if method in ['C','D']else None
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4);trace=[];selection=[];updates=0;exposures=0;best=float('inf');picked=None
    def choose(epoch):
        nonlocal best,picked
        loss=validate(model,method,val_values,val_lo,select,scale['s_V'],device)
        selection.append(dict(epoch=epoch,updates=updates,loss=loss,points=len(select['V'])))
        if loss<best:
            best=loss;picked=epoch
            torch.save(dict(model=model.state_dict(),epoch=epoch,updates=updates,loss=loss),out/'SELECTED.pth')
    choose(0)
    for epoch in range(20):
        model.train()
        for start in range(0,len(fit['V']),128):
            positions=permutations[epoch,start:start+128]
            h=histories(fit_values,fit_lo,fit['absolute_time_index'][positions],device)
            optimizer.zero_grad(set_to_none=True)
            loss_vector,_,_=model_task_loss(model,method,h,fit,positions,scale['s_V'],device)
            assert torch.isfinite(loss_vector).all();loss=loss_vector.mean();loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all()for p in model.parameters())
            grad=float(sum(p.grad.square().sum()for p in model.parameters()if p.grad is not None).sqrt())
            optimizer.step();updates+=1;exposures+=len(positions)
            trace.append(dict(update=updates,epoch=epoch+1,batch=len(positions),loss=float(loss.detach()),gradient_norm=grad))
            if updates>=4000:break
        choose(epoch+1)
        print(json.dumps(dict(event='EPOCH_COMPLETED',domain=domain,method=method,epoch=epoch+1,updates=updates,SELECT_loss=selection[-1]['loss'],selected_epoch=picked)),flush=True)
        if updates>=4000:break
    assert updates==task['training']['actual_max_updates']
    torch.save(dict(model=model.state_dict(),epoch=epoch+1,updates=updates,loss=trace[-1]['loss']),out/'ENDPOINT.pth')
    torch.save(optimizer.state_dict(),out/'OPTIMIZER.pth');dump(out/'TRACE.json',trace);dump(out/'SELECT.json',selection)
    selected=out/'SELECTED.pth'
    dump(out/'TRAIN_RECEIPT.json',dict(status='COMPLETED',domain=domain,method=method,seed=2025,updates=updates,epochs=epoch+1,
        exposure=exposures,selected_epoch=picked,selected_loss=best,checkpoint_relative=str(selected.relative_to(w)),checkpoint_sha256=sha(selected),
        endpoint_sha256=sha(out/'ENDPOINT.pth'),optimizer_sha256=sha(out/'OPTIMIZER.pth'),initial_state_hashes=initial,
        encoder_initial_state_hashes=encoder_initial,parameters=sum(p.numel()for p in model.parameters()),
        SOURCE_FIT_cache_sha256=read(root/'cache'/domain/'SOURCE_FIT/SEALED.json')['sha256'],
        SOURCE_SELECT_cache_sha256=read(root/'cache'/domain/'SOURCE_SELECT/SEALED.json')['sha256'],
        scaler_sha256=task['scaler_sha256'],source_permutations_sha256=sha(permutation_path),
        quantile_scale_sha256=sha(d/'QUANTILE_SCALE.json'),s_V=scale['s_V'],q_bias_f32=scale['q_bias_f32'],
        roles_read=['SOURCE_FIT','SOURCE_SELECT'],labels_read=False,CAL_QUERY_opened=False,
        elapsed_GPU_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    print(json.dumps(dict(event='TRAINING_COMPLETED',domain=domain,method=method,updates=updates,selected_epoch=picked)),flush=True)


def freeze_trainings(w,root):
    rows=[]
    for domain in ['SWAT','PSM']:
        for method in ['B','C','D']:
            folder=root/'train'/domain/method;p=folder/'TRAIN_RECEIPT.json';receipt=read(p)
            checkpoint=w/receipt['checkpoint_relative'];assert sha(checkpoint)==receipt['checkpoint_sha256']
            assert receipt['status']=='COMPLETED'and receipt['roles_read']==['SOURCE_FIT','SOURCE_SELECT']and not receipt['CAL_QUERY_opened']
            rows.append(dict(domain=domain,method=method,checkpoint_relative=str(checkpoint.relative_to(w)),
                checkpoint_sha256=sha(checkpoint),train_receipt_relative=str(p.relative_to(w)),train_receipt_sha256=sha(p)))
    dump(root/'ALL_SIX_MODELS_FROZEN.json',dict(status='COMPLETED',trainings=rows,labels_read=False,no_CAL_QUERY_opened=True))


def readout(w,root,domain,method,role):
    frozen=read(root/'ALL_SIX_MODELS_FROZEN.json');assert frozen['status']=='COMPLETED'
    d=root/domain;task=read(d/'TASK.json');out=root/'banks'/domain/method/role;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();data,cache_receipt=load_cache(w,root,domain,role)
    axis=data['absolute_time_index'];arrays=dict(absolute_time_index=axis,available_index=axis,V=data['V'],
        actual_raw=data['current_raw'],actual_normalized=data['current_normalized'],
        history_start=axis-512,history_end_exclusive=axis,response_window_start=axis-99,response_window_end_exclusive=axis+1)
    selected=None;before=None;device=None
    if method=='A':arrays['canonical_score']=data['V'].astype(np.float64)
    else:
        device=setup(30);scale=read(d/'QUANTILE_SCALE.json');values,_,lo,_=role_values(w,d,task,role)
        selected=next(r for r in frozen['trainings']if r['domain']==domain and r['method']==method)
        checkpoint=w/selected['checkpoint_relative'];assert sha(checkpoint)==selected['checkpoint_sha256']
        seed();model=build_predictor(method,task['channels'],w,scale['q_bias_f32']).to(device)
        saved=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(saved['model'],strict=True);model.eval()
        before=state_hashes(model);rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone());predictions=[]
        with torch.no_grad():
            for start in range(0,len(axis),128):
                h=histories(values,lo,axis[start:start+128],device);predictions.append(model(h).cpu().numpy())
        prediction=np.concatenate(predictions).astype(np.float32,copy=False)
        assert before==state_hashes(model)and torch.equal(rng[0],torch.get_rng_state())and torch.equal(rng[1],torch.cuda.get_rng_state(device))
        assert np.isfinite(prediction).all()
        if method in ['B','C']:
            arrays.update(conditional_score(data['V'],prediction,scale['s_V']));arrays['q_model_f32']=prediction
            arrays['s_V']=np.asarray(scale['s_V'],dtype=np.float64)
        else:
            residual,score=raw_residual_score(prediction,data['current_normalized'])
            arrays.update(yhat_normalized=prediction,residual_per_channel=residual,canonical_score=score)
    assert np.isfinite(arrays['canonical_score']).all()and len(arrays['canonical_score'])==len(axis)
    bank=out/'BANK.npz';np.savez_compressed(bank,**arrays)
    dump(out/'RECEIPT.json',dict(status='COMPLETED',domain=domain,method=method,role=role,relative=str(bank.relative_to(w)),sha256=sha(bank),
        count=len(axis),first=int(axis[0]),last=int(axis[-1]),record_id=task['indexes'][role]['record_id'],
        source_cache_relative=cache_receipt['relative'],source_cache_sha256=cache_receipt['sha256'],scaler_sha256=task['scaler_sha256'],
        selected=selected,labels_read=False,model_state_RNG_immutable=True,optimizer_updates=0,
        elapsed_GPU_seconds=time.monotonic()-started if device else 0.,elapsed_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()if device else 0))
    print(json.dumps(dict(event='READOUT_COMPLETED',domain=domain,method=method,role=role,count=len(axis))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','cache','merge_cache','gates','train','freeze','readout']);p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--method',choices=['A','B','C','D'])
    p.add_argument('--output',type=Path);p.add_argument('--domain',choices=['SWAT','PSM']);p.add_argument('--role',choices=['SOURCE_FIT','SOURCE_SELECT','CAL','QUERY']);p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1)
    a=p.parse_args();out=a.output or a.workspace/'results'/EXPERIMENT
    if a.action=='prepare':prepare(a.workspace,out)
    elif a.action=='cache':cache(a.workspace,out,a.domain,a.role,a.part,a.parts)
    elif a.action=='merge_cache':merge_cache(a.workspace,out,a.domain,a.role)
    elif a.action=='gates':gates(a.workspace,out,a.domain)
    elif a.action=='train':train(a.workspace,out,a.domain,a.method)
    elif a.action=='freeze':freeze_trainings(a.workspace,out)
    elif a.action=='readout':readout(a.workspace,out,a.domain,a.method,a.role)


if __name__=='__main__':main()
