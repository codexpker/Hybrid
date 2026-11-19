"""Fixed two-arm source and 1:1 replay experiment; label-free runner.

Source selection includes step0. Replay is a fixed endpoint, never reselected.
Logical batch16 uses exact microbatch1 accumulation for response isolation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import concurrent.futures
import subprocess
import sys
import time
import numpy as np
import torch
from .conditional_response_energy import ConditionalGenerator, ResponseGeometry, fixed_noise
from .conditional_response_profile import load_native, hashes
from .hybrid_v1.provenance import OfficialSources

ID='tf_conditional_energy_20261119_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False)


def prepare(w,out):
    out.mkdir(parents=True,exist_ok=False)
    for domain in ['SWAT','PSM']:
        d=out/domain;d.mkdir();old=read(w/'results/mlp_replay_20261119_v1'/domain/'TASK.json')
        source=old['data']['source_train'];target=old['data']['target_test']
        for rec in [source,target]:assert sha(w/rec['relative'])==rec['sha256']
        raw=np.load(w/source['relative'],mmap_mode='r');lo,hi=old['roles']['SOURCE_FIT']
        fit=np.asarray(raw[lo:hi],dtype=np.float64);mean=fit.mean(0);std=fit.std(0);std[std<1e-8]=1
        np.savez(d/'SCALER.npz',mean=mean,std=std)
        rng=np.random.default_rng(2025)
        source_plan=rng.integers(lo+512,hi,size=(2000,16),dtype=np.int64)
        rr=np.random.default_rng(2026);u0,u1=old['roles']['U_FIT']
        target_plan=rr.integers(u0+512,u1,size=(2000,8),dtype=np.int64)
        replay_source=rr.integers(lo+512,hi,size=(2000,8),dtype=np.int64)
        select=np.arange(hi+512,raw.shape[0],512,dtype=np.int64) if domain=='PSM' else np.arange(396512,495000,512,dtype=np.int64)
        np.savez(d/'PLAN.npz',source_endpoints=source_plan,target_endpoints=target_plan,replay_source_endpoints=replay_source,select_endpoints=select)
        roles=dict(old['roles']);roles['SOURCE_SELECT']=[hi,raw.shape[0] if domain=='PSM' else 495000]
        cal=[586056,607320]if domain=='SWAT'else[18624,21888]
        query=[608535,944856]if domain=='SWAT'else[23016,87744]
        dump(d/'TASK.json',dict(experiment=ID,domain=domain,seed=2025,data=old['data'],roles=roles,
            score_axes=dict(CAL=cal,QUERY=query),cal_points=np.linspace(cal[0],cal[1]-1,512,dtype=np.int64).tolist(),
            scaler_sha256=sha(d/'SCALER.npz'),plan_sha256=sha(d/'PLAN.npz'),
            window=512,response_window=100,K=4,copies=10,mask_seed=84551,
            train_noise_seed=2025,select_noise_seed=92025,score_noise_seed_base=192025,
            training=dict(updates=2000,logical_batch=16,microbatch=1,optimizer='Adam',lr=1e-4,
                betas=[.9,.999],eps=1e-8,weight_decay=0,source_select_every=100,source_include_step0=True,
                replay='8 target then8 source, fixed endpoint, fresh optimizer',phase_noise_reset=True),
            normalization='SOURCE_FIT float64 mean/std ddof0; std<1e-8->1; official norm0',
            sr_rule='SOURCE_FIT anchored nonoverlapping complete100 windows, entire10copy RMS, floor1e-6',
            energy='meanK distance - ordered_pair_sum/(2K(K-1)); half official JMLR energy',
            labels_accessed=False))


def arrays(w,d,task,roles):
    z=np.load(d/'SCALER.npz');result={}
    for role in roles:
        rec=task['data']['source_train'if role.startswith('SOURCE')else'target_test']
        assert sha(w/rec['relative'])==rec['sha256']
        raw=np.load(w/rec['relative'],mmap_mode='r');lo,hi=task['roles'][role]
        # Physical materialization is strictly restricted to the current role.
        result[role]=(np.asarray((np.asarray(raw[lo:hi],dtype=np.float64)-z['mean'])/z['std'],dtype=np.float32),lo)
    return result


def pair(role,endpoint,device):
    values,lo=role;i=int(endpoint)-lo
    if not(512<=i<len(values)):raise ValueError('513-point footprint crosses role')
    return torch.from_numpy(values[i-512:i].copy()).to(device)[None],torch.from_numpy(values[i].copy()).to(device)[None]


def device_setup():
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(20/45,device)
    torch.manual_seed(2025);torch.cuda.manual_seed(2025);return device


def scale(w,d):
    start=time.monotonic();task=read(d/'TASK.json');device=device_setup()
    fit=arrays(w,d,task,['SOURCE_FIT'])['SOURCE_FIT'][0]
    native,cfg=load_native(w,device);geo=ResponseGeometry(native,1.,task['mask_seed']);before=hashes(native)
    total=0.;count=0;n=0
    with torch.no_grad():
        for begin in range(0,len(fit)-99,100):
            response=geo.response(torch.tensor(fit[begin:begin+100],device=device)[None])
            total+=float(response.double().square().sum());count+=response.numel();n+=1
    rms=max((total/count)**.5,1e-6);assert hashes(native)==before
    dump(d/'RESPONSE_SCALE.json',dict(response_rms=rms,sum_squares=total,element_count=count,windows=n,
        source_fit=task['roles']['SOURCE_FIT'],source_fit_only=True,copies=10,window=100,norm=0,
        checkpoint_sha256=cfg['checkpoint_sha256'],scaler_sha256=task['scaler_sha256'],
        elapsed_GPU_seconds=time.monotonic()-start,peak_GiB=torch.cuda.max_memory_allocated()/2**30))


def load_parts(w,d,device):
    task=read(d/'TASK.json');assert sha(d/'PLAN.npz')==task['plan_sha256'] and sha(d/'SCALER.npz')==task['scaler_sha256']
    native,cfg=load_native(w,device);geo=ResponseGeometry(native,read(d/'RESPONSE_SCALE.json')['response_rms'],task['mask_seed'])
    g=ConditionalGenerator(OfficialSources.from_workspace(w)).to(device)
    return task,g,geo,cfg


def validate(g,geo,role,endpoints,task,device,response):
    modes=[(m,m.training)for m in g.modules()];g.eval();noise=torch.Generator().manual_seed(task['select_noise_seed']);total=0.
    with torch.no_grad():
        for endpoint in endpoints:
            h,y=pair(role,endpoint,device);v=g(h,*fixed_noise(1,h.shape[2],noise,device=device))
            total+=float(geo.score(h,v,y,response)[0])
    for m,mode in modes:m.training=mode
    return total/len(endpoints)


def train(w,d,arm,phase):
    out=d/arm/phase;out.mkdir(parents=True,exist_ok=False);started=time.monotonic();device=device_setup()
    task,g,geo,cfg=load_parts(w,d,device);before=hashes(geo.native)
    roles=['SOURCE_FIT','SOURCE_SELECT'] if phase=='source'else['SOURCE_FIT','U_FIT']
    data=arrays(w,d,task,roles);plans=np.load(d/'PLAN.npz');response=arm=='A'
    origin=None
    if phase=='R':
        origin=d/arm/'source/SELECTED.pth';g.load_state_dict(torch.load(origin,map_location=device,weights_only=True)['model'],strict=True)
    initial=hashes(g);opt=torch.optim.Adam(g.parameters(),lr=1e-4);noise=torch.Generator().manual_seed(2025)
    best=float('inf');selected=None;selection=[];trace=[]
    def checkpoint(step,loss,path):torch.save(dict(model=g.state_dict(),step=step,loss=loss),path)
    if phase=='source':
        loss=validate(g,geo,data['SOURCE_SELECT'],plans['select_endpoints'],task,device,response)
        best=loss;selected=0;selection.append(dict(step=0,loss=loss));checkpoint(0,loss,out/'SELECTED.pth')
    for step in range(2000):
        opt.zero_grad(set_to_none=True);total=0.
        if phase=='source':batch=[('SOURCE_FIT',t)for t in plans['source_endpoints'][step]]
        else:batch=[('U_FIT',t)for t in plans['target_endpoints'][step]]+[('SOURCE_FIT',t)for t in plans['replay_source_endpoints'][step]]
        shared,variate=fixed_noise(16,len(data['SOURCE_FIT'][0][0]),noise,device=device)
        for i,(role,endpoint)in enumerate(batch):
            h,y=pair(data[role],endpoint,device);v=g(h,shared[i:i+1],variate[i:i+1])
            loss=geo.score(h,v,y,response).mean();assert torch.isfinite(loss)
            (loss/16).backward();total+=float(loss.detach())/16
        assert all(p.grad is None or torch.isfinite(p.grad).all()for p in g.parameters())
        grad=float(sum(p.grad.square().sum()for p in g.parameters()if p.grad is not None).sqrt());opt.step()
        trace.append(dict(step=step+1,loss=total,gradient_norm=grad))
        if (step+1)%100==0:
            if phase=='source':
                value=validate(g,geo,data['SOURCE_SELECT'],plans['select_endpoints'],task,device,response)
                selection.append(dict(step=step+1,loss=value))
                if value<best:best=value;selected=step+1;checkpoint(selected,best,out/'SELECTED.pth')
            print(json.dumps(dict(event='TRAIN_PROGRESS',domain=task['domain'],arm=arm,phase=phase,step=step+1,loss=total,elapsed=time.monotonic()-started)),flush=True)
            dump(out/f'PROGRESS_{step+1:04d}.json',trace[-1])
    checkpoint(2000,trace[-1]['loss'],out/'ENDPOINT.pth');torch.save(opt.state_dict(),out/'OPTIMIZER.pth')
    dump(out/'TRACE.json',trace);dump(out/'SELECT.json',selection)
    assert hashes(geo.native)==before and all(p.grad is None for p in geo.native.parameters())
    chosen=out/('SELECTED.pth'if phase=='source'else'ENDPOINT.pth')
    dump(out/'TRAIN_RECEIPT.json',dict(status='COMPLETED',domain=task['domain'],arm=arm,phase=phase,seed=2025,
        updates=2000,exposure=32000,selected_step=selected if phase=='source' else 2000,
        checkpoint_sha256=sha(chosen),checkpoint_relative=str(chosen.relative_to(w)),initial_state_hashes=initial,
        origin_checkpoint_sha256=sha(origin)if origin else None,teacher_unchanged=True,labels_read=False,
        roles_read=roles,plan_sha256=task['plan_sha256'],scaler_sha256=task['scaler_sha256'],
        teacher_checkpoint_sha256=cfg['checkpoint_sha256'],response_scale_sha256=sha(d/'RESPONSE_SCALE.json'),
        parameters=sum(p.numel()for p in g.parameters()),elapsed_GPU_seconds=time.monotonic()-started,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30,pid=os.getpid(),GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    print(json.dumps(dict(event='STAGE_COMPLETED',domain=task['domain'],arm=arm,phase=phase)),flush=True)


def score(w,d,arm,phase):
    start=time.monotonic();device=device_setup();task,g,geo,cfg=load_parts(w,d,device)
    out=d/arm/phase;train=read(out/'TRAIN_RECEIPT.json');checkpoint=w/train['checkpoint_relative']
    assert sha(checkpoint)==train['checkpoint_sha256'];g.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['model']);g.eval()
    before=hashes(g);teacher=hashes(geo.native);data=arrays(w,d,task,['CAL','QUERY']);banks=[]
    for role in ['CAL','QUERY']:
        lo,hi=task['score_axes'][role];times=np.arange(lo,hi,dtype=np.int64);scores=np.empty(len(times),dtype=np.float32)
        with torch.no_grad():
            for i,t in enumerate(times):
                h,y=pair(data[role],t,device);noise=torch.Generator().manual_seed(192025+int(t))
                v=g(h,*fixed_noise(1,h.shape[2],noise,device=device));scores[i]=float(geo.score(h,v,y,arm=='A')[0])
                if(i+1)%10000==0:print(json.dumps(dict(event='SCORE_PROGRESS',role=role,points=i+1,total=len(times),elapsed=time.monotonic()-start)),flush=True)
        path=out/(role+'.npz');np.savez_compressed(path,absolute_time_index=times,available_index=times,
            canonical_score=scores,role_window_start=times-512,response_window_start=times-99)
        banks.append(dict(relative=str(path.relative_to(w)),sha256=sha(path),role=role,points=len(times)))
    assert before==hashes(g)and teacher==hashes(geo.native)
    dump(out/'RAW_BANK_SEAL.json',dict(status='COMPLETED',banks=banks,labels_read=False,checkpoint_sha256=sha(checkpoint),
        immutable_model_and_teacher=True,elapsed_GPU_seconds=time.monotonic()-start,peak_GiB=torch.cuda.max_memory_allocated()/2**30))


def queue(w,out):
    sys.path.insert(0,str(w/'preparations/hybrid_left_joint_cycle_20261109_v2/resources'))
    import owned_session
    from gpu_idle_guard import gpu_guard
    devices={int(line.split(',')[0]):line.split(',')[1].strip()for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
    jobs=[]
    def job(action,domain,arm,phase,gpu):
        identity='_'.join(x for x in [action,domain,arm,phase]if x)
        life=out/'lifecycles'/identity;life.mkdir(parents=True,exist_ok=False)
        started=time.monotonic();child=None;cleanup=None;error=None;code=None;guard=None
        try:
            guard=gpu_guard(devices[gpu].removeprefix('GPU-'),gpu,20)
            dump(life/'GPU_GUARD.json',guard)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=devices[gpu],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            argv=[sys.executable,'-B','-m','hybrid_tsad.conditional_energy_runner',action,'--workspace',str(w),'--output',str(out),'--domain',domain]
            if arm:argv+=['--arm',arm]
            if phase:argv+=['--phase',phase]
            timeout=14400 if action!='score' else 28800
            with(life/'stdout.log').open('xb')as stdout,(life/'stderr.log').open('xb')as stderr:
                child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env,start_new_session=True);owned_session.claim_group(child)
                dump(life/'STARTED.json',dict(pid=child.pid,GPU=gpu,argv=argv,timeout_seconds=timeout))
                print(json.dumps(dict(event='RUNNING',identity=identity,pid=child.pid,GPU=gpu)),flush=True)
                try:code=child.wait(timeout=timeout-10)
                finally:cleanup=owned_session.stop_group(child,grace=5)
            assert code==0 and cleanup['status']=='NO_LIVE_GROUP_MEMBERS'
        except BaseException as exc:error=repr(exc)
        result=dict(identity=identity,action=action,domain=domain,arm=arm,phase=phase,GPU=gpu,pid=child.pid if child else None,
            OS_exit_code=code,error=error,cleanup=cleanup,status='COMPLETED'if error is None else'FAILED',
            actual_GPU_seconds=time.monotonic()-started if child else 0.,guard_elapsed_seconds=time.monotonic()-started if not child else None)
        dump(life/'OS_WAIT.json',result);jobs.append(result)
        if error:raise RuntimeError(result)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
        list(pool.map(lambda item:job('scale',item[0],None,None,item[1]),[('SWAT',0),('PSM',1)]))
        combos=[('SWAT','A',0),('SWAT','B',1),('PSM','A',2),('PSM','B',3)]
        def chain(item):
            domain,arm,gpu=item
            job('train',domain,arm,'source',gpu);job('train',domain,arm,'R',gpu)
        list(pool.map(chain,combos))
        # This barrier is deliberate: no QUERY readout until all8 fixed training
        # endpoints have been frozen, so scores cannot feed back into training.
        dump(out/'ALL_TRAINING_FROZEN.json',dict(status='COMPLETED',training_jobs=8,labels_read=False,
            receipts=[dict(relative=str((out/d/a/p/'TRAIN_RECEIPT.json').relative_to(out)),sha256=sha(out/d/a/p/'TRAIN_RECEIPT.json'))
                      for d,a,_ in combos for p in ['source','R']]))
        def readout_chain(item):
            domain,arm,gpu=item
            job('score',domain,arm,'source',gpu);job('score',domain,arm,'R',gpu)
        list(pool.map(readout_chain,combos))
    banks=[]
    for domain,arm,_ in combos:
        for phase in ['source','R']:
            seal=read(out/domain/arm/phase/'RAW_BANK_SEAL.json')
            for bank in seal['banks']:
                path=w/bank['relative'];assert sha(path)==bank['sha256'];banks.append(dict(bank,domain=domain,arm=arm,phase=phase))
    dump(out/'GLOBAL_RAW_BANK_SEAL.json',dict(status='COMPLETED',banks=banks,jobs=jobs,no_labels_decoded=True,
        actual_GPU_hours=sum(j['actual_GPU_seconds']for j in jobs)/3600))
    print(json.dumps(dict(event='STAGE_COMPLETED',jobs=len(jobs),banks=len(banks))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','scale','train','score','queue']);p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--domain',choices=['SWAT','PSM']);p.add_argument('--arm',choices=['A','B']);p.add_argument('--phase',choices=['source','R'])
    a=p.parse_args();out=a.output or a.workspace/'results'/ID
    if a.action=='prepare':prepare(a.workspace,out)
    elif a.action=='queue':queue(a.workspace,out)
    elif a.action=='scale':scale(a.workspace,out/a.domain)
    elif a.action=='train':train(a.workspace,out/a.domain,a.arm,a.phase)
    else:score(a.workspace,out/a.domain,a.arm,a.phase)


if __name__=='__main__':main()
