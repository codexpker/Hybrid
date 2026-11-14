"""Read-only missing normal-role native endpoint scores; no adaptation or labels."""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name;UP=R/'preparations/left_baseline_build_20261101_v1/upstream'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    with Path(p).open('x')as f:json.dump(d,f,indent=2,allow_nan=False)
def tensorhash(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def configure(domain,device):
    root=R/'results/left_psm_R_seed2025_20261101_v1'if domain=='PSM'else R/'results/cross_domain_left_arx_20261111_v1/left_train'
    pt=root/('model/PSM/Left/checkpoints_0/checkpoint.pth'if domain=='PSM'else 'model/SMD/Left/checkpoints_0/checkpoint.pth')
    cfg=UP/'configs/PSM/model_configs_Left.json'if domain=='PSM'else R/'preparations/cross_domain_left_arx_20261111_v1/configs/SMD/model_configs_Left.json'
    asset=json.loads((P/'ASSET_INVENTORY.json').read_text())['PSM'if domain=='PSM'else'SMD_machine15']
    assert sha(pt)==asset['checkpoint']['sha256']and sha(cfg)==asset['config']['sha256']
    sys.path.insert(0,str(UP));from models.Left.LEFT import Left
    random.seed(2025);np.random.seed(2025);torch.manual_seed(2025)
    if device=='cuda':torch.cuda.manual_seed_all(2025)
    model=Left(SimpleNamespace(**json.loads(cfg.read_text()))).to(device);state=model.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True),strict=False)
    assert not state.missing_keys and all(k.endswith(('total_ops','total_params'))for k in state.unexpected_keys)
    # Preserve the actual earlier endpoint reader's ordinary attribute. In eval
    # curriculum is lambda_max independent of step; future SMD-U must use2200.
    step=0 if domain=='PSM'else json.loads((root/'RUNTIME_STATE.json').read_text())['final_python_step'];model.step=step;model.eval()
    return model,dict(checkpoint_path=str(pt),checkpoint_sha256=sha(pt),config_path=str(cfg),config_sha256=sha(cfg),python_step=step,parameters_buffers='selected checkpoint',PSM_eval_step0_matches_cached_reader=domain=='PSM')
def role_values(domain):
    if domain=='PSM':
        prefix=R/'preparations/left_baseline_build_20261101_v1/normal_prefix_R_v1/prefix.npy'
        if not prefix.exists():prefix=R/'preparations/hybrid_fit_reliability_20261111_v1/assets/PSM_prefix.npy'
        assert sha(prefix)=='f722bce0b19324d85c7b5691efe73453eb56022e867ccc1ba47bb3406787df38';sc=json.loads((R/'results/left_psm_R_seed2025_20261101_v1/SCALER_R.json').read_text())['statistics'];raw=np.load(prefix);v=(raw-np.asarray(sc['mean']))/np.asarray(sc['scale']);return [('FIT',v[:105984],0),('SELECT_CAL',v[105984:132481],105984)]
    p=R/'preparations/cross_domain_left_arx_20261111_v1';sc=json.loads((p/'FIT_ONLY_SCALER.json').read_text());out=[]
    for role,start in [('FIT',0),('SELECT',14767)]:
        with np.load(p/'data'/(role+'.npz'))as z:v=z['values'].astype(np.float64);assert z['observed'].all()
        out.append((role,(v-np.asarray(sc['mean']))/np.asarray(sc['scale']),start))
    return out
def snapshot(m):return dict(step=m.step,modes={n:x.training for n,x in m.named_modules()},state_sha256={n:tensorhash(v)for n,v in m.state_dict().items()},CPU_rng=tensorhash(torch.get_rng_state()),CUDA_rng=tensorhash(torch.cuda.get_rng_state())if next(m.parameters()).is_cuda else None)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('domain',choices=['PSM','SMD']);ap.add_argument('--cpu-check',action='store_true');args=ap.parse_args();domain=args.domain;start=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device='cpu'if args.cpu_check else'cuda'
    if device=='cuda':torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(30*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
    m,identity=configure(domain,device);before=snapshot(m);roles=role_values(domain);outputs=[]
    for name,v,offset in roles:
        index=np.arange(offset+512,offset+len(v));index=index[:2]if args.cpu_check else index;scores=[];per_channel=[]
        with torch.no_grad():
            for a in range(0,len(index),256):
                ends=index[a:a+256]-offset;x=np.stack([np.float32(v[t-191:t+1])for t in ends]);assert x.shape[1:]==(192,25 if domain=='PSM'else 38)
                raw=m.infer(torch.from_numpy(x).to(device),None,None,None)[0].cpu().numpy();assert raw.shape==(len(x),192,v.shape[1]);flat=np.array(raw.reshape(-1,v.shape[1]));score=np.mean(flat,axis=-1).reshape(len(x),192)[:,-1];scores.append(score);per_channel.append(raw[:,-1])
        if args.cpu_check:continue
        out=O/'frozen_normal'/domain;out.mkdir(parents=True,exist_ok=True);file=out/(name+'.npz');assert not file.exists();np.savez_compressed(file,index=index,score=np.concatenate(scores),per_channel_last=np.concatenate(per_channel),available_index=index,valid=np.ones(len(index),bool));outputs.append(dict(role=name,path=str(file),sha256=sha(file),points=len(index),original_start=offset,aliases=['SELECT','CAL']if name=='SELECT_CAL'else[name]))
    assert snapshot(m)==before
    receipt=dict(status='CPU_CHECK_PASS'if args.cpu_check else'COMPLETED',domain=domain,identity=identity,outputs=outputs,mode_parameters_buffers_step_CPU_CUDA_RNG_unchanged=True,no_CAL_or_QUERY_model_selection=True,no_QUERY_score_or_label_opened=True,no_training=True,seconds=time.monotonic()-start)
    if device=='cuda':receipt['peak_allocated']=torch.cuda.max_memory_allocated();receipt['peak_reserved']=torch.cuda.max_memory_reserved()
    if args.cpu_check:dest=P/('CPU_FROZEN_'+domain+'.json')
    else:dest=O/'frozen_normal'/domain/'RECEIPT.json'
    dump(dest,receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
