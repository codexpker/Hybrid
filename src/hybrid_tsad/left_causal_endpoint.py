"""Frozen official LEFT-R rolling W192 endpoint, no model/training changes."""
import argparse,hashlib,importlib.util,json,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

OP='LEFT_R_CAUSAL_ENDPOINT_W192_V1'
PT='e07a50bde4a470b79590b6392f4cb883f715e6205a78c308293f522f5656d64c'
SCALER='41da46ca635f3053d8970f5fe1eb0565117a0f69be92590cac2092d9bb50bfb3'
TOL=dict(atol=1e-5,rtol=1e-5)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def asset_ok(a):assert sha(a['path'])==a['sha256'],a['path'];return a['path']
def prefixes(values,ends):
    ends=np.asarray(ends,dtype=np.int64)
    if len(ends)==0 or ends.min()<191 or ends.max()>=len(values):raise ValueError('Fixed complete W192 prefixes required')
    return np.stack([np.float32(values[t-191:t+1]) for t in ends])
def import_adapter(path):
    s=importlib.util.spec_from_file_location('accepted_left_adapter',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def load_model(c,device):
    import torch
    original=read(asset_ok(c['native_score_config']));adapter=import_adapter(asset_ok(c['adapter']))
    assert original['checkpoint']['sha256']==PT and original['scaler']['sha256']==SCALER
    for rel,h in original['source_hashes'].items():assert sha(Path(original['source_root'])/rel)==h,rel
    asset_ok(original['checkpoint']);asset_ok(original['scaler'])
    random.seed(2025);np.random.seed(2025);torch.manual_seed(2025)
    root=Path(original['source_root']);mod=adapter.import_official(root,'models.Left.LEFT')
    model=mod.Left(SimpleNamespace(**read(root/'configs/PSM/model_configs_Left.json'))).to(device)
    loaded=adapter.load_expected_checkpoint(model,torch.load(original['checkpoint']['path'],map_location='cpu',weights_only=True),original['expected_thop_keys'])
    assert loaded['missing_keys']==[] and len(loaded['unexpected_keys'])==86
    model.eval();assert model.seq_len==192 and model.enc_in==25 and model.smooth_k==3
    return model,adapter,original
def score_batch(model,adapter,windows,device,*,return_channels=False):
    import torch
    raw,_=model.infer(torch.from_numpy(np.ascontiguousarray(windows)).to(device),None,None,None)
    a=raw.detach().cpu().numpy();assert a.shape==windows.shape and a.dtype==np.float32 and np.isfinite(a).all()
    _,score=adapter.author_numpy_scores([a])
    endpoint=score.reshape(len(windows),192)[:,-1].copy()
    return (endpoint,a[:,-1,:].copy()) if return_channels else endpoint
def checks(c,model,adapter,normalized,out,device):
    import torch
    ends=np.arange(191,447,dtype=np.int64);windows=prefixes(normalized,ends)
    before=adapter.score_state(model);cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone() if device.startswith('cuda') else None
    allscores=score_batch(model,adapter,windows,device)
    ids=np.array([0,1,2,64,128,192,254,255]);ones=np.concatenate([score_batch(model,adapter,windows[i:i+1],device) for i in ids])
    permutation=np.random.default_rng(2025).permutation(256);permuted=score_batch(model,adapter,windows[permutation],device)
    restored=np.empty_like(permuted);restored[permutation]=permuted
    assert np.allclose(ones,allscores[ids],**TOL) and np.allclose(restored,allscores,**TOL)
    # Complete raw future is outside every t-prefix. No refit/global test stats.
    changed=normalized.copy();changed[447:]+=999
    assert np.array_equal(prefixes(changed,ends),windows)
    # Witness original symmetric smooth uses only prefix endpoint replication.
    source=sys.modules['models.Left.LEFT'];original=source.moving_average_1d;witness={}
    def tap(x,k):
        result=original(x,k)
        witness.update(maxabs=float((result[:,-1]-(x[:,-2]+2*x[:,-1])/3).abs().max()))
        return result
    source.moving_average_1d=tap
    try:score_batch(model,adapter,windows[:1],device)
    finally:source.moving_average_1d=original
    assert witness['maxabs']<=1e-5
    adapter.require_frozen_score_state(before,adapter.score_state(model))
    assert torch.equal(cpu,torch.get_rng_state()) and (cuda is None or torch.equal(cuda,torch.cuda.get_rng_state()))
    with (out/'BATCH_GATE_ARRAYS.npz').open('xb') as f:np.savez_compressed(f,ends=ends,b256=allscores,b1_ids=ids,b1=ones,permutation=permutation,permuted=permuted)
    result=dict(status='PASS',B1_B256_maxabs=float(np.max(np.abs(ones-allscores[ids]))),permutation_maxabs=float(np.max(np.abs(restored-allscores))),
        tolerances=TOL,prototype_parameter_buffer_step_modes_exact=True,CPU_CUDA_RNG_exact=True,future_outside_prefix_input_exact=True,
        endpoint_smooth_cycle_formula='(u190+2*u191)/3',smooth_formula_maxabs=witness['maxabs'],MS_not_smoothed=True,no_second_rolling_smoothing=True,
        input_permission='original pretest features only; fixed original scaler; no query values or labels',labels_read=False,new_training=False)
    dump(out/'CHECKS.json',result);return result
def execute(c):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);device='cpu' if c['phase']=='cpu_gate' else 'cuda:0'
    if device.startswith('cuda'):
        p=torch.cuda.get_device_properties(0);assert str(p.uuid).removeprefix('GPU-')==c['expected_uuid'] and 'A40' in p.name
        torch.cuda.set_per_process_memory_fraction(30*1024**3/p.total_memory,0);torch.cuda.reset_peak_memory_stats(0)
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    model,adapter,original=load_model(c,device)
    before=adapter.score_state(model)
    dump(out/'STATE_BEFORE.json',before)
    torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if device.startswith('cuda') else None),out/'RNG_BEFORE.pt')
    dump(out/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,CUDA_VISIBLE_DEVICES=__import__('os').environ.get('CUDA_VISIBLE_DEVICES')))
    if c['phase'] in ('cpu_gate','profile'):
        data=c['pretest'];raw=np.load(asset_ok(data),allow_pickle=False,mmap_mode='r')[:768]
        values=adapter.transform_query_with_saved_scaler(np.asfortranarray(raw),read(original['scaler']['path']))
        result=checks(c,model,adapter,values,out,device)
        if c['phase']=='profile':
            timings=[]
            for i in range(3):
                torch.cuda.synchronize();at=time.monotonic();score_batch(model,adapter,prefixes(values,np.arange(191+i*64,447+i*64)),device);torch.cuda.synchronize();timings.append(time.monotonic()-at)
            conservative=max(timings)*1.5
            result.update(B256_seconds=timings,full_87329_batches=342,estimated_full_GPU_seconds=conservative*342+60,formula='1.5*max(B256 timings)*342 +60s startup/io allowance')
    elif c['phase']=='query':
        profile=read(asset_ok(c['profile_receipt']));assert profile['status']=='COMPLETED' and profile['checks']['status']=='PASS'
        assert profile['checks']['estimated_full_GPU_seconds']<=c['cap_seconds']-5
        q=original['query'];assert sha(q['array'])==q['array_sha256'];raw=np.load(q['array'],allow_pickle=False)
        values=adapter.transform_query_with_saved_scaler(raw,read(original['scaler']['path']))
        ends=np.arange(512,87841,dtype=np.int64);scores=[];channels=[];timings=[]
        for at in range(0,len(ends),256):
            t=time.monotonic();scalar,channel=score_batch(model,adapter,prefixes(values,ends[at:at+256]),device,return_channels=True);scores.append(scalar);channels.append(channel);timings.append(time.monotonic()-t)
        scores=np.concatenate(scores);channels=np.concatenate(channels);assert scores.shape==(87329,) and scores.dtype==np.float32 and np.isfinite(scores).all()
        assert channels.shape==(87329,25) and np.array_equal(np.mean(np.array(channels),axis=-1),scores)
        # These original native window endpoints use the same exact prefix; no
        # extra neural calls or shifting earlier offline positions are made.
        anchor=np.array([575,767,959],dtype=np.int64)
        with np.load(asset_ok(c['native_scores']),allow_pickle=False) as old:ref=old['canonical_score'][anchor]
        assert np.allclose(scores[anchor-512],ref,**TOL)
        with (out/'scores.npz').open('xb') as f:np.savez_compressed(f,index=ends,available_index=ends,score=scores,endpoint_channel_scores=channels,channel_names=np.asarray(adapter.FEATURES),valid=np.ones(len(ends),bool),window_start=ends-191,window_end_inclusive=ends,execution_batch_end=np.minimum(512+((np.arange(len(ends))//256+1)*256)-1,87840))
        result=dict(status='PASS',points=87329,axis=[512,87841],anchors=anchor.tolist(),native_endpoint_maxabs=float(np.max(np.abs(scores[anchor-512]-ref))),timings=timings,
            labels_read=False,no_test_statistics_fit=True,score='unchanged author infer last_position, original numpy float32 reduction',batch_size=256,
            availability_semantics='functional causal prefix; batch independence verified; offline computation schedule not measured streaming latency')
    else:raise ValueError(c['phase'])
    after=adapter.score_state(model);adapter.require_frozen_score_state(before,after)
    dump(out/'STATE_AFTER.json',after)
    torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if device.startswith('cuda') else None),out/'RNG_AFTER.pt')
    assert sha(original['checkpoint']['path'])==PT and sha(original['scaler']['path'])==SCALER
    peak=dict(allocated_bytes=torch.cuda.max_memory_allocated(0),reserved_bytes=torch.cuda.max_memory_reserved(0)) if device.startswith('cuda') else None
    receipt=dict(status='COMPLETED',phase=c['phase'],operation=OP,checkpoint_sha256=PT,scaler_sha256=SCALER,checks=result,seconds=time.monotonic()-start,new_training=False,labels_read=False,peak=peak)
    dump(out/'RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    assert sha(a.config)==a.trusted_config_sha256;c=read(a.config);assert c['operation']==OP and c['seed']==2025
    for path,h in c['immutable_sha256'].items():assert sha(path)==h,path
    execute(c)
    for path,h in c['immutable_sha256'].items():assert sha(path)==h,path
if __name__=='__main__':main()
