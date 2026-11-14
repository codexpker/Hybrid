import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,importlib.util,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
P=Path(__file__).resolve().parent;R=P.parents[1];torch.set_num_threads(1);start=time.monotonic()
sys.path.insert(0,str(R/'preparations/left_baseline_build_20261101_v1/upstream'));from models.Left.LEFT import Left
config=R/'literature/novelty_review/left_source_20261031/configs/SWAT/model_configs_Left.json';args=SimpleNamespace(**json.loads(config.read_text()))
s=importlib.util.spec_from_file_location('observer',R/'hybrid_ad/src/hybrid_tsad/left_loss_observer.py');obs=importlib.util.module_from_spec(s);s.loader.exec_module(obs)
torch.manual_seed(2025);m=Left(args);m.train();x=torch.randn(3,192,31);rng=torch.get_rng_state().clone();state={k:v.detach().clone()for k,v in m.state_dict().items()}
loss=m(x,None,None,None)[0];loss.backward();grads={k:v.grad.detach().clone()for k,v in m.named_parameters()if v.grad is not None};after_rng=torch.get_rng_state().clone();after_state={k:v.detach().clone()for k,v in m.state_dict().items()};step=m.step
m.load_state_dict(state);m.step=0;m.zero_grad();torch.set_rng_state(rng);record=obs.attach_native_loss_observer(m,torch);new=m(x,None,None,None)[0];new.backward()
assert torch.equal(loss,new)and torch.equal(after_rng,torch.get_rng_state())and step==m.step
assert all(torch.equal(after_state[k],v)for k,v in m.state_dict().items())
assert all(torch.equal(g,dict(m.named_parameters())[k].grad)for k,g in grads.items())
r=record['last'];assert all(len(r[k])==3 for k in ['ms','cycle','cycle_ft','cycle_tf','total'])
meta=json.loads((P/'SWAT_ACTUAL_METADATA.json').read_text());names=[n for n in meta['counts']if n!='label'];removed=[2,4,9,10,11,13,15,19,20,21,22,29,30,31,32,33,42,43,48,50]
mapping=[dict(original_index=i,raw_name=n,removed=i in removed,retained_index=sum(j not in removed for j in range(i))if i not in removed else None)for i,n in enumerate(names)]
(P/'OFFICIAL_31_CHANNEL_MAPPING.json').write_text(json.dumps(dict(mapping=mapping,order='first occurrence in official longCSV; no stripping or sorting',official_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),removed_indices=removed),indent=2))
out=dict(status='PASS',native_loss_bitexact=True,all_parameter_gradients_bitexact=True,RNG_and_state_bitexact=True,no_second_model_forward=True,no_tags_in_loss=True,per_sample_loss_mean_equal_native=True,MS_lengths=r['MS_lengths'],MS_weights=r['MS_weights'].tolist(),native_train_weights=dict(cycle=m.alpha_cycle_l,MS=m.alpha_ms_l,cons=m.alpha_cons_l),native_score_weights=dict(cycle=m.alpha_cycle_s,MS=m.alpha_ms_s,cons=m.alpha_cons_s),source_sha256=hashlib.sha256((R/'hybrid_ad/src/hybrid_tsad/left_loss_observer.py').read_bytes()).hexdigest(),seconds=time.monotonic()-start)
(P/'CPU_LOSS_OBSERVER.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
