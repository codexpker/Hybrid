import hashlib,importlib.util,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
P=Path(__file__).resolve().parent;R=P.parents[1];start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(2025)
root=R/'preparations/left_baseline_build_20261101_v1/upstream';sys.path.insert(0,str(root));from models.Left.LEFT import Left
cfg=json.loads((P/'configs/SMD/model_configs_Left.json').read_text());assert cfg['enc_in']==38 and cfg['seq_len']==192
model=Left(SimpleNamespace(**cfg));model.train();x=torch.randn(2,192,38);loss,_=model(x,None,None,None);assert loss.ndim==0 and torch.isfinite(loss);loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters());assert all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None)
model.eval();before={k:v.detach().clone()for k,v in model.state_dict().items()};step=model.step
with torch.no_grad():raw,_=model.infer(x,None,None,None)
assert raw.shape==(2,192,38)and torch.isfinite(raw).all();assert model.step==step and all(torch.equal(before[k],v)for k,v in model.state_dict().items())
s=importlib.util.spec_from_file_location('score',R/'hybrid_ad/src/hybrid_tsad/native_score_numpy.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);_,score=m.native_numpy_scores([raw.numpy()],38);assert score.shape==(384,)
sc=json.loads((P/'FIT_ONLY_SCALER.json').read_text());assert sc['n_samples_seen']==14223 and sc['SELECT_CAL_QUERY_not_used']
roles=json.loads((P/'INPUT_QUALIFICATION.json').read_text())['roles'];bounds={e['role']:(e['original_start'],e['original_start']+e['shape'][0])for e in roles}
assert bounds['SELECT'][0]-bounds['FIT'][1]==544 and bounds['CAL'][0]-bounds['SELECT'][1]==544
ends=np.linspace(20020,23704,512,dtype=np.int64);assert ends[0]-512>=bounds['CAL'][0]and ends[-1]<bounds['CAL'][1]
train_windows=(14223-192)+1;val_windows=(4197-192)//192+1;assert train_windows==14032 and val_windows==21
dump=dict(status='PASS',official_SMD_C38_native_loss_backward=True,raw_score_shape=list(raw.shape),native_channel_mean_C38=True,eval_state_immutable=True,ordinary_pythonstep_preserved=True,FIT_only_scaler=True,role_guard544=True,CAL512_full_history513_contained=True,train_windows=train_windows,SELECT_windows=val_windows,QUERY_values_labels_read=False,weights_discarded=True,seconds=time.monotonic()-start,model_config_sha256=hashlib.sha256((P/'configs/SMD/model_configs_Left.json').read_bytes()).hexdigest())
(P/'CPU_GATE.json').write_text(json.dumps(dump,indent=2));print(json.dumps(dump))
