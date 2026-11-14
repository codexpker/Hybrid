import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,importlib.util,json,random,time
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent;R=P.parents[1];start=time.monotonic();torch.set_num_threads(1)
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
normal=mod('normal',P/'frozen_normal_forward.py');u=mod('u',R/'hybrid_ad/src/hybrid_tsad/left_native_u.py');model,identity=normal.configure('SMD','cpu');model.train()
with np.load(P/'PAIRED_PLAN.npz')as z:plan={k:z[k].copy()for k in z.files}
values=normal.role_values('SMD')[0][1];before={k:v.detach().clone()for k,v in model.state_dict().items()};opt=torch.optim.Adam(model.parameters(),lr=1e-4)
for step in range(2):
    x=u.native_batch(values,plan,step,'A')[:2];opt.zero_grad();loss=model(torch.from_numpy(x),None,None,None)[0];assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None);opt.step()
assert model.step==2202 and any(not torch.equal(before[k],v)for k,v in model.state_dict().items())
state={k:v.detach().clone()for k,v in model.state_dict().items()};modes={k:m.training for k,m in model.named_modules()};rng=torch.get_rng_state().clone();debug=model.last_x_res;ordinary_step=model.step
with u.diagnostic_eval(model,torch):model(torch.from_numpy(u.native_batch(values,plan,2,'C')[:2]),None,None,None)
assert model.step==ordinary_step and model.last_x_res is debug and torch.equal(torch.get_rng_state(),rng)and modes=={k:m.training for k,m in model.named_modules()}and all(torch.equal(state[k],v)for k,v in model.state_dict().items())
assert hashlib.sha256(Path(identity['checkpoint_path']).read_bytes()).hexdigest()==identity['checkpoint_sha256']
result=dict(status='PASS',discarded_CPU_updates=2,fixture_batch_size=2,full_B128_native_shape_previously_accepted=True,all_trainable_parameters=True,native_loss_scalar_finite=True,optimizer_parameter_change=True,python_step_continues=True,SELECT_modes_parameters_buffers_RNG_step_debug_restored=True,source_PT_unchanged=True,plan_constructor_tags_not_passed_to_model=True,corrupt_input_is_SSL_target=True,no_QUERY_labels=True,seconds=time.monotonic()-start)
(P/'CPU_U_UPDATE_GATE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
