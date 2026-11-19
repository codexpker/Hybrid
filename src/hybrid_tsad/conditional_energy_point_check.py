"""One post-seal CUDA replay per bank; no labels, fitting, or bank changes.

CPU and CUDA RNGs do not generate identical complementary masks from the same
seed. Consequently this integrated check uses the original CUDA mask factory.
The external official JMLR class is read from SHA-pinned private reference bytes,
not redistributed in this package. Only its optional type annotations are
removed, as in the independent scalar/gradient regression test.
"""
import argparse
import ast
import json
import logging
from pathlib import Path
import time
import numpy as np
import torch
from .conditional_energy_runner import read,dump,sha,load_parts,device_setup,hashes
from .conditional_response_energy import fixed_noise,energy_score

REFERENCE_SHA='8d0e065143385c77e0e1f91970385bb80c72e9780643275e5385855e5e41f7bf'


def official_class(path):
    assert sha(path)==REFERENCE_SHA
    node=next(n for n in ast.parse(Path(path).read_bytes()).body if isinstance(n,ast.ClassDef)and n.name=='EnergyScore')
    node.bases=[]
    for child in ast.walk(node):
        if isinstance(child,ast.arg):child.annotation=None
        elif isinstance(child,ast.FunctionDef):child.returns=None
    namespace=dict(logging=logging,torch=torch,np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['EnergyScore']


def check(w,root,reference):
    # Reject premature calls before allocating a CUDA context.
    seal=read(root/'GLOBAL_RAW_BANK_SEAL.json')
    assert seal['status']=='COMPLETED'and len(seal['banks'])==16
    source=official_class(reference);start=time.monotonic();device=device_setup();rows=[]
    with torch.no_grad():
        for domain in ['SWAT','PSM']:
            d=root/domain;task,g,geo,cfg=load_parts(w,d,device)
            assert sha(d/'SCALER.npz')==task['scaler_sha256']
            scaler=np.load(d/'SCALER.npz');data=task['data']['target_test']
            assert sha(w/data['relative'])==data['sha256'];raw=np.load(w/data['relative'],mmap_mode='r')
            teacher_before=hashes(geo.native)
            for arm in ['A','B']:
                for phase in ['source','R']:
                    folder=d/arm/phase;receipt=read(folder/'TRAIN_RECEIPT.json');checkpoint=w/receipt['checkpoint_relative']
                    assert sha(checkpoint)==receipt['checkpoint_sha256']
                    assert sha(d/'RESPONSE_SCALE.json')==receipt['response_scale_sha256']
                    g.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['model'],strict=True);g.eval()
                    before=hashes(g)
                    for role in ['CAL','QUERY']:
                        binding=next(b for b in seal['banks']if b['domain']==domain and b['arm']==arm and b['phase']==phase and b['role']==role)
                        bank=w/binding['relative'];assert sha(bank)==binding['sha256']
                        with np.load(bank,allow_pickle=False)as z:
                            t=int(z['absolute_time_index'][0]);cached=float(z['canonical_score'][0])
                            assert t==task['score_axes'][role][0]
                            assert int(z['available_index'][0])==t and int(z['role_window_start'][0])==t-512
                            assert int(z['response_window_start'][0])==t-99
                        lo,hi=task['roles'][role];assert lo<=t-512 and t<hi
                        # Independent direct endpoint slice, not the shard loader.
                        window=np.asarray((np.asarray(raw[t-512:t+1],dtype=np.float64)-scaler['mean'])/scaler['std'],dtype=np.float32)
                        h=torch.from_numpy(window[:512].copy()).to(device)[None]
                        y=torch.from_numpy(window[512].copy()).to(device)[None]
                        noise=fixed_noise(1,h.shape[2],torch.Generator().manual_seed(192025+t),device=device)
                        v=g(h,*noise)
                        gv=torch.stack([geo.feature(h[0],value,arm=='A')for value in v[0]])[None]
                        gy=geo.feature(h[0],y[0],arm=='A')[None]
                        project=float(energy_score(gv,gy)[0]);official=float(source(beta=1,mean=True).estimate_score_batch(gv,gy)/2)
                        # Predefined FP32 cross-device/kernel tolerance, not QUERY
                        # performance tuning. Numerical scores are never replaced.
                        assert np.isclose(project,cached,rtol=1e-4,atol=1e-5),(domain,arm,phase,role,'cached',project,cached)
                        assert np.isclose(project,official,rtol=1e-4,atol=1e-5),(domain,arm,phase,role,'official',project,official)
                        rows.append(dict(domain=domain,arm=arm,phase=phase,role=role,absolute_index=t,
                            cached_CUDA_score=cached,project_CUDA_replay=project,official_half_CUDA_score=official,
                            replay_abs_difference=abs(project-cached),official_abs_difference=abs(project-official),
                            checkpoint_sha256=sha(checkpoint),bank_sha256=sha(bank)))
                    assert hashes(g)==before
            assert hashes(geo.native)==teacher_before
    dump(root/'POINT_PARITY.json',dict(status='PASS',rows=rows,points=16,selection='first point of each previously sealed bank',
        reference_sha256=REFERENCE_SHA,script_sha256=sha(Path(__file__)),global_seal_sha256=sha(root/'GLOBAL_RAW_BANK_SEAL.json'),
        same_CUDA_mask_seed=84551,noise='CPU Generator192025+absolute_index; original CUDA transfer',
        rtol=1e-4,atol=1e-5,labels_read=False,optimizer_or_training=False,raw_banks_modified=False,
        CPU_CUDA_same_seed_not_same_masks=True,elapsed_GPU_seconds=time.monotonic()-start,peak_GiB=torch.cuda.max_memory_allocated()/2**30))
    print(json.dumps(dict(event='POINT_PARITY_COMPLETED',points=16,max_replay_abs_difference=max(r['replay_abs_difference']for r in rows),
        max_official_abs_difference=max(r['official_abs_difference']for r in rows))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();check(a.workspace,a.output,a.reference or a.workspace/'literature/reference_source_20261119/jmlr_generative_energy/scoring_rules.official.py')
