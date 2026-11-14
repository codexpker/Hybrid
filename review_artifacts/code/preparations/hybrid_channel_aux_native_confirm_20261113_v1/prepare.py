from pathlib import Path
import json,hashlib,shutil
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1]
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
cfg=R/'preparations/cross_domain_left_arx_20261111_v1/configs/SMD/model_configs_Left.json';shutil.copyfile(cfg,P/'MODEL.json')
for seed in [2025,2026,2027]:
    rng=np.random.default_rng(seed);source=rng.integers(0,17087-191,size=(4000,128),dtype=np.int64);paired=rng.integers(0,17087-191,size=(2000,128),dtype=np.int64);channel=np.random.default_rng(84551+seed-2025).integers(0,38,size=paired.shape,dtype=np.int64)
    np.savez_compressed(P/('PLAN_'+str(seed)+'.npz'),source=source,paired=paired,channels=channel)
runtime=P/'runtime/hybrid_tsad';runtime.mkdir(parents=True,exist_ok=True);(runtime/'__init__.py').write_text('')
for f in ['left_predictive_cycle.py','left_channel_blindspot.py']:shutil.copyfile(R/'hybrid_ad/src/hybrid_tsad'/f,runtime/f)
freeze=dict(id=P.name,entity='SMD machine-1-1',seeds=[2025,2026,2027],source='same target normal FIT common initialization; not multisource Stage1 or target-free Z',source_updates=4000,source_SELECT_updates=[3000,3200,3400,3600,3800,4000],pair_updates=2000,pair_SELECT_updates=list(range(200,2001,200)),SELECT='same mean original native anomaly-score endpoint over nonoverlap H192 normalSELECT windows; minimize, ties earlier',optimizer='fresh Adam1e-4/B128 fixed, paired source same PT/step/scaler/plan',loss='native original SSL vs .5native+.5wholechannel masked-target originalcoeff; no new parameters',readout='ONLY native_readout for source/native/aux; no score_one_channel or masked inference',CAL='continuous role-local>=512, ownq99higher; no QUERY fitting',QUERY='all original512..28478',roles_sha256=sha(P/'ROLES.json'),scaler_sha256=sha(P/'FIT_ONLY_SCALER.json'),model_sha256=sha(P/'MODEL.json'),runtime_sha256={f:sha(runtime/f)for f in ['left_predictive_cycle.py','left_channel_blindspot.py']},plan_sha256={str(s):sha(P/('PLAN_'+str(s)+'.npz'))for s in [2025,2026,2027]},primary_comparison='all3paired aux-minus-native AP/ROC; no bestseed selection; sustained/FP/event evidence separate',stop='no stable learning increment -> stop; old blindscore NOGO unchanged')
(P/'FREEZE.json').write_text(json.dumps(freeze,indent=2));print(json.dumps(dict(status='FROZEN',id=P.name)))
