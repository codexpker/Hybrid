"""Training-side manifest access only; no test-label loading or fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def canonical_domain(name):
    return 'SMD' if name in ('SMD','SMD_SUBSET') else name

class ManifestSampler:
    def __init__(self,manifest,role='fit',exclude_dataset=None,seed=0):
        self.path=Path(manifest);self.meta=json.loads(self.path.read_text())
        self.records=[r for r in self.meta['records'] if r['role']==role and canonical_domain(r['dataset'])!=canonical_domain(exclude_dataset)]
        self.domains=sorted({r['dataset'] for r in self.records})
        if not self.domains:raise ValueError('Empty source pool')
        self.rng=np.random.default_rng(seed);self.cache={}
    def array(self,r):
        key=r['source']
        if key not in self.cache:
            p=Path(key)
            if r['dataset']=='PSM':a=pd.read_csv(p).iloc[:,1:].to_numpy(dtype='float32')
            elif r['dataset']=='SMD_SUBSET':a=np.loadtxt(p,delimiter=',',dtype='float32')
            elif r['dataset']=='TEP':
                a=np.load(self.meta.get('tep_cache_path',self.path.parent/'tep_normal_runs_1_100.npy'),mmap_mode='r')
            else:raise ValueError(r['dataset'])
            self.cache[key]=a
        a=self.cache[key]
        return a[r['cache_row']] if r['dataset']=='TEP' else a
    def take(self,r,origin,channel):
        assert origin in r['forecast_origins'] and 0<=channel<len(r['columns'])
        a=self.array(r);h=r['history'];f=r['horizon']
        assert origin-h>=r['local_start'] and origin+f<=r['local_end_exclusive']
        return a[origin-h:origin,channel].copy(),a[origin:origin+f,channel].copy(),dict(
            dataset=r['dataset'],entity=r['entity'],role=r['role'],channel=channel,
            history_start=origin-h,origin=origin,target_end_exclusive=origin+f)
    def batch(self,n,domain=None):
        # A batch stays in one domain to preserve the original TF shared-order semantics.
        domain=domain or self.rng.choice(self.domains)
        choices=[r for r in self.records if r['dataset']==domain];assert choices
        batch=[]
        for _ in range(n):
            r=choices[self.rng.integers(len(choices))]
            batch.append(self.take(r,int(self.rng.choice(r['forecast_origins'])),
                                   int(self.rng.integers(len(r['columns'])))))
        return np.stack([x[0] for x in batch])[:,:,None],np.stack([x[1] for x in batch])[:,:,None],[x[2] for x in batch]
