"""Pinned official Time-RCD inference; permission changes only normalization.

No training, head replacement, source mutation or label access. A retains the
official whole-role normalization; B applies raw FIT statistics exactly once.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np

ARMS = ('A_official_role', 'B_fit_only')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def normalize_once(data, arm, fit_mean=None, fit_std=None):
    x=np.asarray(data,dtype=np.float64)
    if x.ndim!=2 or not len(x) or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError('Finite nonempty raw time series required')
    if arm=='A_official_role':
        mean=np.mean(x,axis=0);std=np.std(x,axis=0);std=np.where(std==0,1e-8,std)
    elif arm=='B_fit_only':
        mean,std=np.asarray(fit_mean),np.asarray(fit_std)
        if mean.shape!=(x.shape[1],)or std.shape!=mean.shape or not np.isfinite(mean).all()or not np.isfinite(std).all()or np.any(std<=0):
            raise ValueError('Explicit finite FIT mean and official-safe std required')
    else:raise ValueError('Unknown fixed normalization permission')
    return (x-mean)/std,dict(mean=mean.tolist(),std_ddof0_safe=std.tolist(),normalization_count=1,
                           statistics_role='current complete role'if arm=='A_official_role'else'FIT only')


def window_layout(index, win_size=5000):
    index=np.asarray(index)
    if index.ndim!=1 or not len(index)or not np.issubdtype(index.dtype,np.integer)or np.any(np.diff(index)!=1):
        raise ValueError('Nonempty contiguous original int index required')
    width=min(len(index),win_size);starts=np.arange(0,len(index),width,dtype=np.int64)
    lengths=np.minimum(width,len(index)-starts);window_id=np.arange(len(index))//width
    ends=index[starts+lengths-1]
    valid=np.arange(width)[None,:]<lengths[:,None]
    return dict(effective_window=width,starts=starts,valid_lengths=lengths,window_id=window_id,
                window_end_index=ends[window_id],window_start_index=index[starts][window_id],
                window_valid_mask=valid,padding_points=int((~valid).sum()),
                internal_patch_padding=(-width)%16)


def padded_windows(normalized, width):
    padding=(-len(normalized))%width
    data=normalized
    if padding:data=np.vstack([data,np.repeat(data[-1:,:],padding,axis=0)])
    return data


def import_official(source_root, expected_files):
    source_root=Path(source_root).resolve()
    for relative,expected in expected_files.items():
        if sha(source_root/relative)!=expected:raise ValueError('Pinned author source mismatch: '+relative)
    sys.path.insert(0,str(source_root))
    module=importlib.import_module('time_rcd')
    if not Path(module.__file__).resolve().is_relative_to(source_root):
        raise ValueError('Shadowed time_rcd import')
    return module.TimeRCDDetector


def detector_from_config(workspace, config, channels, device):
    def resolve(value):
        p=Path(value);return p if p.is_absolute()else Path(workspace)/p
    checkpoint=resolve(config['checkpoint'])
    if sha(checkpoint)!=config['checkpoint_sha256']:raise ValueError('Official multi checkpoint SHA mismatch')
    Detector=import_official(resolve(config['author_root']),config['author_files_sha256'])
    variant=config.get('variant','multi')
    if (variant=='uni' and channels!=1) or (variant=='multi' and channels<2):
        raise ValueError('Channel count is incompatible with the official checkpoint variant')
    detector=Detector.from_local(checkpoint,variant=variant,win_size=5000,batch_size=1,device=str(device))
    detector._ensure_tester(channels)
    detector._tester.model.eval()
    return detector


def predict_with_cache(detector,data,index,arm,fit_mean=None,fit_std=None):
    import torch
    from time_rcd._inference import _WindowDataset
    raw=np.asarray(data,dtype=np.float64)
    layout=window_layout(index,5000)
    normalized,normalization=normalize_once(raw,arm,fit_mean,fit_std)
    if arm=='A_official_role':
        # Direct use of the pinned official dataset, including precision and pad.
        dataset=_WindowDataset(raw,layout['effective_window'])
        normalized_padded=dataset.data
    else:
        normalized_padded=padded_windows(normalized,layout['effective_window'])
    model=detector._tester.model;device=detector._tester.device
    probs=[];differences=[];head_values=[]
    with torch.no_grad():
        for n,start in enumerate(layout['starts']):
            width=layout['effective_window']
            x=torch.tensor(normalized_padded[start:start+width],dtype=torch.float32,device=device)[None]
            mask=torch.tensor(layout['window_valid_mask'][n],dtype=torch.bool,device=device)[None]
            embeddings=model(time_series=x,mask=mask)
            head=model.anomaly_head(embeddings).mean(dim=-2)
            prob=torch.nn.functional.softmax(head,dim=-1)[...,1]
            diff=head[...,1]-head[...,0]
            length=int(layout['valid_lengths'][n])
            probs.append(prob[0,:length].cpu().numpy());differences.append(diff[0,:length].cpu().numpy())
            head_values.append(head[0,:length].cpu().numpy())
    score=np.concatenate(probs);logit=np.concatenate(differences);heads=np.concatenate(head_values)
    if not np.isfinite(score).all()or not np.isfinite(logit).all()or not np.isfinite(heads).all():
        raise ValueError('Nonfinite official inference result')
    available=np.full(len(index),index[-1],dtype=np.int64)if arm=='A_official_role'else layout['window_end_index']
    bank=dict(index=np.asarray(index),score=score,prob=score,logit=logit,head_logits=heads,
        available_index=available,max_input_index=available,window_end_index=layout['window_end_index'],
        window_start_index=layout['window_start_index'],window_id=layout['window_id'],valid=np.ones(len(index),bool),
        window_starts_offset=layout['starts'],window_valid_lengths=layout['valid_lengths'],
        window_valid_mask=layout['window_valid_mask'],effective_window=np.array(layout['effective_window']),
        padding_points=np.array(layout['padding_points']),internal_patch_padding=np.array(layout['internal_patch_padding']))
    return bank,normalization
