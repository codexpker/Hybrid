"""Same-process past R / future X synthetic source; controlled author generator copy.

Only preprocessing summary support changes: univariate and DAG centering/range
and the noise max-change branch statistic see R, never the later X suffix.
Original author files remain unchanged; generated copies and edits are hashed.
"""
import hashlib, json, os, random, shutil, sys
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def controlled_generator(author_root, output, reference_length=512):
    author_root=Path(author_root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    src=output/"src";src.mkdir();edits=[]
    replacements={
      "ts_multi_generator.py":[
       ("mean = np.mean(timeseries)",f"mean = np.mean(timeseries[:{reference_length}])"),
       ("np.any(np.abs(scaled_timeseries) >= SCALE_THRESHOLD)",f"np.any(np.abs(scaled_timeseries[:{reference_length}]) >= SCALE_THRESHOLD)"),
       ("np.max(np.abs(scaled_timeseries)) / SCALE_THRESHOLD",f"np.max(np.abs(scaled_timeseries[:{reference_length}])) / SCALE_THRESHOLD"),
       ("mean = np.mean(y_0)",f"mean = np.mean(y_0[:{reference_length}])"),
       ("np.any(np.abs(y_0) >= SCALE_THRESHOLD)",f"np.any(np.abs(y_0[:{reference_length}]) >= SCALE_THRESHOLD)"),
       ("np.max(np.abs(y_0)) / SCALE_THRESHOLD",f"np.max(np.abs(y_0[:{reference_length}])) / SCALE_THRESHOLD")],
      "ts_generator.py":[("max_change = np.abs(np.max(y) - np.min(y))",f"max_change = np.abs(np.max(y[:{reference_length}]) - np.min(y[:{reference_length}]))")]
    }
    for p in sorted((author_root/"src").glob("*.py")):
        text=p.read_text(encoding="utf-8");original=sha(p)
        for before,after in replacements.get(p.name,[]):
            count=text.count(before)
            if count not in ([2] if "timeseries" in before else [1]):
                raise ValueError("Unexpected pinned generator text count: "+p.name+" "+before+" "+str(count))
            text=text.replace(before,after);edits.append(dict(file=p.name,before=before,after=after,count=count))
        (src/p.name).write_text(text,encoding="utf-8")
    shutil.copytree(author_root/"config",output/"config")
    manifest=dict(author_files_sha256={p.name:sha(p)for p in sorted((author_root/"src").glob("*.py"))},controlled_files_sha256={p.name:sha(p)for p in sorted(src.glob("*.py"))},edits=edits,reference_length=reference_length,not_unmodified_official_generator=True)
    (output/"CONTROLLED_GENERATOR.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    os.environ["CHATTS_CONFIG_PATH"]=str((output/"config/synthetic.json").resolve())
    sys.path.insert(0,str(src.resolve()))
    import generate_dataset
    if not Path(generate_dataset.__file__).resolve().is_relative_to(src.resolve()):
        raise ValueError("Shadowed controlled generator import")
    return generate_dataset._generate_single_sample_worker,manifest

def legal_pair(sample, reference_length, scored_length):
    x=np.asarray(sample["time_series"]);y=np.asarray(sample["labels"]);normal=np.asarray(sample["normal_time_series"])
    if len(x)!=reference_length+scored_length or x.ndim!=2 or y.shape!=(len(x),):return False,"shape"
    if not np.isfinite(x).all() or not np.isfinite(normal).all():return False,"nonfinite"
    if (y[:reference_length]>.5).any():return False,"reference_label"
    if not np.array_equal(x[:reference_length],normal[:reference_length]):return False,"reference_actual_deviation"
    return True,"LEGAL"

def reference_statistics(reference):
    import torch
    # Same estimator as author collate; only the summary pool is restricted.
    reference=torch.as_tensor(reference,dtype=torch.float32)
    return reference.mean(dim=0,keepdim=True),reference.std(dim=0,keepdim=True)+1e-4

def make_batch(samples, ids, device, mask_seed, patch_size, statistics):
    import torch
    if statistics not in ("batch","reference"):raise ValueError(statistics)
    xs=[torch.as_tensor(samples[i]["time_series"],dtype=torch.float32)for i in ids]
    refs=[torch.as_tensor(samples[i]["reference"],dtype=torch.float32)for i in ids]
    labels=[torch.as_tensor(samples[i]["labels"],dtype=torch.long)for i in ids]
    if statistics=="batch":
        joined=torch.cat(xs,dim=0);mean=joined.mean(0,keepdim=True);std=joined.std(0,keepdim=True)+1e-4
        xs=[(x-mean)/std for x in xs]
    else:
        pairs=[reference_statistics(r)for r in refs];xs=[(x-m)/s for x,(m,s)in zip(xs,pairs)]
    x=torch.nn.utils.rnn.pad_sequence(xs,batch_first=True,padding_value=0)
    y=torch.nn.utils.rnn.pad_sequence(labels,batch_first=True,padding_value=-1)
    valid=torch.arange(x.shape[1])[None,:]<torch.tensor([len(a)for a in xs])[:,None]
    gen=torch.Generator().manual_seed(mask_seed);mask=torch.zeros_like(valid)
    for b,a in enumerate(xs):
        npatch=(len(a)+patch_size-1)//patch_size
        for pos in torch.randperm(npatch,generator=gen)[:int(npatch*.15)]:
            lo=int(pos)*patch_size;mask[b,lo:min(lo+patch_size,len(a))]=True
    masked=x.clone();masked[mask]=torch.randn(masked[mask].shape,generator=gen)*.1
    return {k:v.to(device)for k,v in dict(time_series=x,labels=y,attention_mask=valid,masked_time_series=masked,mask=mask).items()}
