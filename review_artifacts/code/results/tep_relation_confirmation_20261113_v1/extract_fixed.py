"""One CPU parse of existing RData originals, retain only fixed runs201..220."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib,time,datetime,gc,traceback,resource
import numpy as np,pyreadr
P=Path(__file__).resolve().parent;R=P.parents[1];started=time.time()
def sha(f):
    h=hashlib.sha256()
    with Path(f).open('rb')as src:
        for b in iter(lambda:src.read(2**20),b''):h.update(b)
    return h.hexdigest()
def dump(name,x):(P/name).write_text(json.dumps(x,indent=2,allow_nan=False))
config=json.loads((P/'FIXED_PROTOCOL.json').read_text());dump('EXTRACTION_STARTED.json',dict(pid=os.getpid(),unix=started,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(P/'FIXED_PROTOCOL.json'),source_sha256=sha(__file__),CPU_only=True))
try:
    data=P/'data';data.mkdir(exist_ok=False);receipts=[]
    for role,cfg in [('normal',config['new_long_normal']),('faulty',config['new_fault_test'])]:
        path=R/cfg['path'];assert sha(path)==cfg['sha256'];t=time.monotonic();objects=pyreadr.read_r(str(path));assert cfg['object']in objects;df=objects[cfg['object']];expected=(480000,55)if role=='normal'else(9600000,55);assert df.shape==expected
        selected=df.loc[df.simulationRun.between(201,220),['faultNumber','simulationRun','sample','xmeas_9','xmv_10','xmeas_21']].copy();del df,objects;gc.collect();assert not selected.duplicated(['faultNumber','simulationRun','sample']).any()and np.isfinite(selected.to_numpy()).all();assert set(selected.simulationRun)==set(range(201,221))
        faults=[0]if role=='normal'else list(range(1,21));assert set(selected.faultNumber)==set(faults);values=[];ids=[]
        for fault in faults:
            for run in range(201,221):
                g=selected[(selected.faultNumber==fault)&(selected.simulationRun==run)].sort_values('sample');assert np.array_equal(g['sample'].to_numpy(),np.arange(1,961));values.append(g[['xmeas_9','xmv_10','xmeas_21']].to_numpy(np.float64));ids.append([fault,run])
        dest=data/(role+'.npz');np.savez_compressed(dest,values=np.stack(values),fault_run=np.array(ids),samples=np.arange(1,961));receipts.append(dict(role=role,source_path=str(path),source_sha256=cfg['sha256'],original_shape=expected,object=cfg['object'],selected_runs=[201,220],selected_rows=len(selected),series=len(values),shape=list(np.stack(values).shape),array_columns=['xmeas_9','xmv_10','xmeas_21'],output_path=str(dest),output_sha256=sha(dest),seconds=time.monotonic()-t));print(json.dumps(dict(event='EXTRACTED',**receipts[-1])),flush=True);del selected,values;gc.collect()
    dump('EXTRACTION_RECEIPT.json',dict(status='COMPLETED',exit_code=0,seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,receipts=receipts,unselected_run_model_scores_computed=False,model_training=False,GPU_hours=0));print('STAGE_COMPLETED_EXTRACTION',flush=True)
except BaseException:
    dump('EXTRACTION_FAILED.json',dict(status='FAILED',seconds=time.time()-started,traceback=traceback.format_exc()));raise
