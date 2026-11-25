"""Full-model frozen forward per window, with two preserved native evidences."""
import sys
import fcntl
import time
from pathlib import Path
import numpy as np
import torch
from .native_source_full import OUT,ARMS
from .native_tep_readout import inputs,category,FixedFrft,AXIS_SHA
from .external_frozen_bank import OUT as EXTERNAL
from .native_v_conditioning import load_native,fixed_native_rng,actual_mask_bits
from .native_v_conditioning_runner import setup,read,dump,sha


class Engine:
    def __init__(self,model,c):
        self.model=model
        self.c=c
        self.mask=torch.from_numpy(actual_mask_bits(c,"cuda")).cuda().reshape(10*c,20,1)
        self.transform=FixedFrft()
        self.graphs={}

    def forward(self,x,key):
        m=self.model
        t=x.permute(0,2,1).reshape(self.c,100)
        order=m.net(t.mean(0)).squeeze()
        z=self.transform.forward(t.to(torch.complex64),order,key,1)
        def patch(z,layer):
            a=layer(z.unfold(-1,5,5))
            return a.unsqueeze(0).expand(10,-1,-1,-1).reshape(10*self.c,20,64).masked_fill(self.mask,0.)
        h=m.encoder(patch(z.real,m.real_input_embed),patch(z.imag,m.imag_input_embed),order)
        r=self.transform.forward(m.decoder(h),order,key,-1).abs().reshape(10,1,self.c,100).permute(0,1,3,2)
        return torch.stack([order,m.cal_anomaly_score(x,r,None,1)[0,99],
                            (r[:,0,99].mean(0)-x[0,99]).square().mean()])

    def block(self,batch):
        approx=self.model.net(batch.mean(2)).reshape(-1).cpu().numpy()
        keys=category(approx)
        values=torch.empty((len(batch),3),device="cuda")
        for key in np.unique(keys):
            ix=np.flatnonzero(keys==key)
            if key not in self.graphs:
                x=batch[ix[0]:ix[0]+1].contiguous()
                true=int(category(self.model.net(x[0].T.mean(0)).item()))
                if true!=key:
                    for i in ix:
                        xx=batch[i:i+1]
                        k=int(category(self.model.net(xx[0].T.mean(0)).item()))
                        values[i]=self.forward(xx,k)
                    continue
                static=x.clone()
                for _ in range(3):
                    result=self.forward(static,int(key))
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result=self.forward(static,int(key))
                self.graphs[key]=(graph,static,result)
                self.check(x)
            graph,static,result=self.graphs[key]
            for i in ix:
                static.copy_(batch[i:i+1]);graph.replay();values[i].copy_(result)
        out=values.cpu().numpy()
        for i in np.flatnonzero(category(out[:,0])!=keys):
            out[i]=self.forward(batch[i:i+1],int(category(out[i,0]))).cpu().numpy()
        return out[:,1:]

    def check(self,x):
        key=int(category(self.model.net(x[0].T.mean(0)).item()))
        fast=self.forward(x,key)
        with fixed_native_rng("cuda"):
            _,r=self.model(x,norm=0,mask_mode="c",copies=10)
        direct=torch.stack([self.model.cal_anomaly_score(x,r,None,1)[0,99],
                            (r[:,0,99].mean(0)-x[0,99]).square().mean()])
        torch.testing.assert_close(fast[1:],direct,rtol=0,atol=0)
        return direct


@torch.inference_mode()
def score(w,lane):
    setup()
    root=w/OUT
    models=[];weights=[]
    for arm in ARMS:
        p=root/arm/"selected.pt";rec=read(p.with_name("SELECTION.json"))
        assert sha(p)==rec["selected_checkpoint_sha256"]
        m=load_native(w,"cuda")
        state=torch.load(p,map_location="cpu",weights_only=True)
        assert state["update"]==rec["selected_step"]
        m.load_state_dict(state["model"],strict=True);m.requires_grad_(False)
        models.append(m);weights.append(sha(p))
    manifest=w/EXTERNAL/"AXES_AND_INPUTS.json"
    assert sha(manifest)==AXIS_SHA
    jobs=sorted(read(manifest)["records"],key=lambda r:-r["test_points"])
    engines={}
    for rec in jobs[lane::3]+jobs:
        d,e=rec["domain"],rec["entity"]
        dest=root/"scores"/d;dest.mkdir(parents=True,exist_ok=True)
        bank=dest/(e+".npz")
        with bank.with_suffix(".lock").open("a") as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if bank.with_suffix(".json").exists():
                assert read(bank.with_suffix(".json"))["weight_shas"]==weights
                continue
            tick=time.perf_counter()
            windows,ci,qi,inverse,_=inputs(w,d,e)
            np.testing.assert_array_equal(qi,np.arange(*rec["causal_axis"]))
            c=windows.shape[-1]
            if c not in engines:engines[c]=[Engine(m,c) for m in models]
            active,reverse=np.unique(inverse,return_inverse=True)
            values=np.empty((len(active),3,2),np.float32)
            checkids=np.unique(np.linspace(0,len(active)-1,7,dtype=int))
            for j,engine in enumerate(engines[c]):
                for idx in checkids:
                    x=windows[active[idx]:active[idx]+1].contiguous().cuda()
                    direct=engine.check(x).cpu().numpy()
                    np.testing.assert_array_equal(engine.block(x)[0],direct)
            folder=dest/(e+".parts");folder.mkdir(exist_ok=True)
            for lo in range(0,len(active),512):
                ix=active[lo:lo+512];path=folder/f"{lo}.npz"
                if path.exists():
                    with np.load(path) as z:
                        np.testing.assert_array_equal(z["weights"],weights)
                        np.testing.assert_array_equal(z["window_ids"],ix)
                        values[lo:lo+len(ix)]=z["values"]
                else:
                    x=windows[ix].contiguous().cuda()
                    for j,engine in enumerate(engines[c]):
                        values[lo:lo+len(ix),j]=engine.block(x)
                    tmp=path.with_suffix(".part.npz")
                    np.savez(tmp,weights=weights,window_ids=ix,values=values[lo:lo+len(ix)])
                    tmp.replace(path)
            out=values[reverse]
            assert np.isfinite(out).all()
            tmp=bank.with_suffix(".part.npz")
            formal=np.stack([out[:,0,0],out[:,1,0],out[:,2,1]],axis=1)
            np.savez_compressed(tmp,keys=ARMS,evidence=["V","E"],cal=formal[:len(ci)],query=formal[len(ci):],
                cal_VE=out[:len(ci)],query_VE=out[len(ci):],
                cal_index=ci,query_index=qi,cal_window_start=ci-99,query_window_start=qi-99,
                cal_available_index=ci,query_available_index=qi)
            tmp.replace(bank)
            dump(bank.with_suffix(".json"),dict(domain=d,entity=e,sha256=sha(bank),weight_shas=weights,
                unique_windows=len(active),cal_points=len(ci),query_points=len(qi),native_parity_points=7,
                seconds=time.perf_counter()-tick,peak_reserved=torch.cuda.max_memory_reserved(),labels_read=False))
            print(d,e,round(time.perf_counter()-tick,1),flush=True)


if __name__=="__main__":
    score(Path.cwd(),int(sys.argv[1]))
