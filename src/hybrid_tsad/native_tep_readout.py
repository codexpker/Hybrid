"""Exact per-window TimeRadar readout with shared encoder and isolated native masks."""
import argparse
import fcntl
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from . import native_event_smd28 as smd
from . import native_event_space as nasa
from .native_e_vs_v_psm import inputs as psm_inputs, OUT as PSM_BANK
from .native_e_vs_v_smd import OUT as SMD_BANK
from .native_e_vs_v_space import OUT as NASA_BANK
from .native_tep_support import OUT, ARMS
from .native_v_conditioning import load_native, fixed_native_rng, actual_mask_bits
from .native_v_conditioning_runner import setup, read, dump, sha
from .external_frozen_bank import OUT as EXTERNAL

AXIS_SHA = "a79c649f93d65017c2fcb2f1827a4d1c8305264eb817e0431deae7980d212875"


def inputs(w,d,e):
    if d=="PSM":
        x,ci,_ = psm_inputs(w)
        qi = np.arange(99,len(x))
        bank = w/PSM_BANK/"banks.npz"
        receipt = bank.with_name("SCORES.json")
        windows = torch.from_numpy(x).unfold(0,100,1).transpose(1,2)
        with np.load(bank) as z:
            prior = z["query"][:,z["keys"].tolist().index("official_V")]
            np.testing.assert_array_equal(z["index"],qi)
            np.testing.assert_array_equal(z["cal_index"],ci)
        inverse = np.r_[ci-99,np.arange(len(qi))]
    else:
        if d=="SMD":
            rec = next(r for r in smd.records(w) if r["id"]==e)
            cx,co = smd.data(w,rec,"CAL")
            qx,qo = smd.data(w,rec,"QUERY")
            assert qo==0
            ci = np.linspace(99,len(cx)-1,512,dtype=np.int64)+co
            qi = np.arange(99,len(qx))
            x = torch.from_numpy(np.concatenate([cx,qx]))
            windows = x.unfold(0,100,1).transpose(1,2)
            inverse = np.r_[ci-co-99,qi-99+len(cx)]
            # Only the listed indices are legal, no CAL/QUERY crossing.
            bank = w/SMD_BANK/(e+".npz")
        else:
            x,ci,qi,_ = nasa.read_roles(w,d,e)
            flat = np.ascontiguousarray(np.concatenate(x))
            _,first,inverse = np.unique(flat.view("V400").ravel(),return_index=True,return_inverse=True)
            windows = torch.from_numpy(flat[first])[:,:,None]
            bank = w/NASA_BANK/d/(e+".npz")
        receipt = bank.with_suffix(".json")
        with np.load(bank) as z:
            pos = z["keys"].tolist().index("official_V")
            np.testing.assert_array_equal(z["query_index"],qi)
            np.testing.assert_array_equal(z["cal_index"],ci)
            prior = np.r_[z["cal"][:,pos],z["query"][:,pos]]
    assert sha(bank)==read(receipt)["sha256"]
    if d=="PSM":
        prior = prior[inverse]
    return windows,ci,qi,np.asarray(inverse).reshape(-1),prior


class FixedFrft:
    """Upstream torch-frft arithmetic specialized to the fixed even length 100.

    Keep constant tensors outside CUDA graph capture. At sigmoid order in [0,1]
    the upstream forward/inverse paths have the same three branch boundaries.
    """
    def __init__(self):
        self.x = torch.arange(-200,200,device="cuda")/torch.sqrt(torch.tensor(400,device="cuda"))
        self.t = torch.arange(-399,400,device="cuda")/torch.sqrt(torch.tensor(400,device="cuda"))
        self.root = torch.sqrt(torch.tensor(400,device="cuda"))
        self.double = torch.ones(100,device="cuda")
        self.double[0]=2
        self.crop = torch.arange(399,799,device="cuda")
        self.trim = torch.arange(100,300,device="cuda")
        self.constants = {j:self.coefficients(torch.tensor(float(j))) for j in (-1,1)}

    def coefficients(self,a):
        phi = a*torch.pi/2
        alpha = -1j*torch.pi*torch.tan(phi/2)
        beta = 1j*torch.pi/torch.sin(phi)
        amplitude = torch.exp(-1j*(torch.pi*torch.sign(torch.sin(phi))/4-phi/2))/torch.sqrt(torch.abs(torch.sin(phi)))
        chirp = torch.exp(alpha*self.x**2)
        kernel = torch.fft.fft(torch.exp(beta*self.t**2),n=2048)
        return chirp,kernel,amplitude*chirp

    def core(self,x,coefficients):
        chirp,kernel,last = coefficients
        conv = torch.fft.ifft(torch.fft.fft(x*chirp,n=2048)*kernel)
        return conv.index_select(-1,self.crop)*last/self.root

    def forward(self,x,a,key,sign):
        from torch_frft.frft_module import _bizinter, _bizdec
        a = torch.fmod(a*sign,4)
        if key==0:
            return x+a*torch.zeros_like(x)
        z = _bizinter(x,dim=-1)
        pad = torch.zeros_like(z[...,:100])
        z = torch.cat([pad,z,pad],dim=-1)
        if key==1:
            z = self.core(z,self.constants[sign])
            a = a-sign
        z = self.core(z,self.coefficients(a)).index_select(-1,self.trim)
        return _bizdec(z,dim=-1)*self.double


class JointReadout(torch.nn.Module):
    """The official patch encoder runs once for each independent B=1 order."""
    def __init__(self,nets,c):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.register_buffer("mask",torch.from_numpy(actual_mask_bits(c,"cuda")).cuda().reshape(10*c,20,1))
        self.c = c
        self.frft = FixedFrft()

    def forward(self,x,key):
        model = self.nets[0]
        t = x.permute(0,2,1).reshape(self.c,100)
        order = model.net(t.mean(dim=0)).squeeze()
        z = self.frft.forward(t.to(torch.complex64),order,key,1)
        def patches(component,layer):
            v = layer(component.unfold(-1,5,5))
            return v.unsqueeze(0).expand(10,-1,-1,-1).reshape(10*self.c,20,64).masked_fill(self.mask,0.)
        enc = model.encoder(patches(z.real,model.real_input_embed),patches(z.imag,model.imag_input_embed),order)
        scores = []
        projections = torch.stack([net.decoder(enc) for net in self.nets[1:]])
        all_copies = self.frft.forward(projections,order,key,-1).abs()
        for j,net in enumerate(self.nets[1:]):
            copies = all_copies[j].reshape(10,1,self.c,100).permute(0,1,3,2)
            if j==2:
                s = (copies[:,0,99].mean(0)-x[0,99]).square().mean()
            else:
                s = net.cal_anomaly_score(x,copies,None,1)[0,99]
            scores.append(s)
        return torch.stack([order,*scores])


def category(a):
    return np.where(a==0,0,np.where(a<0.5,1,2))


class Replay:
    def __init__(self,net):
        self.net = net
        self.programs = {}

    def build(self,x,key):
        # The upstream FRFT uses scalar order to select a finite control-flow branch.
        # Capture one branch; every replay still computes the actual order from x.
        fn = lambda value:self.net(value,key)
        static = x.clone().contiguous()
        for _ in range(3):
            result = fn(static)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn(static)
        self.programs[key] = graph,static,result
        self.check_direct(x,fn)

    def check_direct(self,x,result_fn):
        out = result_fn(x)
        for j,model in enumerate(self.net.nets[1:]):
            with fixed_native_rng(x.device):
                _,copy = model(x,norm=0,mask_mode="c",copies=10)
            score = ((copy[:,0,99].mean(0)-x[0,99]).square().mean() if j==2 else
                     model.cal_anomaly_score(x,copy,None,1)[0,99])
            torch.testing.assert_close(score,out[1+j],atol=0,rtol=0)

    def block(self,windows):
        # The preliminary order only dispatches graphs. Check the exact B=1 branch afterwards.
        approx = self.net.nets[0].net(windows.mean(2)).reshape(-1)
        keys = category(approx.cpu().numpy())
        out = torch.empty((len(windows),4),device="cuda")
        for key in np.unique(keys):
            ix = np.flatnonzero(keys==key)
            if key not in self.programs:
                x = windows[ix[0]:ix[0]+1]
                exact = self.net.nets[0].net(x[0].T.mean(0)).item()
                if category(exact)!=key:
                    for i in ix:
                        xx=windows[i:i+1]
                        actual=self.net.nets[0].net(xx[0].T.mean(0)).item()
                        out[i]=self.net(xx,int(category(actual)))
                    continue
                self.build(x,int(key))
            graph,static,result = self.programs[key]
            for i in ix:
                static.copy_(windows[i:i+1])
                graph.replay()
                out[i].copy_(result)
        vals = out.cpu().numpy()
        for i in np.flatnonzero(category(vals[:,0])!=keys):
            vals[i]=self.net(windows[i:i+1],int(category(vals[i,0]))).cpu().numpy()
        return vals[:,1:]


def load_models(w):
    base = load_native(w,"cuda")
    nets = [base]
    ids = []
    for arm in ARMS:
        p = w/OUT/arm/"selected.pt"
        rec = read(p.with_name("SELECTION.json"))
        assert sha(p)==rec["selected_checkpoint_sha256"]
        state = torch.load(p,map_location="cpu",weights_only=True)
        assert state["update"]==rec["selected_step"] and state["arm"]==arm
        net = deepcopy(base)
        net.load_state_dict(state["model"],strict=True)
        assert all(torch.equal(v,base.state_dict()[k]) for k,v in net.state_dict().items() if not k.startswith("decoder."))
        nets.append(net)
        ids.append(sha(p))
    return nets,ids


@torch.inference_mode()
def score(w,lane,lanes):
    setup()
    nets,ids = load_models(w)
    manifest = w/EXTERNAL/"AXES_AND_INPUTS.json"
    assert sha(manifest)==AXIS_SHA
    jobs = sorted(read(manifest)["records"],key=lambda r:-r["test_points"])
    replays = {}
    for r in jobs[lane::lanes]+jobs:
        d,e = r["domain"],r["entity"]
        dest = w/OUT/"scores"/d
        dest.mkdir(parents=True,exist_ok=True)
        bank = dest/(e+".npz")
        with bank.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if bank.with_suffix(".json").exists():
                assert read(bank.with_suffix(".json"))["weight_shas"]==ids
                continue
            start = time.perf_counter()
            windows,ci,qi,inverse,prior = inputs(w,d,e)
            np.testing.assert_array_equal(qi,np.arange(*r["causal_axis"]))
            c = windows.shape[-1]
            if c not in replays:
                replays[c] = Replay(JointReadout(nets,c).eval())
            engine = replays[c]
            active,reverse = np.unique(inverse,return_inverse=True)
            check_rows = np.unique(np.linspace(0,len(inverse)-1,7,dtype=int))
            for i in check_rows:
                x = windows[inverse[i]:inverse[i]+1].contiguous().cuda()
                with fixed_native_rng("cuda"):
                    _,copies=nets[0](x,norm=0,mask_mode="c",copies=10)
                assert nets[0].cal_anomaly_score(x,copies,None,1)[0,99].item()==prior[i]
                fast=engine.block(x)[0]
                for j,net in enumerate(nets[1:]):
                    with fixed_native_rng("cuda"):
                        _,copies=net(x,norm=0,mask_mode="c",copies=10)
                    s = ((copies[:,0,99].mean(0)-x[0,99]).square().mean() if j==2 else
                         net.cal_anomaly_score(x,copies,None,1)[0,99])
                    assert fast[j]==s.item()
            values = np.empty((len(active),3),np.float32)
            parts = dest/(e+".parts")
            parts.mkdir(exist_ok=True)
            for lo in range(0,len(active),1024):
                ix = active[lo:lo+1024]
                path=parts/f"{lo}.npz"
                if path.exists():
                    with np.load(path) as z:
                        np.testing.assert_array_equal(ix,z["window_ids"])
                        np.testing.assert_array_equal(ids,z["weights"])
                        values[lo:lo+len(ix)]=z["values"]
                else:
                    batch = windows[ix].contiguous().cuda()
                    tick=time.monotonic()
                    values[lo:lo+len(ix)]=engine.block(batch)
                    tmp=path.with_suffix(".part.npz")
                    np.savez(tmp,window_ids=ix,weights=ids,values=values[lo:lo+len(ix)],seconds=time.monotonic()-tick)
                    tmp.replace(path)
            out = values[reverse]
            assert np.isfinite(out).all()
            tmp = bank.with_suffix(".part.npz")
            np.savez_compressed(tmp,keys=ARMS,query_index=qi,cal_index=ci,query=out[len(ci):],
                cal=out[:len(ci)],query_window_start=qi-99,cal_window_start=ci-99,query_available_index=qi,
                cal_available_index=ci)
            tmp.replace(bank)
            dump(bank.with_suffix(".json"),dict(domain=d,entity=e,sha256=sha(bank),weight_shas=ids,
                query_points=len(qi),cal_points=len(ci),unique_windows=len(active),channels=c,
                native_parity_points=len(check_rows),seconds=time.perf_counter()-start,peak_reserved=torch.cuda.max_memory_reserved(),
                lane=lane,labels_read=False))
            print(d,e,len(active),time.perf_counter()-start,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("lane",type=int)
    a=p.parse_args()
    score(Path.cwd(),a.lane,4)

