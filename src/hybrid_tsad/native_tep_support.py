"""Training-only TEP support and fixed source schedules for endpoint supervision."""
from pathlib import Path
import hashlib
import numpy as np
from . import native_event_rank_training as old
from .native_position_aligned_training import PRIOR, PLAN_SHA
from .left_swat_author import read, dump, sha

OUT = Path("results/native_tep_support_20261125_v1")
ARMS = ("single_V", "expanded_V", "expanded_E")
RUNS = np.r_[1:21, 61:66]
RAW = Path("data/tep_rieth2017_v1")
FILES = (
    ("TEP_FaultFree_Training.RData", "fault_free_training", "ec126484534331f85001d8c4ebce6d17", 1),
    ("TEP_Faulty_Training.RData", "faulty_training", "c5f594d54c47e620ff877feb58407fda", 20),
)


def extract(w):
    import pyreadr
    out = w/OUT
    out.mkdir(parents=True, exist_ok=True)
    columns = None
    sources = []
    for filename, obj, md5, count in FILES:
        path = w/RAW/filename
        with path.open("rb") as f:
            assert hashlib.file_digest(f, "md5").hexdigest() == md5
        frame = pyreadr.read_r(str(path))[obj]
        assert frame.shape == (count*250000, 55)
        assert list(frame.columns[:3]) == ["faultNumber", "simulationRun", "sample"]
        names = list(frame.columns[3:])
        assert len(set(names)) == 52 and (columns is None or columns == names)
        columns = names
        frame = frame.loc[frame.simulationRun.isin(RUNS)]
        faults = [0] if count == 1 else range(1,21)
        data = np.empty((count,len(RUNS),500,52),np.float64)
        groups = frame.groupby(["faultNumber","simulationRun"])
        assert len(groups) == count*len(RUNS)
        for i, fault in enumerate(faults):
            for j, run in enumerate(RUNS):
                block = groups.get_group((fault,run)).sort_values("sample")
                np.testing.assert_array_equal(block["sample"],np.arange(1,501))
                data[i,j] = block[names].to_numpy()
        assert np.isfinite(data).all()
        if count == 1:
            fit = data[0,:20].reshape(-1,52)
            mean, std = fit.mean(0), fit.std(0)
            std[std == 0] = 1.
            np.savez(out/"tep_scaler.npz",mean=mean,std=std,columns=columns)
        np.save(out/(obj+".npy"),((data-mean)/std).astype(np.float32))
        sources.append(dict(filename=filename,md5=md5,sha256=sha(path),object=obj,columns=columns,
                            derived_sha256=sha(out/(obj+".npy"))))
        del frame,groups,data
    dump(out/"INPUT.json",dict(doi="10.7910/DVN/6C3JR1",files=sources,fit_runs=RUNS[:20].tolist(),
        select_runs=RUNS[20:].tolist(),train_fault_start_zero_based=20,first_legal_positive_endpoint=119,
        sampling_minutes=3,points_per_run=500,scaler_sha256=sha(out/"tep_scaler.npz")))


def prepare(w):
    root = w/OUT
    assert sha(w/PRIOR/"PLAN.npz") == PLAN_SHA
    plan = old.load_plan(w/PRIOR)
    rng = np.random.default_rng(np.random.SeedSequence([2025, 3031242]))
    steps = old.STEPS
    copy_domain = np.arange(steps*32).reshape(steps,32)%3
    # SWaT and SMD draws are prefixes of the original equal-domain plan.
    normal_run = rng.integers(20,size=(steps,11))
    normal_end = rng.integers(99,500,size=(steps,11))
    shared = np.zeros((steps,15),bool)
    for j in range(0,steps,2):
        order = rng.permutation(15)
        shared[j,order[:8]] = True
        shared[j+1,order[8:]] = True
    n = int((~shared).sum())
    faults = np.concatenate([rng.permutation(20) for _ in range(n//20)])
    tep_pair = np.zeros((steps,15,5),int)
    runs = np.empty(n,int)
    phases = np.empty(n,int)
    for fault in range(20):
        take = np.flatnonzero(faults==fault)
        runs[take] = np.concatenate([rng.permutation(20) for _ in range(5)])[:len(take)]
        phases[take] = rng.permutation(np.arange(len(take))%3)
    # 381 complete fault endpoints split into three equal strata.
    tep_pair[~shared] = np.column_stack([faults,runs,119+127*phases+rng.integers(127,size=n),
                                        rng.integers(20,size=n),rng.integers(99,500,size=n)])
    select = np.array([(f,j,119+127*k+rng.integers(127),rng.integers(20,25),rng.integers(99,500))
                       for f in range(20) for j in range(20,25) for k in range(3)])
    assert np.bincount(copy_domain.ravel()).tolist()==[2560]*3
    assert np.all(shared.sum(axis=0)==120) and np.bincount(faults).tolist()==[90]*20
    assert len(set(zip(faults,runs)))==400 and select.shape==(300,5)
    with (root/"SUPPORT_PLAN.npz").open("xb") as f:
        np.savez_compressed(f,copy_domain=copy_domain,normal_run=normal_run,normal_end=normal_end,
                            shared=shared,tep_pair=tep_pair,select=select)
    dump(root/"PLAN.json",dict(seed=2025,base_plan_sha256=PLAN_SHA,plan_sha256=sha(root/"SUPPORT_PLAN.npz"),
        source_input_sha256=sha(root/"INPUT.json"),arms=ARMS,normal_exposures=[2560]*3,
        normal_channel_window_exposures=2560*(31+38+52),single_pair_channel_window_exposures=3600*2*31,
        expanded_pair_channel_window_exposures=1800*2*(31+52),fit_tep_fault_runs=400,select_tep_fault_runs=100,
        select_domain_weights=[0.5,0.5],select_steps=list(old.SELECT_STEPS),swat_fit_events=5,swat_select_events=4,
        fit_weight="SWaT normalized original stratum mass with inverse inclusion probability; TEP equal fault and stratum mass",
        scale="original official decoder median on original 4632 SWaT FIT normal endpoints, separate for V and E",
        test_read=False))
