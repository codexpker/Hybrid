"""Posthoc paired G36 diagnosis. No model, fitting, inference, or threshold update.

The healthy counterpart is an analysis instrument, NEVER a deployable input.
Coincident alarms do not by themselves prove a common causal mechanism.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .g36_data import FILES, Y_COLUMNS, U_COLUMNS
from .g36_evaluation import confirm, load_bank, spans


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def window_all(mask, length):
    """Past-inclusive complete window; no padding passes as observed data."""
    mask = np.asarray(mask, bool)
    out = np.zeros(len(mask), bool)
    if len(mask) >= length:
        out[length-1:] = np.convolve(mask.astype(int), np.ones(length, int), 'valid') == length
    return out


def quadrants(healthy, fault):
    # 0 neither, 1 healthy only, 2 fault only, 3 both.
    return np.asarray(healthy, np.int8) + 2*np.asarray(fault, np.int8)


def average(values):
    return float(np.mean(values)) if len(values) else None


def longest(mask):
    return max([b-a for a, b in spans(mask)] or [0])


def run(root, source, output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    rules = dict(
        purpose='paired posthoc diagnosis only; not new scores, CAL, or model selection',
        same_time='original indices and seconds, outdoor weather exact; no time shifting',
        switch='existing baseline mode switch +/-6 points',
        steady='original injection start2100 +12 points through10020 inclusive',
        mode='current mismatch separate; any mismatch in52-point footprint also separate',
        observability='raw four-response exact equality; canonical nine-input52-window exact equality',
        numerical_envelope='per-channel maximum healthy/fault discrepancy BEFORE injection; not a sensor noise bound',
        continuous_interval='all spans archived; >=12 consecutive samples described as hour-scale exposure',
        alarm='original sealed threshold strict >; original causal C3 computed before strata',
        quadrants=['neither', 'healthy_only', 'fault_only', 'both'],
        limits=['overlapping strata are not additive independent samples',
                'same weather does not make endogenous histories identical',
                'joint alarm is co-occurrence, not proof of a common cause',
                'paired score differences are diagnostic only and never a new detector',
                'absence of exact equality does not establish practical detectability',
                'no instrument noise bound or repeated healthy simulations are available'])
    (output/'RULES.json').write_text(json.dumps(rules, indent=2), encoding='utf-8')
    data = root/'data'
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    banks = {k: Path(v) for k, v in json.loads((root/'bank_paths.json').read_text()).items()}
    threshold_file = root/'metrics/normal_false_alarms.csv'
    thresholds = pd.read_csv(threshold_file, float_precision='round_trip').set_index('method')
    tracked = [threshold_file, data/'manifest.json', data/'scaler.npz']
    tracked += list(data.glob('*.npz'))
    tracked += [p/f'{r}__QUERY.npz' for p in banks.values() for r in FILES]
    tracked += [source/(r+'.csv') for r in FILES]
    before = {str(p): sha(p) for p in set(tracked)}
    frames = {r: pd.read_csv(source/(r+'.csv'), float_precision='round_trip') for r in FILES}
    base = frames[FILES[0]].to_numpy(float)
    original_t = base[:, 0]
    assert np.all(np.diff(original_t) == 300)
    for r, frame in frames.items():
        assert sha(source/(r+'.csv')) == manifest['raw_audit'][r]['sha256']
        assert list(frame.columns) == list(frames[FILES[0]].columns)
        assert np.array_equal(frame.iloc[:, 0].to_numpy(), original_t)
        assert np.array_equal(frame.iloc[:, [48, 49]].to_numpy(), base[:, [48, 49]])
    eps = np.max(np.stack([np.max(np.abs(frames[r].iloc[:2100, Y_COLUMNS].to_numpy()
                                       -base[:2100, Y_COLUMNS]), axis=0) for r in FILES[1:]]), axis=0)
    with np.load(data/'scaler.npz') as z:
        scale = z['scale'][:4]
    with np.load(data/'BaselineSystem__QUERY.npz') as z:
        bx, raw_index = z['x'].copy(), z['index'].copy()
    index = raw_index[51:]
    assert np.array_equal(index, np.arange(2067,10021))
    baseline_mode = base[:,71]
    switching = np.zeros(len(base), bool)
    for p in np.flatnonzero(np.diff(baseline_mode) != 0)+1:
        switching[max(0,p-6):min(len(base),p+7)] = True
    injected = index >= 2100
    steady = index >= 2112
    alarm_rows, observation_rows, mismatch_rows, interval_rows, ridge_rows = [], [], [], [], []
    integrity = {}
    for r in FILES[1:]:
        f = frames[r].to_numpy(float)
        mode_equal = baseline_mode == f[:,71]
        aligned = window_all(mode_equal,52)[index]
        current_equal = mode_equal[index]
        with np.load(data/f'{r}__QUERY.npz') as z:
            assert np.array_equal(z['index'], raw_index)
            fx = z['x'].copy()
        equal_input52 = window_all(np.all(bx == fx, axis=1),52)[51:]
        delta_all = f[:,Y_COLUMNS] - base[:,Y_COLUMNS]
        absolute = np.abs(delta_all[index])
        exact_point = np.all(delta_all[index] == 0, axis=1)
        within_numeric = np.all(absolute <= eps+1e-12, axis=1)
        numeric52 = window_all(np.all(np.abs(delta_all) <= eps+1e-12,axis=1)
                               & mode_equal,52)[index]
        masks = dict(all_injected=injected, steady_all=steady,
            steady_current_mode_mismatch=steady & ~current_equal,
            steady_current_match_history_mismatch=steady & current_equal & ~aligned,
            steady_aligned=steady & aligned,
            steady_aligned_switch=steady & aligned & switching[index],
            steady_aligned_non_switch=steady & aligned & ~switching[index],
            steady_aligned_occupied=steady & aligned & (baseline_mode[index]>.5),
            steady_aligned_unoccupied=steady & aligned & (baseline_mode[index]<.5),
            injected_identical_inputs52=injected & equal_input52,
            injected_within_prefault_envelope52=injected & numeric52)
        for i in np.flatnonzero(~current_equal):
            mismatch_rows.append(dict(run=r,index=int(index[i]),time_s=float(original_t[index[i]]),
                healthy_mode=baseline_mode[index[i]],fault_mode=f[index[i],71],
                injected=bool(injected[i]),switching=bool(switching[index[i]])))
        for label, mask in [('four_response_exact',injected & exact_point),
                            ('nine_inputs52_exact',injected & equal_input52),
                            ('four_response_prefault_envelope',injected & within_numeric),
                            ('response_mode52_prefault_envelope',injected & numeric52),
                            ('mode_history_mismatch',injected & ~aligned)]:
            for a,b in spans(mask):
                interval_rows.append(dict(run=r,kind=label,start_index=int(index[a]),
                    end_index=int(index[b-1]),points=b-a,sample_exposure_minutes=(b-a)*5,
                    elapsed_minutes=(b-a-1)*5,hour_scale=b-a>=12))
        for name, mask in masks.items():
            row = dict(run=r,stratum=name,n=int(mask.sum()),
                exact_response_points=int((mask & exact_point).sum()),
                exact_input52_points=int((mask & equal_input52).sum()),
                within_prefault_envelope_points=int((mask & within_numeric).sum()),
                within_prefault_envelope52_points=int((mask & numeric52).sum()),
                longest_exact_input52_minutes=5*longest(mask & equal_input52),
                longest_numeric52_minutes=5*longest(mask & numeric52))
            for c, short in enumerate(['supply_K','mixed_K','mass_kg_s','pressure_Pa']):
                row[short+'_mean_abs_delta']=average(absolute[mask,c])
                row[short+'_median_abs_delta']=float(np.median(absolute[mask,c])) if mask.any() else None
                row[short+'_mean_abs_delta_over_FIT_sd']=average(absolute[mask,c]/scale[c])
            observation_rows.append(row)
        arrays = dict(index=index,time_s=original_t[index],raw_response_delta=delta_all[index],
            mode_match=current_equal,mode_footprint_match=aligned,input52_identical=equal_input52,
            response_within_prefault_envelope=within_numeric,
            footprint_within_prefault_envelope=numeric52,baseline_switch=switching[index])
        for method, folder in banks.items():
            hi, hs, _, _ = load_bank(folder,data,FILES[0],'QUERY')
            fi, fs, _, _ = load_bank(folder,data,r,'QUERY')
            assert np.array_equal(hi,index) and np.array_equal(fi,index)
            threshold = float(thresholds.loc[method,'threshold'])
            ha, fa = hs>threshold, fs>threshold
            assert int(ha.sum()) == thresholds.loc[method,'normal_false_points']
            arrays[method+'_healthy_score']=hs; arrays[method+'_fault_score']=fs
            arrays[method+'_strict_quad']=quadrants(ha,fa)
            arrays[method+'_c3_quad']=quadrants(confirm(ha),confirm(fa))
            identical_error = float(np.max(np.abs(fs[equal_input52]-hs[equal_input52]))) if equal_input52.any() else None
            integrity[r+'__'+method] = dict(identical_input_points=int(equal_input52.sum()),
                max_score_difference_on_identical_inputs=identical_error)
            for alarm_kind, h, f_alarm in [('strict',ha,fa),('c3',confirm(ha),confirm(fa))]:
                q = quadrants(h,f_alarm)
                for name, mask in masks.items():
                    n=int(mask.sum()); fault_count=int((f_alarm & mask).sum())
                    row=dict(method=method,run=r,stratum=name,alarm_kind=alarm_kind,n=n,
                        threshold=threshold,fault_alarm_points=fault_count,
                        healthy_alarm_points=int((h & mask).sum()),
                        fault_score_gt_healthy_rate=average((fs[mask]>hs[mask]).astype(float)),
                        score_tie_rate=average((fs[mask]==hs[mask]).astype(float)),
                        median_paired_score_delta=float(np.median(fs[mask]-hs[mask])) if n else None,
                        longest_no_fault_alarm_minutes=5*longest(mask & ~f_alarm),
                        longest_no_fault_exclusive_alarm_minutes=5*longest(mask & ~(f_alarm & ~h)))
                    for code, label in enumerate(['neither','healthy_only','fault_only','both']):
                        count=int(((q==code)&mask).sum());row[label+'_points']=count
                        row[label+'_rate']=count/n if n else None
                    row['fraction_fault_alarms_shared_with_healthy']=row['both_points']/fault_count if fault_count else None
                    assert sum(row[k+'_points'] for k in ['neither','healthy_only','fault_only','both'])==n
                    alarm_rows.append(row)
            if method.startswith('ridge_'):
                with np.load(folder/f'{FILES[0]}__QUERY.npz') as z:
                    hp,ht=z['prediction'].copy(),z['target'].copy()
                with np.load(folder/f'{r}__QUERY.npz') as z:
                    fp,ft=z['prediction'].copy(),z['target'].copy()
                dy,dp=ft-ht,fp-hp
                dr=dy-dp
                for name,mask in masks.items():
                    ss=float(np.sum(dy[mask].astype(float)**2))
                    ridge_rows.append(dict(method=method,run=r,stratum=name,n=int(mask.sum()),
                        target_pair_RMS=float(np.sqrt(np.mean(dy[mask]**2))) if mask.any() else None,
                        prediction_pair_RMS=float(np.sqrt(np.mean(dp[mask]**2))) if mask.any() else None,
                        residual_pair_RMS=float(np.sqrt(np.mean(dr[mask]**2))) if mask.any() else None,
                        prediction_projection_on_target=float(np.sum(dp[mask]*dy[mask])/ss) if ss>0 else None,
                        residual_pair_RMS_over_target_RMS=float(np.sqrt(np.sum(dr[mask]**2)/ss)) if ss>0 else None))
        np.savez_compressed(output/(r+'__paired.npz'),**arrays)
    pd.DataFrame(alarm_rows).to_csv(output/'paired_alarms.csv',index=False)
    pd.DataFrame(observation_rows).to_csv(output/'paired_observations.csv',index=False)
    pd.DataFrame(mismatch_rows).to_csv(output/'mode_mismatch_points.csv',index=False)
    pd.DataFrame(interval_rows).to_csv(output/'continuous_intervals.csv',index=False)
    pd.DataFrame(ridge_rows).to_csv(output/'ridge_prediction_response_decomposition.csv',index=False)
    assert all(sha(p)==h for p,h in before.items())
    receipt=dict(status='COMPLETED',CPU_wall_seconds=time.monotonic()-started,GPU_hours=0,
        training=0,model_forward=0,threshold_fitting=0,original_files_unchanged=True,
        source_sha256=sha(__file__),input_sha256=before,prefault_envelope=eps.tolist(),
        envelope_units=['K','K','kg/s','Pa'],identical_input_score_checks=integrity,
        full_method='selected step0 fixed Student prior, not a learned conditional model')
    (output/'RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ['status','CPU_wall_seconds','GPU_hours','prefault_envelope']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.root,args.source,args.output)
