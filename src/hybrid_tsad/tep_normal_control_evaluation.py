"""Pure post-hoc normal-control endpoints; no I/O, models, score loading or fit."""
import math
import numbers
from .engine.events import binary_events

RUNS=tuple(range(51,71))
SEEDS=(2025,2026,2027)
METHODS=('LEFT','SELF','SYNC','PEAK','MULTI','ZMAX')
READOUTS=('last','mean32')
RISK_METRICS=('new_segment_false_hit_40h','new_segment_false_hit_60min',
              'any_confirmed_state_40h','carry_in_confirmed_state')

def evaluate_normal(scores,end_index_0based,threshold):
    import numpy as np
    scores=np.asarray(scores);ends=np.asarray(end_index_0based)
    if scores.shape!=(861,) or scores.dtype.kind not in 'fi' or not np.isfinite(scores).all():
        raise ValueError('Exactly861 finite continuous scores required')
    if ends.dtype.kind not in 'iu' or not np.array_equal(ends,np.arange(99,960)):
        raise ValueError('Complete original endpoints99..959; no gaps/reordering')
    if isinstance(threshold,bool) or not isinstance(threshold,numbers.Real) or not math.isfinite(threshold):
        raise ValueError('Finite fixed normal-CAL threshold required')
    over=scores>threshold
    segments=[]
    for lo,hi in binary_events(over):
        start=100+lo;end=99+hi;confirm=start+2 if hi-lo>=3 else None
        segments.append(dict(start_sample=start,end_sample_inclusive=end,confirm_sample=confirm,
            raw_exceedance_points=hi-lo,left_truncated_at_first_scored_sample=start==100,
            spans_entry=start<161<=end,
            preconfirmed_carry_in=confirm is not None and confirm<161<=end,
            boundary_confirmed_after_entry=start<161 and confirm is not None and confirm>=161))
    new=[s for s in segments if s['start_sample']>=161 and s['confirm_sample'] is not None]
    confirmations=[s for s in segments if s['confirm_sample'] is not None and 161<=s['confirm_sample']<=960]
    carry=[s for s in segments if s['preconfirmed_carry_in']]
    present=[s for s in segments if s['confirm_sample'] is not None and s['end_sample_inclusive']>=max(161,s['confirm_sample'])]
    prefix=[s for s in segments if s['confirm_sample'] is not None and s['confirm_sample']<=160]
    first=min((s['confirm_sample'] for s in new),default=None)
    first_confirmation=min((s['confirm_sample'] for s in confirmations),default=None)
    metrics=dict(new_segment_false_hit_40h=int(bool(new)),new_segment_false_hit_60min=int(first is not None and first<=181),
        new_confirmed_segment_count_40h=len(new),confirmation_count_in_40h_window=len(confirmations),
        carry_in_confirmed_state=int(bool(carry)),any_confirmed_state_40h=int(bool(present)),
        boundary_confirmed_after_entry_count=sum(s['boundary_confirmed_after_entry'] for s in segments),
        new_segment_count_per_hour_40h=len(new)/40,
        confirmation_count_per_hour_40h=len(confirmations)/40,
        new_segment_count_per_hour_39_95h=len(new)/39.95,
        confirmation_count_per_hour_39_95h=len(confirmations)/39.95,
        point_false_alarm_fraction_40h=float(over[61:].mean()),
        prefix_confirmed_segment_count_61points=len(prefix),
        prefix_confirmed_segments_per_hour_183min=len(prefix)/(183/60),
        prefix_point_false_alarm_fraction_61points=float(over[:61].mean()),
        AP=None,ROC_AUC=None)
    return dict(metrics=metrics,segments=segments,first_eligible_new_confirmation_sample=first,
        first_in_window_confirmation_sample=first_confirmation,
        first_new_confirmation_delay_minutes=None if first is None else (first-161)*3,
        first_new_confirmation_administratively_right_censored=first is None,
        censor_sample=960,censor_delay_minutes=2397,
        followup_sample_exposure_hours=40,followup_timestamp_span_hours=39.95,
        prefix_scored_points=61,positive_points=0,
        timing_note='No reset at161; carry-in is an existing observed state, not a new detection or zero-delay onset.',
        metric_note='AP/ROC undefined for all-normal control; no positive target or new thresholds.')

def wilson20(successes):
    if type(successes) is not int or successes not in range(21):
        raise ValueError('Fixed20 independent normal runs; integer successes0..20')
    n=20;z=1.959963984540054;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(successes=successes,n=20,proportion=p,CI95=[max(0.,center-half),min(1.,center+half)],
        method='Wilson score interval; independent-run Bernoulli assumption',
        zero_success_note='0/20 is not proof of zero risk')

def summarize_normal_rows(rows):
    """320 results,16 streams; never treat LEFT seeds as60 independent runs."""
    expected={(m,d,s,r) for m in METHODS for d in READOUTS for s in (SEEDS if m=='LEFT' else (None,)) for r in RUNS}
    indexed={}
    for row in rows:
        key=(row['method'],row['readout'],row['seed'],row['run'])
        if key not in expected or key in indexed or type(row['run']) is not int or (row['seed'] is not None and type(row['seed']) is not int):
            raise ValueError('Unknown/duplicate/noninteger stream identity')
        for name in RISK_METRICS:
            v=row['result']['metrics'][name]
            if type(v) is not int or v not in (0,1):raise ValueError('Per-run binary risk required')
        indexed[key]=row
    if set(indexed)!=expected:raise ValueError('All320run/stream results required, no missing cases')
    streams=[]
    for m in METHODS:
        for d in READOUTS:
            for seed in (SEEDS if m=='LEFT' else (None,)):
                selected=[indexed[m,d,seed,r] for r in RUNS]
                names=selected[0]['result']['metrics']
                streams.append(dict(method=m,readout=d,seed=seed,n_independent_normal_runs=20,
                    risks={name:wilson20(sum(x['result']['metrics'][name] for x in selected)) for name in RISK_METRICS},
                    descriptive_mean_metrics={name:None if any(x['result']['metrics'][name] is None for x in selected)
                        else sum(x['result']['metrics'][name] for x in selected)/20 for name in names},
                    right_censored_runs=[x['run'] for x in selected if x['result']['first_new_confirmation_administratively_right_censored']]))
    seed_averaged=[]
    for d in READOUTS:
        for run in RUNS:
            names=indexed['LEFT',d,2025,run]['result']['metrics']
            seed_averaged.append(dict(method='LEFT',readout=d,run=run,n_training_seeds=3,
                metrics={name:None if any(indexed['LEFT',d,s,run]['result']['metrics'][name] is None for s in SEEDS)
                    else sum(indexed['LEFT',d,s,run]['result']['metrics'][name] for s in SEEDS)/3 for name in names},
                interpretation='Descriptive seed average per independent process run; not Bernoulli60',Wilson_CI=None))
    return dict(rows=rows,streams=streams,LEFT_per_run_seed_averages=seed_averaged,
        independent_normal_runs=20,result_rows=320,stream_count=16,bootstrap_performed=False,
        zero_risk_not_claimed=True)
