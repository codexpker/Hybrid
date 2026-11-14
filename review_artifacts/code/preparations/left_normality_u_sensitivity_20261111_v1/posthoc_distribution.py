"""Fixed posthoc score distribution diagnosis; no model or threshold fitting."""
import csv, hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent; R=P.parents[1]; O=R/'results'/P.name
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p) as z: return {k:z[k].copy() for k in z.files}
def table(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
    start=time.monotonic(); E=O/'posthoc_distribution';E.mkdir(exist_ok=False)
    seal=O/'U_fixed_scores/ALL_RAW_MASKS_THRESHOLDS_SEALED.json'
    assert (O/'U_fixed_evaluation/RESULT.json').exists()
    spec=importlib.util.spec_from_file_location('alarm',R/'hybrid_ad/src/hybrid_tsad/alarm_complementarity.py'); a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    psm_tau=read(R/'results/hybrid_left_native_history_lag_20261110_v1/N/cal_common/RECEIPT.json')['threshold']
    smd_tau=read(R/'results/cross_domain_left_arx_20261111_v1/fixed_evaluation/CALIBRATION.json')['thresholds']['LEFT_native']
    cfg=read(R/'preparations/hybrid_left_joint_cycle_20261109_v2/EVALUATION.ACTUAL.json')
    truth_path=Path(str(cfg['truth']['path']).replace('__RESEARCH_ROOT__',str(R)))
    labels={'PSM':load(truth_path)['full_labels'],'SMD1-5':load(R/'data/prospective_smd_machine15_roles_20261106_v1/audit_only/ORIGINAL_TEST_LABELS.npz')['labels']}
    assets=[]; rows=[]; sorted_bank={}; plot_series={'PSM':[],'SMD1-5':[],'SMD_U_CAL':[]}
    quantiles=[.01,.1,.25,.5,.75,.9,.95,.99,.995,.999]
    def add(domain,model,role,path,tau,query=False,axis=None):
        d=load(path); idx=d['index']; score=d['score']; assert np.all(np.diff(idx)==1)
        conf=a.confirmed(idx,score>tau)
        groups=[('all',np.ones(len(idx),bool))]
        if query: groups += [('normal',labels[domain][idx]==0),('anomaly',labels[domain][idx]==1)]
        for group,mask in groups:
            v=score[mask]; name=f'{domain}__{model}__{role}__{group}'; sv=np.sort(v)
            sorted_bank[name]=sv
            row=dict(domain=domain,model=model,role=role,group=group,count=len(v),mean=float(v.mean()),std=float(v.std()),minimum=float(v.min()),maximum=float(v.max()),reference_tau=tau,strict_points=int((v>tau).sum()),strict_fraction=float((v>tau).mean()),C3_points=int(conf[mask].sum()),C3_fraction=float(conf[mask].mean()),C3_computed_before_label_group_filter=True)
            row.update({f'q{q}':float(np.quantile(v,q)) for q in quantiles});rows.append(row)
            if axis and (not query or group!='all'):plot_series[axis].append((f'{model} {role} {group}',sv))
        assets.append(dict(path=str(path),sha256=sha(path)))
    for role in ['FIT','SELECT_CAL']:add('PSM','base',role,O/'frozen_normal/PSM'/f'{role}.npz',psm_tau,axis='PSM')
    add('PSM','base','QUERY',R/'results/left_causal_endpoint_20261109_v1/QUERY/scores.npz',psm_tau,True,'PSM')
    for role in ['FIT','SELECT']:add('SMD1-5','base',role,O/'frozen_normal/SMD'/f'{role}.npz',smd_tau,axis='SMD1-5')
    old=R/'results/cross_domain_left_arx_20261111_v1/left_score'
    add('SMD1-5','base','CAL',old/'CAL.npz',smd_tau,axis='SMD1-5')
    add('SMD1-5','base','QUERY',old/'QUERY.npz',smd_tau,True,'SMD1-5')
    add('SMD1-5','base','CAL',old/'CAL.npz',smd_tau,axis='SMD_U_CAL')
    for c in 'ABC':
        add('SMD1-5','U_'+c,'CAL',O/'U'/c/'score/CAL.npz',smd_tau,axis='SMD_U_CAL')
        add('SMD1-5','U_'+c,'QUERY',O/'U'/c/'score/QUERY.npz',smd_tau,True)
    table(E/'score_distribution_groups.csv',rows);np.savez_compressed(E/'sorted_scores_for_CDF.npz',**sorted_bank)
    fig,axes=plt.subplots(1,3,figsize=(16,4.7))
    for ax,(domain,series) in zip(axes,plot_series.items()):
        for name,sv in series:ax.plot(sv,np.arange(1,len(sv)+1)/len(sv),label=name,lw=1.25)
        ax.set_xscale('symlog',linthresh=.1);ax.set_ylim(0,1.005);ax.set_xlabel('Native LEFT score (symlog)');ax.set_ylabel('Empirical CDF');ax.set_title(domain);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Frozen score diagnosis; QUERY groups are posthoc only; PSM SELECT = CAL')
    fig.tight_layout();fig.savefig(E/'native_score_CDF.png',dpi=180);fig.savefig(E/'native_score_CDF.svg');plt.close(fig)
    receipt=dict(status='POSTHOC_FIXED_DIAGNOSIS_COMPLETE',raw_seal_sha256=sha(seal),assets=assets,label_PSM_path=str(truth_path),label_PSM_sha256=sha(truth_path),source_sha256=sha(Path(__file__)),fixed_quantiles=quantiles,QUERY_groups_not_used_for_training_or_selection=True,PSM_SELECT_CAL_reused=True,background_health_not_certified=True,normal_FIT_in_sample=True,seconds=time.monotonic()-start)
    (E/'RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='COMPLETED',rows=len(rows),seconds=receipt['seconds'])))
if __name__=='__main__':main()
