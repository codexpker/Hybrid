"""Exact new8-case cache analysis plus immutable old20 reuse and paired n=3 SD.

Default is preview only. Root supplies the final METRICS path/SHA before execute.
No training, model import/forward, CAL rewriting or old score recomputation.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time

OLD_METRICS_SHA = 'daa93070e72348ef8d304bcb184f85222b7c14228097fa1fef2db6d1d8346133'
OLD_CSV_SHA = '37949f91e9e2969b1ee78838dcbaeb59fe1512f067a25b6a70d21037f3b593b6'
OLD_MANIFEST_SHA = 'e800887918bf648b4faf1c23d7e43a06b3268dbbbe2a7d37bbe75aace028e9ad'
CASES = [(d, m, s) for d in ('SWAT', 'PSM') for m in ('R', 'R_target_mean') for s in (2026, 2027)]
SEEDS = (2025, 2026, 2027)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle: return list(csv.DictReader(handle))


def identity(row):
    seed = row.get('seed')
    return row['domain'], row['method'], None if seed in ('', None, 'None') else int(seed), row['stream']


def normalized_value(value):
    """Only CSV representation conversion; never round or impute a number."""
    return '' if value is None else str(value)


def merge_rows(old, new):
    expected_old = {(d, m, None if m == 'noU' else 2025, stream)
        for d in ('SWAT', 'PSM') for m in ('noU', 'TT', 'SS', 'R', 'R_target_mean')
        for stream in ('strict', 'confirmed3')}
    expected_new = {case+(stream,) for case in CASES for stream in ('strict', 'confirmed3')}
    if len(old) != 20 or {identity(r) for r in old} != expected_old:
        raise ValueError('Exactly the original20 identities required')
    if len(new) != 16 or {identity(r) for r in new} != expected_new:
        raise ValueError('Exactly8 new states/16 stream rows required')
    merged = [dict(r) for r in old]+[dict(r) for r in new]
    if len({identity(r) for r in merged}) != 36: raise ValueError('Duplicated supplement identity')
    # Require exact all-field representation preservation for the old20 rows.
    if any({k:normalized_value(v) for k,v in row.items()} != old[i] for i,row in enumerate(merged[:20])):
        raise ValueError('Old20 fields changed during reuse')
    return merged


def scientific_fields(rows):
    """Exclude ROC-point ordinal/identity/exposure fields from scientific pairs."""
    return sorted({k for row in rows for k in row if
        (k.startswith('pAUC_') or k.startswith('cap') or k.startswith('CAL_operating_QUERY_'))
        and not k.endswith('_curve_point')})


def triplet_stats(values):
    if len(values) != 3: raise ValueError('Exactly three paired seeds required')
    numbers = [None if v in ('',None,'None') else float(v) for v in values]
    if any(v is None for v in numbers):
        return dict(n=3, valid_n=sum(v is not None for v in numbers), mean=None, SD=None,
                    ddof=1, NA_reason='NOT_DEFINED_FOR_ALL_THREE_SEEDS', seed_values=numbers)
    if not all(math.isfinite(v) for v in numbers): raise ValueError('Finite numeric fields only')
    return dict(n=3, valid_n=3, mean=statistics.mean(numbers), SD=statistics.stdev(numbers),
                ddof=1, NA_reason=None, seed_values=numbers)


def summaries(rows):
    lookup={identity(r):r for r in rows}
    method_rows=[]; pairs=[]
    for domain in ('SWAT','PSM'):
        for stream in ('strict','confirmed3'):
            fields=scientific_fields([lookup[domain,m,s,stream] for m in ('R','R_target_mean') for s in SEEDS])
            for field in fields:
                values={m:[lookup[domain,m,s,stream].get(field) for s in SEEDS] for m in ('R','R_target_mean')}
                # Strict does not acquire fictional C3 event metrics from wide CSV blanks.
                if all(v in ('',None,'None') for vals in values.values() for v in vals):continue
                for method in ('R','R_target_mean'):
                    method_rows.append(dict(domain=domain,stream=stream,method=method,metric=field,
                        seeds=list(SEEDS),**triplet_stats(values[method])))
                differences=[float(a)-float(b) if a not in ('',None,'None') and b not in ('',None,'None') else None
                    for a,b in zip(values['R_target_mean'],values['R'])]
                pairs.append(dict(domain=domain,stream=stream,contrast='R_target_mean_minus_R',metric=field,
                    seeds=list(SEEDS),**triplet_stats(differences)))
    if any('_curve_point' in r['metric'] for r in pairs):raise AssertionError('Ordinal entered science')
    return method_rows,pairs


def write_csv(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def verify_old_operating_identity(old, metrics, old_metrics):
    """Bind old20 reused values to unchanged QUERY bank and original CAL operation."""
    old_lookup={identity(r):r for r in old_metrics['rows']}
    current={identity(r):r for r in metrics['rows']}
    for row in old:
        key=identity(row);a=old_lookup[key];b=current[key]
        if a['bank_QUERY_sha256'] != row['QUERY_bank_sha256']:
            raise ValueError('Old supplement QUERY identity differs')
        for field in ('bank_QUERY_sha256','bank_CAL_sha256','threshold','TP','FP','normal_FPR','Recall'):
            if a[field] != b[field]:raise ValueError('Old CAL/QUERY operating identity changed: '+field)
        for metric,field in [('normal_FPR','CAL_operating_QUERY_FPR'),('Recall','CAL_operating_QUERY_Recall'),
                             ('TP','CAL_operating_QUERY_TP'),('FP','CAL_operating_QUERY_FP')]:
            if float(row[field]) != float(b[metric]):raise ValueError('Old CAL operating value changed')


def table_markdown(rows):
    pct=lambda v:'NA' if v in ('',None) else f'{float(v)*100:.3f}%'
    def area(r,k):return 'NA' if r.get('pAUC_'+k) in ('',None) else f"{float(r['pAUC_'+k]):.7f} / {float(r['pAUC_div_bound_'+k]):.4f}"
    def cap(r,k):
        result=pct(r[k+'_FPR'])+'/'+pct(r[k+'_TPR'])
        if r['stream']=='confirmed3':
            result+='; C '+pct(r[k+'_event_point_coverage'])+'; L '+pct(r[k+'_long_steady_macro_coverage'])+'/'+pct(r[k+'_long_steady_point_coverage'])
            result+='; miss '+str(r[k+'_max_event_miss_points'])+'; M/A '+str(r[k+'_M'])+'/'+str(r[k+'_auxiliary_any_hit_events'])
        return result
    lines=['| 域 | 方法 | seed | stream | 原CAL位QUERY FPR/Recall | pAUC .01 / area÷.01 | pAUC .05 / area÷.05 | cap1%可达点 | cap3%可达点 | cap5%可达点 |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        seed=str(identity(r)[2]);seed+=' ref' if r['method'] in ('TT','SS') else ''
        fields=[r['domain'],r['method'],seed,'C3' if r['stream']=='confirmed3' else 'strict',
            pct(r['CAL_operating_QUERY_FPR'])+'/'+pct(r['CAL_operating_QUERY_Recall']),area(r,'01'),area(r,'05'),
            cap(r,'cap01'),cap(r,'cap03'),cap(r,'cap05')]
        lines.append('| '+' | '.join(fields)+' |')
    return '\n'.join(lines)


def execute(args):
    start=time.perf_counter();w=Path(args.workspace).resolve();old=Path(args.old).resolve();out=Path(args.out).resolve()
    if out.exists():raise FileExistsError(out)
    if sha(old/'MANIFEST.json') != OLD_MANIFEST_SHA or sha(old/'LOW_FPR_SUPPLEMENT.csv') != OLD_CSV_SHA:
        raise ValueError('Original20 scientific artifacts changed')
    if sha(args.metrics) != args.metrics_sha:raise ValueError('Root final METRICS identity differs')
    old_metrics_path=w/'results/timeradar_target_mean_metrics_20261120_v1/METRICS.json'
    if sha(old_metrics_path) != OLD_METRICS_SHA:raise ValueError('Original2025 metrics identity differs')
    metrics=json.loads(Path(args.metrics).read_text(encoding='utf-8'))
    old_metrics=json.loads(old_metrics_path.read_text(encoding='utf-8'));old_rows=read_csv(old/'LOW_FPR_SUPPLEMENT.csv')
    verify_old_operating_identity(old_rows,metrics,old_metrics)
    sys.path.insert(0,str(w/'hybrid_ad/src'))
    from hybrid_tsad.native_v_conditioning_curves import low_fpr_cached
    # Exactly new8 cases; never call the old20 calculation again.
    new_rows,new_receipt=low_fpr_cached(w,args.root,args.metrics,None,out,
        expected_metrics_sha=args.metrics_sha,methods=('R','R_target_mean'),cases=CASES)
    merged=merge_rows(old_rows,new_rows);method_rows,pairs=summaries(merged)
    write_csv(out/'LOW_FPR_SUPPLEMENT_36.csv',merged)
    write_csv(out/'LOW_FPR_METHOD_MEAN_SD.csv',method_rows);write_csv(out/'LOW_FPR_PAIRED_MEAN_SD.csv',pairs)
    (out/'LOW_FPR_SUMMARY.json').write_text(json.dumps(dict(rows=merged,method_mean_SD=method_rows,
        paired_mean_SD=pairs,seeds=list(SEEDS),ddof=1,old20_recomputed=False),indent=2,allow_nan=False),encoding='utf-8')
    # Existing20 values must survive a realCSV roundtrip exactly.
    roundtrip=read_csv(out/'LOW_FPR_SUPPLEMENT_36.csv')
    if roundtrip[:20] != old_rows:raise ValueError('Old20 CSV roundtrip changed original values')
    doc=Path(args.docs);doc.mkdir(parents=True,exist_ok=True)
    old_md=(doc/'LOW_FPR_SUPPLEMENT.md').read_text(encoding='utf-8')
    definition=old_md.split('计算状态：')[0].replace('呈现相同20行', '呈现相同36行')
    text=definition+'计算状态：已完成36行；旧20行逐字段原件复用，仅8个新状态/16行重新计算。noU单参考，TT/SS仅2025参考；R/MeanT各3seed。\n\n'
    text+='cap列为实际FPR/点TPR；C为完整事件点覆盖，L为long steady macro/pooled，miss为最长完整事件未报点数，M/A为1to1/any-hit。CAL原操作点不变，所有cap仅QUERY回顾描述。\n\n'+table_markdown(merged)+'\n\n'
    text+='三seed方法均值±SD与同seed MeanT−R配对差分别见 `LOW_FPR_METHOD_MEAN_SD.csv` 和 `LOW_FPR_PAIRED_MEAN_SD.csv`；n=3，SD使用ddof=1，完整seed值保留。统计包括原CAL操作点、pAUC/归一面积、各cap实际FPR/TPR/TP/FP及真实C3事件覆盖/long macro/pooled/最长miss/匹配/any-hit；不把曲线点序号作为科学指标。NA不补零，不为noU/TT/SS伪造三seed统计。\n\n'
    text+=f'旧20 CSV SHA `{OLD_CSV_SHA}`；旧科学MANIFEST SHA `{OLD_MANIFEST_SHA}`。新16行分析CPU wall={new_receipt["cpu_wall_seconds"]:.9f}s；GPU/神经forward=0。所有QUERY预算点均不输出为部署阈值，不反流训练或原CAL。\n'
    (doc/'LOW_FPR_SUPPLEMENT.md').write_text(text,encoding='utf-8')
    for source,name in [('LOW_FPR_SUPPLEMENT_36.csv','LOW_FPR_SUPPLEMENT.csv'),
                        ('LOW_FPR_METHOD_MEAN_SD.csv','LOW_FPR_METHOD_MEAN_SD.csv'),
                        ('LOW_FPR_PAIRED_MEAN_SD.csv','LOW_FPR_PAIRED_MEAN_SD.csv')]:
        (doc/name).write_bytes((out/source).read_bytes())
    receipt=dict(status='COMPLETED_36_ROWS',old20_csv_sha256=OLD_CSV_SHA,old20_manifest_sha256=OLD_MANIFEST_SHA,
        old20_recomputed=False,old20_exact_CSV_roundtrip=True,new_cases=CASES,new_rows=16,rows=36,
        final_METRICS_sha256=args.metrics_sha,export_source_sha256=sha(__file__),
        new_analysis_manifest_sha256=sha(out/'MANIFEST.json'),CPU_wall_seconds=time.perf_counter()-start,
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},neural_forwards=0,GPU_hours=0)
    (out/'MERGE_MANIFEST.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


def self_test():
    assert triplet_stats([1,2,3])['mean']==2 and triplet_stats([1,2,3])['SD']==1
    assert triplet_stats([1,None,3])['mean'] is None
    rows=[]
    for d in ('SWAT','PSM'):
        for m in ('R','R_target_mean'):
            for s in SEEDS:
                for stream in ('strict','confirmed3'):
                    rows.append(dict(domain=d,method=m,seed=s,stream=stream,pAUC_div_bound_01=float(s-2024)+(m=='R_target_mean'),cap01_curve_point=s))
    _,pairs=summaries(rows)
    assert len(pairs)==4 and all(r['mean']==1 and r['SD']==0 and r['n']==3 for r in pairs)
    assert all('curve_point' not in r['metric'] for r in pairs)
    print(json.dumps(dict(status='SYNTHETIC_SELF_TEST_PASS',n=3,ddof=1,ordinal_excluded=True,real_assets_read=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--self-test',action='store_true')
    for key in ('workspace','root','metrics','metrics-sha','old','out','docs'):parser.add_argument('--'+key)
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.execute:
        if any(getattr(args,k.replace('-','_')) is None for k in ('workspace','root','metrics','metrics-sha','old','out','docs')):
            parser.error('Root final exact METRICS/path/SHA and all output paths required')
        execute(args)
    else:print(json.dumps(dict(status='PREPARATION_ONLY_WAITING_ROOT_METRICS_IDENTITY',cases=CASES,
        new_rows=16,reused_rows=20,total_rows=36,real_assets_read=False,neural_forwards=0,GPU_hours=0)))
