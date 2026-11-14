"""Frozen paired initialization experiment, original h1/CAL1024/common PSM axis.

All three NEW raw banks seal before truth. Old compact inputs and identical-axis
accepted operating results are reused; no model forward or score fitting.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hybrid_tsad.hybrid_v1 import evaluation
from hybrid_tsad.hybrid_v1.pretrained_psm_comparison import (
    SEEDS, bound, read, sha, write, validate_new_banks, archive_scalars, paired_tables)
from hybrid_tsad.hybrid_v1.public_native_evaluation import event_diagnostics
from hybrid_tsad.hybrid_v1.controlled_wave_evaluation import reject_dependency_na
from compare_psm_common_axis import load_method
from evaluate_p32_readouts import operating_columns  # Output flattening only, no P32 scoring.


def table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def crop(data, common):
    index = data['index']
    where = np.searchsorted(index, common)
    if np.any(where >= len(index)) or not np.array_equal(index[where], common):
        raise ValueError('Fixed common time axis missing; no score-selected intersection')
    if not data['valid'][where].all() or not np.isfinite(data['score'][where]).all():
        raise ValueError('UNKNOWN coverage must be resolved/reported, not silently dropped')
    return where


def run(config):
    began = time.monotonic()
    if config['status'] != 'FROZEN_READY' or config['common_axis'] != [512,87744]:
        raise PermissionError('Pending template or incorrect fixed original h1 axis')
    for path, expected in config['source_sha256'].items():
        if sha(path) != expected:
            raise PermissionError('Frozen evaluation dependency changed: '+path)
    science = read(config['science_freeze'])
    if science['seeds'] != list(SEEDS) or science['common_external_axis'] != [512,87744]:
        raise ValueError('Authoritative three-seed science mismatch')
    if not isinstance(config['resource_receipts'],list) or not config['resource_receipts']:
        raise PermissionError('Bind actual job OS/resource receipts before production')
    for asset in config['resource_receipts']:
        bound(asset)
    gates = validate_new_banks(config['new_methods'])
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=False)
    archived = []
    for d in config['new_methods']:
        asset = archive_scalars(d, output / ('OFFICIAL528_seed%d_raw_scores.npz' % d['seed']))
        archived.append(dict(seed=d['seed'], raw=asset))
    write(output/'ALL_THREE_RAW_SEALED_BEFORE_LABELS.json', dict(status='ALL_THREE_COMPLETE_RAW_ARCHIVED',
          query_label_reads=0, gates=gates, archives=archived, source_config=config))
    # This is the sole first numerical truth opener; not reached on partial runs.
    with np.load(bound(config['truth']), allow_pickle=False) as truth:
        labels = truth['full_labels'].copy()
    if labels.shape != (87841,):
        raise ValueError('Retain original 87841 native labels, including UNKNOWN tail')
    common = np.arange(512,87744); y = labels[common]
    descriptors = list(config['old_methods'])
    arrays = {}
    for d, archive in zip(config['new_methods'], archived):
        name = 'Hybrid_OFFICIAL528_seed%d' % d['seed']
        with np.load(archive['raw']['path'], allow_pickle=False) as data:
            arrays[name] = {k: data[k].copy() for k in ('index','score','valid','available_index')}
        descriptors.append(dict(id=name, seed=d['seed'], initialization='OFFICIAL528', online=True,
            threshold=archive['raw']['threshold'], protocol='original_h1_R3_sourceCAL1024_sourceSELECT8000',
            role_permissions='official additional pretraining + matched subsequent source8000; no targetFIT/CAL'))
    rows, phases = [], []
    for d in descriptors:
        name = d['id']
        data = arrays.get(name)
        if data is None:
            data = load_method(d, labels)
        where = crop(data, common); score = data['score'][where]
        available = data.get('available_index', data.get('available'))[where]
        row = dict(method=name, seed=d.get('seed'), initialization=d.get('initialization'),
            points=len(common), positive_points=int(y.sum()), AP=float(average_precision_score(y,score)),
            ROC_AUC=float(roc_auc_score(y,score)), online=d['online'], threshold=d.get('threshold'),
            protocol=d['protocol'], role_permissions=d['role_permissions'])
        folder = output/name; folder.mkdir()
        with (folder/'common_axis_inputs.npz').open('xb') as handle:
            np.savez_compressed(handle,index=common,score=score,valid=np.ones(len(common),bool),
                                available_index=available,full_labels=labels)
        if d['online']:
            if (available>common).any():
                raise PermissionError('Offline scores cannot become causal by cropping')
            if 'accepted_common_evaluation' in d:
                # Exact old 2025/Mean512 axis and input scores: no repeated evaluation.
                previous = d['accepted_common_evaluation']
                result = read(previous['result'])
                provenance = read(previous['provenance'])
                if provenance['calculation_inputs_sha256'] != previous['inputs']['sha256']:
                    raise ValueError('Prior operating inputs lack accepted provenance')
                with np.load(bound(previous['inputs']), allow_pickle=False) as old:
                    for key, expected in (('index',common),('score',score),('available_index',available),('full_labels',labels)):
                        if not np.array_equal(old[key],expected):
                            raise ValueError('Prior operating result is not the exact same input')
                prior_method = next(v for v in provenance['provenance']['config']['methods'] if v['id']==previous['original_method'])
                if prior_method['threshold'] != d['threshold']:
                    raise ValueError('Prior fixed threshold differs')
                write(folder/'REUSED_ACCEPTED_OPERATING_RESULT.json', previous)
            else:
                result = evaluation.evaluate_arrays(index=common,score=score,valid=np.ones(len(common),bool),
                    available_index=available,invalid_reasons=[None]*len(common),full_labels=labels,
                    threshold=d['threshold'],bank_complete=True,original_length=87841,warmup_points=512,
                    affiliation_identity=bound(config['affiliation_identity']))
                reject_dependency_na(result)
                evaluation.save_evaluation(folder/'fixed_source_tau_online',result,index=common,score=score,
                    valid=np.ones(len(common),bool),available_index=available,invalid_reasons=[None]*len(common),
                    full_labels=labels,provenance=dict(config=config,method=name,no_target_threshold_fit=True))
            p = event_diagnostics(result)
            phases.extend(dict(method=name,**v) for v in p)
            row.update(operating_columns(result,p))
            for prefix, stream in (('strict','strict_point'),('confirmed3','confirmed3_current')):
                affiliation=result['streams'][stream]['affiliation']['global_result']
                for key in ('Aff_P','Aff_R','Aff_F1'):
                    row[prefix+'_'+key]=affiliation.get(key)
            write(folder/'OPERATING_RESULT_REFERENCE.json',dict(
                reused='accepted_common_evaluation' in d, ranking_recomputed_same_sklearn=True,
                event_coverage=result['event_coverage'],streams=result['streams']))
        else:
            row['online_NA_reason'] = 'NATIVE_OFFLINE_WINDOW_AND_BATCH; target statistics/training differ; availability is only archived input lower bound'
        rows.append(row)
    paired, summary = paired_tables(rows)
    table(output/'all_methods_common_axis.csv',rows)
    table(output/'paired_seed_differences.csv',paired)
    table(output/'three_seed_mean_SD_extrema.csv',summary)
    table(output/'event_phases.csv',phases)
    write(output/'RESULT.json',dict(status='COMPLETED_PAIRED_INITIALIZATION_PSM',seconds=time.monotonic()-began,
        common_axis=[512,87744],rows=rows,paired=paired,summary=summary,source_config=config,
        GPUh=0,new_forward=False,new_training=False,labels_after_all_three_sealed=True,
        inference_limit='Three optimization seeds, one historically exposed development domain; only subsequent 8000 updates matched, not total external pretraining; no novel-method claim from initialization alone'))
    print(json.dumps(dict(status='COMPLETED',seconds=time.monotonic()-began,rows=len(rows))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    parser.add_argument('--trusted-config-sha256',required=True); args=parser.parse_args()
    if sha(args.config)!=args.trusted_config_sha256:
        raise PermissionError('Frozen final comparison config required')
    run(json.loads(Path(args.config).read_text(encoding='utf-8')))
