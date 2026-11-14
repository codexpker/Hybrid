"""One source-discrimination candidate, canonical h1 scores, fixed full PSM axis.

No model import, fitting, threshold choice or official528/three-seed admission.
Run only after root binds the completed full raw bank and production evidence.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from .pretrained_psm_comparison import bound, read, sha, write, archive_scalars
from .psm_matched_u_evaluation import complete_steady_all_missed

INITIAL = '9a8f18ba57f64bc0ecbf17cab20d4fc1bc9758b433667f135de6eaa27ace6071'
PLAN = '243c5ccf7026222d2c406811bc143f01322a83523f8b583d3e4959db5b53ec93'
BASELINE = 'HybridV1_Z_source2025'
OLD_SOURCE_CAL = 'da239eac4c5ab1e76718cd651c48f6f89b8d5658b5b9b2a20cb400d88f83d585'
RANGES = [dict(domain='PSM', entity='anonymous_PSM', run='test_original', start=512, end=87841)]


def exact_fields(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError('Required production evidence differs: ' + key)


def admission(config):
    """Metadata/hash closure only, before opening any numerical query truth."""
    if config.get('status') != 'FROZEN_READY' or config.get('seed') != 2025 or config.get('common_axis') != [512, 87744]:
        raise PermissionError('Only the final root-bound single-candidate protocol is executable')
    for path, expected in config['source_sha256'].items():
        if sha(path) != expected:
            raise PermissionError('Changed frozen dependency: ' + path)
    science = read(config['science_freeze'])
    exact_fields(science['baseline'], dict(initial=INITIAL, plan=PLAN, seed=2025, normal_updates=8000))
    exact_fields(science['loss'], {'lambda': .1, 'margin': .1, 'additional_parameters': 0, 'no_search': True})
    d = config['candidate']
    if d['initial']['sha256'] != INITIAL or d['source_plan']['sha256'] != PLAN:
        raise ValueError('Only the matched original initial and source plan')
    bound(d['initial']); bound(d['source_plan']); bound(d['source_CAL_records'])
    train, cal, query = [read(d[key]) for key in ('training_receipt', 'calibration_receipt', 'query_receipt')]
    for receipt, phase, status in ((train,'train','VALID_SOURCE_SELECTED_ENDPOINT'),
            (cal,'cal','VALID_SOURCE_CAL'), (query,'query',d['expected_query_scientific_status'])):
        exact_fields(receipt, dict(status='COMPLETED', phase=phase, seed=2025,
                                  scientific_status=status, labels_read=False))
    exact_fields(train, dict(attempts=8000, updates=8000, initial_sha256=INITIAL))
    producer_config = read(d['training_config'])
    exact_fields(train,dict(config_sha256=d['training_config']['sha256']))
    exact_fields(producer_config['initial'],dict(sha256=INITIAL))
    if producer_config['original_config']['sha256'] != d['original_source_config']['sha256']:
        raise ValueError('Normal source training configuration changed')
    bound(d['original_source_config'])
    if producer_config['units']['sha256'] != train['units_sha256']:
        raise ValueError('FIT-only detached amplitude unit provenance differs')
    bound(d['fit_units'])
    if d['fit_units']['sha256'] != train['units_sha256']:
        raise ValueError('FIT48 units are not the source CAL1024')
    checkpoint = d['checkpoint']['sha256']; bound(d['checkpoint'])
    for receipt in (train, cal, query):
        exact_fields(receipt, dict(checkpoint_sha256=checkpoint))
        exact_fields(receipt, config['expected_shared_receipt_fields'])
    report = read(d['training_report'])
    exact_fields(report, dict(attempts=8000, updates=8000, selected_attempt=train['selected_attempt']))
    if len(report['steps']) != 8000:
        raise ValueError('Incomplete normal source8000 training trace')
    if [v['id'] for v in config['baselines']] != [BASELINE,'HistoryMean512_ZS','TimeRadar_official_native','LEFT_R_targettrained']:
        raise ValueError('The four fixed historical comparators cannot be selected by performance')
    aux = config['fixed_auxiliary_diagnostics']
    exact_fields(aux,dict(enabled=True,components=['r_tf','r_ms','r_pred'],selection_allowed=False,
                         change_primary_scores=False,axis=[512,87744]))
    if aux['original_source_calibration']['sha256'] != OLD_SOURCE_CAL:
        raise ValueError('One prespecified original c00 source CAL, not target Z-T CAL')
    oldcal=read(aux['original_source_calibration'])
    exact_fields(oldcal,dict(valid_windows=1024,source_domain_equal_weight=True,
        domains=['SMD','Sunspot'],weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),
        threshold=config['baselines'][0]['threshold']))
    for asset in config['resource_receipts']:
        bound(asset)
    if not config['resource_receipts'] or not config['required_evidence']:
        raise PermissionError('Bind actual CPU/L0/gradient/mode/RNG evidence and OS receipts')
    for evidence in config['required_evidence']:
        exact_fields(read(evidence['asset']), evidence['expected_fields'])
    manifest, calibration = read(d['bank_manifest']), read(d['source_CAL1024'])
    exact_fields(cal, dict(records=1024, calibration_sha256=d['source_CAL1024']['sha256']))
    exact_fields(query, dict(points=87329, calibration_sha256=d['source_CAL1024']['sha256'],
                            bank_manifest_sha256=d['bank_manifest']['sha256']))
    exact_fields(calibration, dict(valid_windows=1024, domains=['SMD','Sunspot'],
        source_domain_equal_weight=True, weights=dict(r_tf=.25,r_ms=.25,r_pred=.5)))
    exact_fields(manifest, dict(complete=True, committed_records=87329))
    contract = manifest['contract']
    exact_fields(contract, dict(ranges=RANGES, threshold=calibration['threshold']))
    keys = {'input_sha256','config_sha256','checkpoint_sha256','calibration_sha256','source_sha256'}
    if set(d['expected_five_identity']) != keys:
        raise PermissionError('All five actual producer identities must be bound')
    exact_fields(contract['identity'], d['expected_five_identity'])
    exact_fields(contract['identity'], dict(checkpoint_sha256=checkpoint, calibration_sha256=d['source_CAL1024']['sha256']))
    seal = read(config['whole_raw_seal'])
    exact_fields(seal, dict(labels_read=False, bank_manifest_sha256=d['bank_manifest']['sha256'],
                           checkpoint_sha256=checkpoint, calibration_sha256=d['source_CAL1024']['sha256']))
    return dict(status='SINGLE_CANDIDATE_ADMITTED', checkpoint_sha256=checkpoint,
        manifest_sha256=d['bank_manifest']['sha256'], calibration_sha256=d['source_CAL1024']['sha256'])


def import_bound(asset, name):
    path = bound(asset)
    # Accepted scripts import one another from their existing scripts directory.
    if str(path.parent) not in sys.path:
        sys.path.insert(0,str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def crop(data, common):
    index = data['index']; where = np.searchsorted(index, common)
    if np.any(where >= len(index)) or not np.array_equal(index[where],common):
        raise ValueError('The prespecified full common axis must be present')
    if not data['valid'][where].all() or not np.isfinite(data['score'][where]).all():
        raise ValueError('UNKNOWN coverage cannot be silently removed')
    return where


def corrected_operating_columns(formatter, result, phases):
    row = formatter(result, phases)
    for prefix, stream in (('strict','strict_point'),('confirmed3','confirmed3_current')):
        steady = [p for p in phases if p['stream']==stream and p['phase']=='steady_after512' and p['duration']>512]
        count = complete_steady_all_missed(steady)
        # Retain the accepted schema, fixing its UNKNOWN-only aggregation.
        row[prefix+'_steady_all_zero_detected_events'] = count
        row[prefix+'_steady_main_zero_detected_events'] = count
        aff = result['streams'][stream]['affiliation']['global_result']
        for key in ('Aff_P','Aff_R','Aff_F1'):
            row[prefix+'_'+key] = aff.get(key)
    return row


def reuse_operating(descriptor, index, score, available, labels):
    previous = descriptor['accepted_common_evaluation']
    result, provenance = read(previous['result']), read(previous['provenance'])
    if provenance['calculation_inputs_sha256'] != previous['inputs']['sha256']:
        raise ValueError('Old operating input provenance changed')
    with np.load(bound(previous['inputs']), allow_pickle=False) as old:
        for key, value in dict(index=index,score=score,available_index=available,full_labels=labels).items():
            if not np.array_equal(old[key],value):
                raise ValueError('Accepted operating metrics require identical axis/raw/truth')
    prior = next(p for p in provenance['provenance']['config']['methods'] if p['id']==previous['original_method'])
    if prior['threshold'] != descriptor['threshold']:
        raise ValueError('Old source-only threshold changed')
    return result


def table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def fixed_old_calibration_scores(raw_components, calibration):
    """Use the original scalar implementation; no new fit or score-version choice."""
    from .scoring import apply_calibration
    components=('r_tf','r_ms','r_pred')
    if set(raw_components)!=set(components) or len({len(raw_components[k]) for k in components})!=1:
        raise ValueError('All three matching canonical residual columns required')
    result=[]
    for values in zip(*(raw_components[k] for k in components)):
        scored=apply_calibration(dict(zip(components,values)),calibration)
        if not scored['valid']:
            raise ValueError('No dropping component-invalid points for a favorable auxiliary ranking')
        result.append(scored['score'])
    return np.asarray(result,dtype=np.float64)


def run(config):
    began = time.monotonic(); gate = admission(config)
    from sklearn.metrics import average_precision_score, roc_auc_score
    from . import evaluation
    from .public_native_evaluation import event_diagnostics
    from .controlled_wave_evaluation import require_metric_dependencies, reject_dependency_na
    require_metric_dependencies()
    baseline_loader = import_bound(config['baseline_loader'], 'accepted_psm_comparison')
    formatter = import_bound(config['operating_formatter'], 'accepted_operating_formatter').operating_columns
    output = Path(config['output']); output.mkdir(parents=True,exist_ok=False)
    d = config['candidate']
    archive = archive_scalars(dict(manifest=d['bank_manifest'],calibration=d['source_CAL1024']), output/'CANDIDATE_CANONICAL_SCALARS.npz')
    write(output/'RAW_SEALED_BEFORE_LABELS.json',dict(gate=gate,archive=archive,labels_read=False))
    # Sole first numerical truth access; complete metadata and scalar closure above.
    with np.load(bound(config['truth']),allow_pickle=False) as raw:
        labels = raw['full_labels'].copy()
    if labels.shape != (87841,):
        raise ValueError('Original full event axis required')
    with np.load(archive['path'],allow_pickle=False) as raw:
        new_data = {k:raw[k].copy() for k in ('index','score','valid','available_index','r_tf','r_ms','r_pred')}
    descriptors = list(config['baselines']) + [dict(id=d['id'],online=True,threshold=archive['threshold'],
        protocol='source8000_score_aligned_aux0.1_margin0.1_original_h1_R3_CAL1024',role_permissions='sourceFIT_only_no_targetFIT_or_targetCAL')]
    if len(descriptors)!=5 or len({v['id'] for v in descriptors})!=5 or sum(v['online'] for v in descriptors)!=3:
        raise ValueError('Exactly five fixed methods, three causal operating streams required')
    common=np.arange(512,87744); y=labels[common]; rows=[]; phases=[]
    for descriptor in descriptors:
        name=descriptor['id']; data=new_data if name==d['id'] else baseline_loader.load_method(descriptor,labels)
        where=crop(data,common); score=data['score'][where]
        available=data.get('available_index',data.get('available'))[where]
        row=dict(method=name,points=len(common),positive_points=int(y.sum()),AP=float(average_precision_score(y,score)),
            ROC_AUC=float(roc_auc_score(y,score)),online=descriptor['online'],threshold=descriptor.get('threshold'),
            protocol=descriptor['protocol'],role_permissions=descriptor['role_permissions'])
        folder=output/name;folder.mkdir()
        with (folder/'common_axis_inputs.npz').open('xb') as handle:
            np.savez_compressed(handle,index=common,score=score,valid=np.ones(len(common),bool),available_index=available,full_labels=labels)
        if descriptor['online']:
            if (available>common).any():raise PermissionError('Offline evidence cannot become online by axis cropping')
            if name!=d['id']:
                result=reuse_operating(descriptor,common,score,available,labels)
                write(folder/'REUSED_ACCEPTED_OPERATING_RESULT.json',descriptor['accepted_common_evaluation'])
            else:
                result=evaluation.evaluate_arrays(index=common,score=score,valid=np.ones(len(common),bool),available_index=available,
                    invalid_reasons=[None]*len(common),full_labels=labels,threshold=descriptor['threshold'],bank_complete=True,
                    original_length=87841,warmup_points=512,affiliation_identity=bound(config['affiliation_identity']))
                reject_dependency_na(result)
                evaluation.save_evaluation(folder/'fixed_source_tau_online',result,index=common,score=score,valid=np.ones(len(common),bool),
                    available_index=available,invalid_reasons=[None]*len(common),full_labels=labels,provenance=dict(config=config,method=name))
            p=event_diagnostics(result);phases.extend(dict(method=name,**v) for v in p)
            row.update(corrected_operating_columns(formatter,result,p))
        else:
            row['online_NA_reason']='native retrospective windows/batch and target statistics/training; same axis does not equal same permissions'
        rows.append(row)
    baseline=next(v for v in rows if v['method']==BASELINE);new=next(v for v in rows if v['method']==d['id'])
    delta={key:float(new[key]-value) for key,value in baseline.items() if key not in ('points','positive_points','threshold','online')
        and isinstance(value,(int,float)) and not isinstance(value,bool) and isinstance(new.get(key),(int,float))}
    table(output/'all_methods_common_axis.csv',rows);table(output/'candidate_minus_source.csv',[delta]);table(output/'event_phases.csv',phases)
    # Prespecified attribution only. Main five canonical rows above are immutable.
    auxfolder=output/'fixed_attribution_diagnostics';auxfolder.mkdir()
    where=crop(new_data,common)
    components={key:new_data[key][where] for key in ('r_tf','r_ms','r_pred')}
    component_rows=[]
    for key,score in components.items():
        if not np.isfinite(score).all():raise ValueError('Nonfinite required component')
        component_rows.append(dict(component=key,points=len(common),positive_points=int(y.sum()),
            AP=float(average_precision_score(y,score)),ROC_AUC=float(roc_auc_score(y,score)),
            operational_NA_reason='raw component attribution only; no component threshold selected'))
    table(auxfolder/'candidate_original_components.csv',component_rows)
    oldcal=read(config['fixed_auxiliary_diagnostics']['original_source_calibration'])
    fixedscore=fixed_old_calibration_scores(components,oldcal)
    available=new_data['available_index'][where]
    fixedresult=evaluation.evaluate_arrays(index=common,score=fixedscore,valid=np.ones(len(common),bool),
        available_index=available,invalid_reasons=[None]*len(common),full_labels=labels,threshold=oldcal['threshold'],
        bank_complete=True,original_length=87841,warmup_points=512,affiliation_identity=bound(config['affiliation_identity']))
    reject_dependency_na(fixedresult)
    evaluation.save_evaluation(auxfolder/'candidate_in_original_c00_source_units',fixedresult,index=common,score=fixedscore,
        valid=np.ones(len(common),bool),available_index=available,invalid_reasons=[None]*len(common),full_labels=labels,
        provenance=dict(config=config,diagnostic_only=True,new_fit=False,primary_scores_changed=False))
    fixedphases=event_diagnostics(fixedresult)
    fixedrow=dict(method=d['id']+'_OLD_C00_SOURCE_CAL_DIAGNOSTIC',points=len(common),positive_points=int(y.sum()),
        AP=float(average_precision_score(y,fixedscore)),ROC_AUC=float(roc_auc_score(y,fixedscore)),threshold=oldcal['threshold'])
    fixedrow.update(corrected_operating_columns(formatter,fixedresult,fixedphases))
    table(auxfolder/'candidate_fixed_old_source_calibration.csv',[fixedrow])
    table(auxfolder/'fixed_old_source_event_phases.csv',fixedphases)
    write(auxfolder/'DIAGNOSTIC_RECEIPT.json',dict(component_rows=component_rows,fixed_old_source_calibration=fixedrow,
        original_source_baseline=baseline,primary_candidate=new,new_fit=False,new_forward=False,selection_allowed=False,
        primary_rows_unchanged=True,scope='Attribution diagnostics only; gains cannot be selected into a new readout.'))
    write(output/'RESULT.json',dict(status='COMPLETED_SINGLE_SOURCE_DISCRIMINATION_PSM',seconds=time.monotonic()-began,
        common_axis=[512,87744],rows=rows,candidate_minus_source=delta,source_config=config,labels_after_raw_seal=True,
        GPUh=0,new_forward=False,new_training=False,limits=config['limits'],auxiliary_diagnostics='fixed_attribution_diagnostics/DIAGNOSTIC_RECEIPT.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--trusted-config-sha256',required=True)
    args=parser.parse_args()
    if sha(args.config)!=args.trusted_config_sha256:raise PermissionError('Final configuration hash mismatch')
    run(json.loads(Path(args.config).read_text(encoding='utf-8')))


if __name__=='__main__':main()
