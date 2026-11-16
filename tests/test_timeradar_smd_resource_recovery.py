"""Synthetic metadata-only fixtures; no experiment files or model dependencies."""
import json
from pathlib import Path
import tempfile
import unittest
from hybrid_tsad.timeradar_smd_full_evaluation import verify_resource_recovery
from hybrid_tsad.timeradar_cross_variable_evaluation import sha


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8');return path


def fixture(workspace,*,missing=False):
    w=Path(workspace);root=w/'results/timeradar_smd_full_20261116_v1';prep=w/'preparations/timeradar_smd_full_20261116_v1'
    names=[f'fixture-{i}'for i in range(25)];rid=names[-1];arm='self'
    expected={f'{r}/{a}/score/{role}.npz'for r in names for a in ('zero','decoder_head','self','cross')for role in ('CAL','QUERY')}
    additions={f'{rid}/{arm}/score/{role}.npz'for role in ('CAL','QUERY')};banks=[]
    for relative in sorted(expected):
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(relative.encode())
        banks.append(dict(relative=relative,path=str(p),sha256=sha(p)))
    oldbanks=[b for b in banks if b['relative']not in additions]
    jobs=[dict(record=r,arm=a,phase=phase,exit_code=0,gpu=0,pid=i)
          for i,(r,a,phase)in enumerate((r,a,p)for r in names for a in ('zero','decoder_head','self','cross')
              for p in (('score',)if a=='zero'else('train','score'))
              if(r,a,p)!=(rid,arm,'score'))]
    reject=dict(record=rid,arm=arm,gpu=0,reason='unstarted preflight')
    original=write(root/'RAW_BANK_SEAL.json',dict(status='COMPLETED_WITH_FAILURES',banks=oldbanks,jobs=jobs,failures=[reject]))
    prior_global=write(prep/'ALL_SMD_EXTENSION_RAW_BANK_SEAL.json',dict(result_seal_sha256=sha(original)))
    rejection=write(prep/'jobs/rejection.json',reject);config=write(prep/'runtime/config_wave2.json',dict(seed=2024))
    start=write(prep/'RESOURCE_RECOVERY_START.json',dict(original_result_sha256=sha(original),original_global_sha256=sha(prior_global),
        no_retraining=True,no_prior_score_execution=True,preflight_rejection_GPU_hours=0,recovery_gpu=1,
        preflight_rejection_path=str(rejection),preflight_rejection_sha256=sha(rejection),config_sha256=sha(config),selected_sha256='a'*64))
    job=dict(record=rid,arm=arm,phase='score',exit_code=0,gpu=1,pid=999)
    command=['python','runner','score','--record',rid,'--arm',arm,'--config',str(config)]
    write(prep/'jobs'/f'{rid}__{arm}__score_LAUNCH.json',dict(pid=999,gpu=1,command=command))
    write(prep/'jobs'/f'{rid}__{arm}__score_EXIT.json',job)
    write(root/rid/arm/'train/TRAIN_RECEIPT.json',dict(record=rid,arm=arm,updates=2000,selected_sha256='a'*64))
    write(root/rid/arm/'score/SCORE_RECEIPT.json',dict(status='COMPLETED',record=rid,arm=arm,
        checkpoint_sha256='a'*64,config_sha256=sha(config),banks=[b for b in banks if b['relative']in additions]))
    raw=dict(status='COMPLETED',resource_recovery_only=True,original_198_banks_unchanged=True,
        banks=banks[:-1]if missing else banks,jobs=jobs+[job],resource_preflight_rejections=[reject],recovery_job=job,
        original_controller_result_sha256=sha(original),original_controller_global_sha256=sha(prior_global),recovery_start_sha256=sha(start),
        waves=[dict(records=names,config_sha256=sha(config))])
    seal=write(root/'RECOVERED_RAW_BANK_SEAL.json',raw)
    write(prep/'RESOURCE_RECOVERY_FINAL_SEAL.json',dict(raw,recovered_result_seal_sha256=sha(seal)))
    return w,root,seal,expected,original


class ResourceRecovery(unittest.TestCase):
    def test_valid_198_plus_two_identity_chain(self):
        with tempfile.TemporaryDirectory()as d:
            w,root,seal,expected,old=fixture(d);before=old.read_bytes()
            inputs=verify_resource_recovery(w,root,seal,expected)
            self.assertIn(str(old),inputs);self.assertEqual(old.read_bytes(),before)

    def test_changed_original_or_missing_bank_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            w,root,seal,expected,old=fixture(d);old.write_bytes(old.read_bytes()+b' ')
            with self.assertRaises(PermissionError):verify_resource_recovery(w,root,seal,expected)
        with tempfile.TemporaryDirectory()as d:
            w,root,seal,expected,old=fixture(d,missing=True)
            with self.assertRaises(ValueError):verify_resource_recovery(w,root,seal,expected)


if __name__=='__main__':unittest.main()
