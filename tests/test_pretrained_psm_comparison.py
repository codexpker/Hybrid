"""CPU-only gate, scalar fidelity and paired statistics; no scientific QUERY."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.hybrid_v1 import pretrained_psm_comparison as core
from hybrid_tsad.hybrid_v1.chunked_bank import _pack


class ComparisonTests(unittest.TestCase):
    def test_replay_tolerance_never_replaces_canonical_score_or_ranking(self):
        canonical=np.array([1.+1e-13, 1., 2.],dtype=np.float64)
        original=canonical.copy(); original_rank=np.argsort(canonical).copy()
        replay=np.array([1.,1.+1e-13,2.+1e-13],dtype=np.float64)
        differences=[core.validate_score_replay(a,b) for a,b in zip(canonical,replay)]
        self.assertGreater(max(v[0] for v in differences),0.)
        self.assertTrue(np.array_equal(canonical,original))
        self.assertTrue(np.array_equal(np.argsort(canonical),original_rank))
        self.assertFalse(np.array_equal(np.argsort(canonical),np.argsort(replay)))
        for invalid in (1.+1e-8,np.inf,np.nan):
            with self.assertRaises(ValueError):core.validate_score_replay(1.,invalid)

    def test_crop_requires_exact_original_axis_and_validity(self):
        scripts=Path(__file__).resolve().parents[1]/'scripts'
        sys.path.insert(0,str(scripts))
        spec=importlib.util.spec_from_file_location('paired_comparison_test_consumer',scripts/'compare_pretrained_psm_initialization.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        common=np.arange(512,516)
        good={'index':np.arange(512,518),'score':np.ones(6),'valid':np.ones(6,bool)}
        self.assertTrue(np.array_equal(module.crop(good,common),np.arange(4)))
        bad=copy.deepcopy(good);bad['valid'][2]=False
        with self.assertRaises(ValueError):module.crop(bad,common)
        bad=copy.deepcopy(good);bad['index'][2]=514+1
        with self.assertRaises(ValueError):module.crop(bad,common)

    def test_typed_scalar_preserves_canonical_value_and_skips_prediction(self):
        record={'raw':{'score':np.float64(-0.),'index':512,'valid':True,
                       'prediction':np.full((32,25),np.nan),'domain':'PSM'}}
        arrays=_pack([record]); tree=json.loads(arrays['tree_utf8'].tobytes())[0]
        raw=core._field(tree,'raw')
        score=core._scalar(core._field(raw,'score'),arrays)
        self.assertTrue(np.signbit(score))
        self.assertEqual(core._scalar(core._field(raw,'index'),arrays),512)
        self.assertEqual(core._scalar(core._field(raw,'domain'),arrays),'PSM')
        with self.assertRaises(ValueError):
            core._scalar(core._field(raw,'prediction'),arrays)

    def test_partial_three_seed_completion_rejected(self):
        with self.assertRaises(PermissionError):
            core.validate_new_banks([{'seed':2025},{'seed':2026}])
        with self.assertRaises(PermissionError):
            core.validate_new_banks([{'seed':2025},{'seed':2025},{'seed':2027}])

    def test_source_role_and_receipt_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); descriptors=[]
            def asset(name,obj):
                path=folder/name; path.write_text(json.dumps(obj),encoding='utf-8')
                return {'path':str(path),'sha256':core.sha(path)}
            checkpoint=asset('PT.json',{'synthetic_not_model':True})
            for seed in core.SEEDS:
                calibration={'valid_windows':1024,'domains':['SMD','Sunspot'],
                    'source_domain_equal_weight':True,'weights':{'r_tf':.25,'r_ms':.25,'r_pred':.5},'threshold':2.}
                ca=asset(str(seed)+'_calibration.json',calibration)
                identity={'checkpoint_sha256':checkpoint['sha256'],'calibration_sha256':ca['sha256'],
                    'input_sha256':'1'*64,'config_sha256':'2'*64,'source_sha256':'3'*64}
                ma=asset(str(seed)+'_manifest.json',{'complete':True,'committed_records':87329,
                    'contract':{'ranges':[{'domain':'PSM','entity':'anonymous_PSM','run':'test_original','start':512,'end':87841}],
                    'identity':identity,'threshold':2.}})
                base={'seed':seed,'status':'COMPLETED','official_sha256':core.OFFICIAL,
                    'labels_read':False,'checkpoint_sha256':checkpoint['sha256'],'initializer_sha256':'4'*64}
                train={**base,'phase':'train','attempts':8000,'updates':8000,'scientific_status':'VALID_SOURCE_SELECTED_ENDPOINT'}
                cal={**base,'phase':'cal','records':1024,'calibration_sha256':ca['sha256'],'scientific_status':'VALID_SOURCE_CAL'}
                query={**base,'phase':'query','points':87329,'bank_manifest_sha256':ma['sha256'],'calibration_sha256':ca['sha256']}
                descriptors.append({'seed':seed,'train_receipt':asset(str(seed)+'train.json',train),
                    'cal_receipt':asset(str(seed)+'cal.json',cal),'query_receipt':asset(str(seed)+'query.json',query),
                    'manifest':ma,'calibration':ca,'checkpoint':checkpoint,'expected_identity':identity})
            self.assertEqual(len(core.validate_new_banks(descriptors)),3)
            # Same completed raw bank cannot conceal a target-calibration role.
            bad=copy.deepcopy(descriptors)
            obj=json.loads(Path(bad[0]['calibration']['path']).read_text());obj['source_domain_equal_weight']=False
            bad[0]['calibration']=asset('targetcal.json',obj)
            with self.assertRaises(PermissionError): core.validate_new_banks(bad)

    def test_paired_all_seeds_sample_SD_and_NA(self):
        rows=[]
        for seed,old,new in zip(core.SEEDS,(.2,.3,.4),(.3,.2,.5)):
            rows.extend([dict(initialization='RANDOM',seed=seed,AP=old,delay=None),
                         dict(initialization='OFFICIAL528',seed=seed,AP=new,delay=None)])
        paired,summary=core.paired_tables(rows)
        ap=next(v for v in summary if v['initialization']=='OFFICIAL528' and v['metric']=='AP')
        self.assertEqual(ap['n_valid'],3);self.assertEqual(ap['min_seed'],[2026])
        self.assertAlmostEqual(ap['sample_SD'],np.std([.3,.2,.5],ddof=1))
        self.assertTrue(all(v['mean'] is None for v in summary if v['metric']=='delay'))
        self.assertEqual(len([v for v in paired if v['metric']=='AP']),3)


if __name__=='__main__':unittest.main()
