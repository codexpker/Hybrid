"""Synthetic dataframe and config tests only, no RData/real arrays/model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from hybrid_tsad.tep_faulty_inputs import (unique_dataframe,group_arrays,config_metadata,
    require_release,validate_table_header,validate_candidate_rows,VARIABLES,RAW_PATH,RAW_BYTES,RAW_MD5,OUTPUT)


class FaultyInputs(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(faultNumber=np.ones(960),simulationRun=np.full(960,51.),sample=np.arange(1.,961.),
            xmeas_9=np.linspace(0.,1.,960),xmeas_21=np.linspace(2.,3.,960),xmv_10=np.linspace(4.,5.,960),
            _source_row_0based=np.arange(10000,10960,dtype=np.int64)))

    def config(self):
        return dict(operation='TEP_FAULTY_INPUT_EXTRACTION',raw_path=RAW_PATH,raw_bytes=RAW_BYTES,published_MD5=RAW_MD5,
            raw_SHA256_before_execution=None,fault_ids=list(range(1,21)),run_ids=list(range(51,71)),samples_1based=[1,960,1],
            columns=list(VARIABLES),output=OUTPUT,max_seconds=900,max_RSS_GiB=24,max_AS_GiB=32,
            minimum_MemAvailable_GiB=48,minimum_disk_free_GiB=5,expected_object_name=None,
            object_rule='exactly_one_DataFrame_record_actual_name',helper_hashes={},
            official_metadata={'path':'/home/a312/tsad_research/metadata.json','sha256':'a'*64},
            contract={'path':'/home/a312/tsad_research/contract.md','sha256':'b'*64})

    def test_unique_object_rule_never_chooses_arbitrary_table(self):
        f=self.frame();name,actual=unique_dataframe({'observed_name':f})
        self.assertEqual(name,'observed_name');self.assertIs(actual,f)
        for objects in [{},{'a':f,'b':f},{'a':np.zeros((1,1))}]:
            with self.assertRaises(ValueError):unique_dataframe(objects)

    def test_raw_float64_and_original_axes(self):
        arrays=group_arrays(self.frame(),1,51)
        self.assertEqual((arrays['values'].shape,arrays['values'].dtype),((960,3),np.float64))
        np.testing.assert_array_equal(arrays['sample_1based'],np.arange(1,961))
        np.testing.assert_array_equal(arrays['source_row_0based'],np.arange(10000,10960))
        self.assertEqual(set(arrays),{'values','sample_1based','source_row_0based'})

    def test_duplicate_missing_order_or_wrong_identity_fails(self):
        for case in ['duplicate','missing','order','run']:
            f=self.frame()
            if case=='duplicate':f.loc[2,'sample']=2
            elif case=='missing':f=f.iloc[:-1]
            elif case=='order':f=f.iloc[::-1]
            else:f.loc[2,'simulationRun']=52
            with self.assertRaises(ValueError):group_arrays(f,1,51)
        with self.assertRaises(PermissionError):group_arrays(self.frame(),1,50)

    def test_bad_column_dtype_or_nonfinite_fails(self):
        for case in ['column','float32','nonfinite','row']:
            f=self.frame()
            if case=='column':f=f.drop(columns=['xmv_10'])
            elif case=='float32':f.loc[:,'xmeas_9']=f.xmeas_9.astype(np.float32);f=f.astype({v:np.float32 for v in VARIABLES})
            elif case=='nonfinite':f.loc[0,'xmeas_9']=np.nan
            else:f.loc[1,'_source_row_0based']=10000
            with self.assertRaises((ValueError,KeyError)):group_arrays(f,1,51)

    def test_table_header_rejects_small_wrong_table(self):
        with self.assertRaises(ValueError):validate_table_header(self.frame())

    def test_metadata_and_wrong_paths_roles_do_not_open_raw(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';cfg=self.config();p.write_text(json.dumps(cfg))
            self.assertEqual(config_metadata(p)['raw_SHA256_before_execution'],None)
            for key,value in [('raw_path',RAW_PATH.replace('/','\\')),('run_ids',list(range(1,21))),
                              ('columns',list(reversed(VARIABLES))),('raw_SHA256_before_execution','f'*64)]:
                bad=copy.deepcopy(cfg);bad[key]=value;p.write_text(json.dumps(bad))
                with self.assertRaises(PermissionError):config_metadata(p)

    def test_pending_release_refused_before_helper_or_raw_access(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cfg.json';r=Path(d)/'release.json'
            cfg=self.config();p.write_text(json.dumps(cfg));r.write_text(json.dumps({'status':'PENDING_ROOT_RELEASE'}))
            with self.assertRaises(PermissionError):require_release(p,r,cfg)

    def test_fixed400_candidate_identity_plan_synthetic_only(self):
        f=pd.DataFrame(dict(faultNumber=np.repeat(np.arange(1,21),20*960),
            simulationRun=np.tile(np.repeat(np.arange(51,71),960),20),sample=np.tile(np.arange(1,961),400)))
        self.assertTrue(validate_candidate_rows(f))
        with self.assertRaises(ValueError):validate_candidate_rows(f.iloc[:-1])
        f.loc[f.faultNumber.eq(20)&f.simulationRun.eq(70),'simulationRun']=69
        with self.assertRaises(ValueError):validate_candidate_rows(f)


if __name__=='__main__':unittest.main()
