"""Actual save_bank -> production metadata glue -> production compact writer.

Only20 synthetic banks and a fake NumPy scorer; never a model/real array.
"""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from hybrid_tsad import tep_normal_control as c
from hybrid_tsad import tep_normal_control_runtime as r
from hybrid_tsad.normal_b1_readout import save_bank

def fp():return dict(state='same',gradients='none',modes={'':False},step=0,rng='same')

class SaveGlue(unittest.TestCase):
    def test20_real_saves_metadata_and_compact_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);banks=[];compacts=[]
            for run,entity in zip(c.RUNS,c.ENTITIES):
                values=np.arange(2880,dtype=np.float64).reshape(960,3)/100+run
                bank=c.collect_normal_bank(values,run=run,seed=2025,scorer=lambda batch:batch.copy(),fingerprint=fp)
                bank['source_values_float64']=values
                bank['normalized_input_windows']=np.stack([values[i:i+100].astype(np.float32) for i in range(861)])
                row=dict(entity_id=entity,simulationRun=run)
                saved=save_bank(out/entity,bank,dict(role='NORMAL_CONTROL',run=run,seed=2025,source=row))
                self.assertEqual(saved['run'],run);self.assertEqual(saved['seed'],2025)
                if run==51:
                    # Reproduce the original glue TypeError on a real saved return value.
                    with self.assertRaises(TypeError):dict(entity_id=entity,run=run,**saved)
                r.append_saved_left_bank(banks,compacts,row,2025,bank,saved)
                self.assertNotIn('entity_id',saved)  # no mutation of save_bank receipt
            r.save_left_compact(out,compacts)
            self.assertEqual([b['entity_id'] for b in banks],list(c.ENTITIES))
            self.assertEqual([b['run'] for b in banks],list(c.RUNS))
            with np.load(out/'compact_scores.npz',allow_pickle=False) as bank:
                self.assertEqual(set(bank.files),{'entity_id','last','mean32','channel_last','channel_mean32','end_index_0based','available_index_0based'})
                self.assertEqual(bank['entity_id'].tolist(),list(c.ENTITIES))
                np.testing.assert_array_equal(bank['end_index_0based'],np.tile(np.arange(99,960),(20,1)))
                np.testing.assert_array_equal(bank['available_index_0based'],bank['end_index_0based'])
                for i,entity in enumerate(c.ENTITIES):
                    with np.load(out/entity/'raw_bank.npz',allow_pickle=False) as raw:
                        for key in ['last','mean32','channel_last','channel_mean32']:
                            np.testing.assert_array_equal(bank[key][i],raw[key])

    def test_mismatched_saved_identity_refused_before_append(self):
        for key,value in [('run',52),('seed',2026),('role','QUERY'),('entity_id','entity_0001')]:
            saved=dict(run=51,seed=2025,role='NORMAL_CONTROL');saved[key]=value
            banks=[];compact=[]
            with self.assertRaises(ValueError):r.append_saved_left_bank(banks,compact,dict(entity_id='normal_run_0051',simulationRun=51),2025,{},saved)
            self.assertEqual(banks,[]);self.assertEqual(compact,[])

if __name__=='__main__':unittest.main(verbosity=2)
