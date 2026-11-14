"""Only the additional accepted full-axis baseline reuse boundary."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.hybrid_v1 import target_scope_psm_comparison as target


class FullReuse(unittest.TestCase):
    def test_no_common_crop_masquerades_as_full(self):
        with self.assertRaises(ValueError):target.reuse_full_mean({},np.arange(512,87744),None,None,None)

    def test_actual_full_inputs_and_threshold_required(self):
        index=np.arange(512,87841);score=np.zeros(len(index));labels=np.zeros(87841,int)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fake.npz'
            np.savez_compressed(path,index=index,score=score,available_index=index,full_labels=labels)
            d=dict(threshold=2.,accepted_full_evaluation=dict(result={'path':'result'},provenance={'path':'provenance'},
                inputs={'path':str(path),'sha256':'fakehash'},original_gate_id='history_mean512'))
            metadata={'result':{'fixture':'accepted'},'provenance':dict(calculation_inputs_sha256='fakehash',
                provenance={'gates':[dict(id='history_mean512',threshold=2.,records=87329)]})}
            with patch.object(target,'read',side_effect=lambda a:metadata[a['path']]),patch.object(target,'bound',side_effect=lambda a:Path(a['path'])):
                self.assertEqual(target.reuse_full_mean(d,index,score,index,labels),{'fixture':'accepted'})
                d['threshold']=3.
                with self.assertRaises(ValueError):target.reuse_full_mean(d,index,score,index,labels)


if __name__=='__main__':unittest.main(verbosity=2)
