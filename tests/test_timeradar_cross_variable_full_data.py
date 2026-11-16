import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from hybrid_tsad.timeradar_cross_variable_data import git_blob,_smd_record


class NativeSMDData(unittest.TestCase):
    def test_git_blob_identity_uses_git_header(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'empty';p.write_bytes(b'')
            self.assertEqual(git_blob(p),'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')

    def test_roles_fit_only_guard_and_unknown_tail(self):
        with tempfile.TemporaryDirectory()as d:
            w=Path(d);out=w/'data/test_roles';(out/'evaluation').mkdir(parents=True)
            rng=np.random.default_rng(2024);train=rng.normal(size=(1500,38));test=rng.normal(size=(1207,38))
            label=np.zeros(1207);label[1100:]=1;paths=[]
            for role,values in [('train',train),('test',test),('test_label',label)]:
                path=w/f'{role}.txt';np.savetxt(path,values,delimiter=',',fmt='%.6f');paths.append(dict(role=role,path=str(path)))
            record,evaluation=_smd_record(w,dict(id='fixture',sources=paths),out)
            roles={r['role']:r for r in record['roles']}
            self.assertEqual(roles['FIT']['bounds_half_open'],[0,900])
            self.assertEqual(roles['SELECT']['bounds_half_open'],[1000,1200])
            self.assertEqual(roles['CAL']['bounds_half_open'],[1300,1500])
            for role,spec in roles.items():
                with np.load(w/spec['path'],allow_pickle=False)as z:
                    self.assertEqual(set(z.files),{'x','index'})
                    np.testing.assert_array_equal(z['index'],np.arange(*spec['bounds_half_open']))
            with np.load(w/record['scaler']['path'])as z:
                np.testing.assert_allclose(z['mean'],np.round(train[:900],6).mean(0),rtol=0,atol=1e-12)
            self.assertEqual(evaluation['unscored_tail'],7)
            event,=evaluation['events']
            self.assertFalse(event['eligible_for_complete_event_metric'])
            self.assertEqual(event['missing_indices'],list(range(1200,1207)))


if __name__=='__main__':unittest.main()
