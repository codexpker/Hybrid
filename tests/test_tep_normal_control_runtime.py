"""Small synthetic parse hooks; real read_r/raw files/model are never opened."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
import pandas as pd
from hybrid_tsad import tep_normal_control_runtime as runtime

def groups(frame):
    return [(r,dict(values=np.full((960,3),float(r),np.float64),sample_1based=np.arange(1,961,dtype=np.int64),
        source_row_0based=np.arange((r-1)*960,r*960,dtype=np.int64))) for r in range(51,71)]

class ParseFixture(unittest.TestCase):
    def test_exactly_once_mock_reader_and_twenty_normal_groups(self):
        reader=mock.Mock(return_value={'fault_free_testing':pd.DataFrame({'fixture':[1]})})
        name,result=runtime.read_normal_once('synthetic_no_file',reader,group_function=groups)
        reader.assert_called_once_with('synthetic_no_file')
        self.assertEqual(name,'fault_free_testing');self.assertEqual([r for r,a in result],list(range(51,71)))

    def test_wrong_object_count_type_and_name_no_retry(self):
        for value in ({'faulty_testing':pd.DataFrame()}, {'fault_free_testing':object()},
                      {'fault_free_testing':pd.DataFrame(),'extra':pd.DataFrame()}):
            reader=mock.Mock(return_value=value)
            with self.assertRaises(ValueError):runtime.read_normal_once('synthetic',reader,group_function=groups)
            self.assertEqual(reader.call_count,1)

    def test_reader_failure_never_retried(self):
        reader=mock.Mock(side_effect=ValueError('mock parse failure'))
        with self.assertRaises(ValueError):runtime.read_normal_once('synthetic',reader,group_function=groups)
        self.assertEqual(reader.call_count,1)

    def test_existing_output_refuses_before_reader(self):
        with tempfile.TemporaryDirectory() as td:
            reader=mock.Mock(side_effect=AssertionError('No parser'))
            with self.assertRaises(FileExistsError):runtime.parse_and_write('synthetic',td,reader,group_function=groups)
            reader.assert_not_called();self.assertEqual(list(Path(td).iterdir()),[])

    def test_bad_normal_group_not_accepted(self):
        reader=mock.Mock(return_value={'fault_free_testing':pd.DataFrame()})
        for bad in (lambda frame:groups(frame)[:-1],lambda frame:list(reversed(groups(frame)))):
            with self.assertRaises(ValueError):runtime.read_normal_once('synthetic',reader,group_function=bad)

    def test_pending_release_no_source_ref_access(self):
        with mock.patch.object(runtime,'sha',return_value='a'*64),mock.patch.object(runtime,'read',return_value={'status':'PENDING'}) as read:
            with self.assertRaises(PermissionError):runtime.require_release('fakecfg','fakerelease','a'*64,{'binding_state':'PENDING','helper_hashes':{}})
            self.assertEqual(read.call_count,1)

if __name__=='__main__':unittest.main(verbosity=2)
