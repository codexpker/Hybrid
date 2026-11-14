"""Native-axis planned profile continuation; no model/data download involved."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hybrid_tsad.hybrid_v1.public_native_zs import length_groups,write_record_segment
from hybrid_tsad.hybrid_v1.chunked_bank import read_records,read_entries


class NativeTests(unittest.TestCase):
    def test_planned_profile_commits_once_and_remainder_keeps_original_axis(self):
        with tempfile.TemporaryDirectory() as folder:
            identity={k:'a'*64 for k in ['input_sha256','config_sha256','checkpoint_sha256','calibration_sha256','source_sha256']}
            span=[dict(domain='SMD',entity='machine-fixture',run='native_test',start=8958,end=9031)]
            called=[]
            def factory(plan):
                for e in plan:
                    t=e['index'];called.append(t)
                    yield dict(domain='SMD',entity='machine-fixture',run='native_test',index=t,
                               available_index=t,score=2.,valid=True)
            first=write_record_segment(factory,folder,identity=identity,declared_ranges=span,threshold=1.,max_new_points=64)
            self.assertFalse(first['complete']);self.assertEqual(first['new_points'],64)
            self.assertEqual(len(list(read_records(folder))),64)
            with self.assertRaises(PermissionError):
                write_record_segment(factory,folder,identity=identity,declared_ranges=span,threshold=1.,max_new_points=64,resume=True)
            second=write_record_segment(factory,folder,identity=identity,declared_ranges=span,threshold=1.,resume=True)
            self.assertTrue(second['complete']);self.assertEqual(second['new_points'],9)
            self.assertEqual(called,list(range(8958,9031)))
            entries=list(read_entries(folder))
            self.assertEqual(len(entries),73)
            self.assertFalse(entries[0]['derived']['confirmed3_current'])
            self.assertTrue(entries[64]['derived']['confirmed3_current'])
            self.assertEqual(entries[-1]['raw']['index'],9030)
            before=len(called)
            done=write_record_segment(factory,folder,identity=identity,declared_ranges=span,threshold=1.,resume=True)
            self.assertTrue(done['already_complete']);self.assertEqual(len(called),before)

    def test_length_assignment_uses_only_input_metadata(self):
        entries=[dict(domain='SMAP',entity=str(i),run=str(i),role='QUERY',expected_shape=[512+n,1])
                 for i,n in enumerate([10,40,20,30])]
        a=length_groups(entries,2);b=length_groups(list(reversed(entries)),2)
        self.assertEqual(a,b);self.assertEqual(a['points'],[50,50])
        self.assertEqual(sum(len(g) for g in a['groups']),4)


if __name__=='__main__':unittest.main(verbosity=2)
