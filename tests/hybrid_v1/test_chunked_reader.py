"""Synthetic-only strict parity with the immutable8189 reader; count pool reads."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hybrid_tsad.hybrid_v1 import chunked_bank as new


class Reader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[3]
        cls.reference=root/"notes/hybrid_v1_20261102/chunked_reader_v1/chunked_bank_8189_before.py"
        if not cls.reference.exists():
            # Remote test packets may explicitly bind the archived reference path.
            import os
            cls.reference=Path(os.environ["HYBRID_V1_OLD_BANK_READER"])
        assert hashlib.sha256(cls.reference.read_bytes()).hexdigest()=="818902d5fd89cfb071f6f55cc6adaac268170bc04472386f5f2ffd182eb1318f"
        spec=importlib.util.spec_from_file_location("hybrid_tsad.hybrid_v1._old8189_reader",cls.reference)
        cls.old=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.old)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/"synthetic_bank"
        self.contract=dict(identity={k:"a"*64 for k in new.IDENTITY_KEYS},threshold=1.,chunk_size=3,
            ranges=[dict(domain="synthetic",entity="C3",run="r",start=512,end=523)])
        def records():
            for index in range(512,523):
                value=float("nan") if index==515 else 1. if index==512 else 2.
                yield dict(domain="synthetic",entity="C3",run="r",index=index,available_index=index,
                    score=value,valid=bool(np.isfinite(value)),reason="NO_CHANNELS" if index==515 else None,
                    prediction=np.array([[index,-0.,np.nan]],dtype=np.float32),
                    per_channel={"r_pred":[.25,.5,.75]},
                    replay=dict(hidden=np.array([True,False,True]),signed=np.array([-7,5],np.int16),
                                empty=np.empty((0,2),np.float64),shape=[1,2,3],
                                nested=[dict(x=float(index),y=False),dict(x=np.inf,y=True)]))
        with new.ChunkedBank(self.path,**self.contract) as bank:
            bank.write(records());bank.finalize()

    def test_complete_raw_derived_and_file_bytes_exact_old_reader_parity(self):
        def hashes():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.path.iterdir() if p.is_file()}
        before=hashes();old=list(self.old.read_entries(self.path));actual=list(new.read_entries(self.path))
        self.assertEqual([new._fingerprint(e) for e in old],[new._fingerprint(e) for e in actual])
        for entry in actual:
            a=entry["raw"]["prediction"]
            self.assertEqual(a.dtype,np.float32);self.assertEqual(a.shape,(1,3))
            self.assertTrue(np.signbit(a[0,1]));self.assertTrue(np.isnan(a[0,2]))
            self.assertEqual(entry["raw"]["replay"]["signed"].dtype,np.int16)
            self.assertEqual(entry["raw"]["replay"]["empty"].shape,(0,2))
        self.assertEqual([e["raw"]["index"] for e in actual if e["derived"]["confirmed3_current"]],list(range(518,523)))
        with new.ChunkedBank(self.path,**self.contract) as reopened:
            self.assertTrue(reopened.manifest["complete"])
        self.assertEqual(before,hashes())

    def _count_loads(self,action):
        real=np.load;counts={}
        class Lazy:
            def __init__(self,path,**kwargs):
                self.path=Path(path).name;self.inner=real(path,**kwargs)
                counts[self.path]={key:0 for key in self.inner.files}
            @property
            def files(self):return self.inner.files
            def __getitem__(self,key):
                counts[self.path][key]+=1
                return self.inner[key]
            def __enter__(self):return self
            def __exit__(self,*args):self.inner.close()
        with patch.object(np,"load",Lazy):action()
        return counts

    def test_read_entries_materializes_each_lazy_member_exactly_once(self):
        counts=self._count_loads(lambda:list(new.read_entries(self.path)))
        self.assertEqual(len(counts),4)
        self.assertTrue(all(n==1 for chunk in counts.values() for n in chunk.values()))
        old_counts=self._count_loads(lambda:list(self.old.read_entries(self.path)))
        self.assertTrue(any(n>1 for chunk in old_counts.values() for n in chunk.values()))

    def test_validate_existing_materializes_each_lazy_member_exactly_once(self):
        def reopen():
            with new.ChunkedBank(self.path,**self.contract):pass
        counts=self._count_loads(reopen)
        self.assertEqual(len(counts),4)
        self.assertTrue(all(n==1 for chunk in counts.values() for n in chunk.values()))

    def test_only_two_read_lines_change_codec_writer_and_math_unchanged(self):
        old=self.reference.read_bytes()
        current=Path(new.__file__).read_bytes()
        self.assertEqual(current.count(b'_unpack({name: arrays[name] for name in arrays.files})'),2)
        restored=current.replace(b'_unpack({name: arrays[name] for name in arrays.files})',b'_unpack(arrays)')
        self.assertEqual(old,restored)


if __name__=="__main__":unittest.main()
