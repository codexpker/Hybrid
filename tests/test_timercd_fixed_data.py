import unittest
import numpy as np
from hybrid_tsad.timercd_fixed_data import validate_raw_role


class TimeRCDFixedData(unittest.TestCase):
    def test_original_raw_matches_accepted_float32_roles(self):
        raw=np.array([[1.123456789,2.0],[2.34567891,2.0]],dtype=np.float64)
        mean=np.array([1.,2.]);scale=np.array([.25,1.])
        accepted=((raw-mean)/scale).astype(np.float32)
        validate_raw_role(raw,accepted,mean,scale)
        with self.assertRaises(ValueError):validate_raw_role(raw+.001,accepted,mean,scale)
        with self.assertRaises(ValueError):validate_raw_role(raw.astype(np.float32),accepted,mean,scale)


if __name__=='__main__':unittest.main()
