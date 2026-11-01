"""Small normal-only data/schedule checks, no TEP file or model access."""
import unittest
import numpy as np
from hybrid_tsad.tep_normal_assets import lag_profiles,window_schedule,validate_roles,RAW_SHA,ROLES


class TEPNormalAssetsTests(unittest.TestCase):
    def test_past_lag_matches_explicit_within_run_Pearson(self):
        rng=np.random.default_rng(91)
        arrays=[rng.normal(size=(500,3)),rng.normal(size=(500,3))+20]
        rows=lag_profiles(arrays)
        self.assertEqual(len(rows),3*3*64)
        row=next(x for x in rows if x['target']=='xmeas_9' and x['past_predictor']=='xmv_10' and x['lag_points']==64)
        y=np.concatenate([a[64:,0] for a in arrays]);past=np.concatenate([a[:436,2] for a in arrays])
        self.assertAlmostEqual(row['Pearson'],float(np.corrcoef(y,past)[0,1]),places=13)
        self.assertEqual(row['paired_points'],872)
        self.assertEqual(row['lag_minutes'],192)

    def test_schedule_and_prefault_context_are_not_hidden_changes(self):
        a=window_schedule()
        self.assertEqual((a['FIT_windows'],a['batch128_updates_per_epoch'],a['max_optimizer_updates']),(12030,94,1880))
        self.assertEqual(a['last_batch'],126)
        self.assertAlmostEqual(a['maximum_training_lambda'],.132)
        self.assertFalse(a['full_ramp_reachable_within_20epochs'])
        self.assertEqual(a['prefault_H100_stride1_complete_windows'],61)
        self.assertEqual(a['prefault_H192_complete_windows'],0)
        self.assertFalse(a['H192_automatically_modified'])

    def test_old_roles_exact_and_no_reassignment(self):
        records=[{'role':role,'selector':{'faultNumber':0,'simulationRun':i},'source_identity':RAW_SHA,
                  'local_start':0,'local_end_exclusive':500} for role,ids in ROLES.items() for i in ids]
        self.assertEqual(validate_roles({'records':records}),ROLES)
        records[0]['selector']['simulationRun']=21
        with self.assertRaises(ValueError):validate_roles({'records':records})


if __name__=='__main__':unittest.main()
