import unittest
import numpy as np
from src.exploration.uncertainty import UncertaintyFusion
from src.metrics.pose import pose_correspondences,score_pose_correspondences,pose_temporal_metrics


class DResearchTests(unittest.TestCase):
    def setUp(self):
        self.z=np.full((24,32),2.,np.float32)
        self.rgb=np.full((24,32,3),100,np.uint8)
        self.flow=np.zeros((24,32,2),np.float32)

    def test_confidence_changes_trust_in_noisy_current_observation(self):
        results=[]
        for enabled in (False,True):
            f=UncertaintyFusion(use_confidence=enabled)
            f.step(self.z,self.rgb,variance=np.full_like(self.z,.001))
            z,_=f.step(self.z*1.05,self.rgb,self.flow,self.flow,variance=np.full_like(self.z,.1))
            results.append(z)
        self.assertLess(float(abs(results[1]-self.z).mean()),float(abs(results[0]-self.z).mean()))

    def test_scene_cut_clears_even_photometrically_similar_patch(self):
        changed=self.rgb.copy(); changed[6:]=255
        for enabled in (False,True):
            f=UncertaintyFusion(enable_reset=enabled)
            f.step(self.z,self.rgb)
            z,d=f.step(self.z*1.05,changed,self.flow,self.flow)
            self.assertEqual(d['reset'],enabled)
            self.assertEqual(f.resets,int(enabled))
            if enabled: np.testing.assert_array_equal(z,self.z*1.05)
            else: self.assertGreater(float(d['weight'][:6].mean()),0.)

    def test_low_coverage_reset_preserves_invalid_mask(self):
        f=UncertaintyFusion(minimum_coverage=.1,scene_cut_threshold=.9)
        f.step(self.z,self.rgb)
        current=self.z.copy(); current[0,0]=np.nan
        z,d=f.step(current,self.rgb+60,self.flow,self.flow)
        self.assertTrue(d['reset']); self.assertTrue(np.isnan(z[0,0]))
        np.testing.assert_array_equal(z[1:],current[1:])

    def test_large_disocclusion_rejects_stale_history(self):
        f=UncertaintyFusion(); f.step(self.z,self.rgb)
        z,d=f.step(self.z*2,self.rgb,self.flow,self.flow)
        np.testing.assert_array_equal(z,self.z*2)
        self.assertEqual(float(d['weight'].max()),0.)

    def test_shared_pose_visibility_is_independent_of_predictions(self):
        pose=[0,0,0,0,0,0,1]; k=dict(fx=30,fy=30,cx=15.5,cy=11.5)
        gt=self.z.copy(); gt[3:6,4:8]=np.nan
        c=pose_correspondences(gt,gt,pose,pose,k)
        source=c['source'].copy(); pixel=c['pixel'].copy()
        direct=pose_temporal_metrics(self.z,self.z*1.1,gt,gt,pose,pose,k)
        cached=score_pose_correspondences(self.z,self.z*1.1,c)
        self.assertEqual(direct,cached)
        score_pose_correspondences(self.z*2,self.z*3,c)
        np.testing.assert_array_equal(c['source'],source)
        np.testing.assert_array_equal(c['pixel'],pixel)


if __name__=='__main__': unittest.main()
