from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.exploration.application import ObstacleWarning,warning_metrics
from src.exploration.calibration import fit_scale,apply_scale
from src.exploration.compare import DirectionProcessor,compare,load_calibration,select_winners
from src.exploration.edge_aware import refine_edges
from src.exploration.efficient import ReducedFlow
from src.exploration.uncertainty import UncertaintyFusion
from src.io import RunWriter,write_json


class DirectionTests(unittest.TestCase):
    def test_D_guarded_bounds_updates_and_preserves_boundary(self):
        f=UncertaintyFusion(alpha_max=.25,max_log_update=.01,edge_threshold=.08)
        z=np.full((24,32),2.,np.float32)
        z[:,16:]=4.
        rgb=np.full((24,32,3),100,np.uint8)
        flow=np.zeros((24,32,2),np.float32)
        f.step(z,rgb)
        current=z*1.15
        result,debug=f.step(current,rgb,flow,flow,variance=np.full_like(z,.1))
        self.assertLessEqual(float(np.max(abs(np.log(result/current)))),.010001)
        np.testing.assert_allclose(result[:,15:17],current[:,15:17])
        self.assertGreater(float(debug['weight'][:,2:12].mean()),0.)

    def test_E_local_detects_small_changed_region(self):
        before=np.zeros((64,64,3),np.uint8)
        after=before.copy()
        after[24:32,24:32]=255
        f=ReducedFlow(.5,adaptive=True,local_motion=True)
        f(before,after)
        self.assertEqual(f.used_scales,[1.])
        original=ReducedFlow(.5,adaptive=True)
        original(before,after)
        self.assertEqual(original.used_scales,[.5])

    def test_F_sparse_data_breaks_confirmation_streak(self):
        f=ObstacleWarning(confirmations=2,min_coverage=.5)
        near=np.ones((8,8),np.float32)
        self.assertFalse(f.step(near)['warning'])
        sparse=np.full_like(near,np.nan)
        sparse[0,0]=1
        self.assertIsNone(f.step(sparse)['warning'])
        self.assertFalse(f.step(near)['warning'])
        self.assertTrue(f.step(near)['warning'])

    def test_selection_rejects_excess_delta1_loss(self):
        rows=[dict(method='RAW',AbsRel=.1,delta1=.9,edge_F1=.8,temporal_log_error=.03),
              dict(method='A',AbsRel=.105,delta1=.897,edge_F1=.8,temporal_log_error=.025),
              dict(method='D',AbsRel=.106,delta1=.89,edge_F1=.8,temporal_log_error=.02)]
        self.assertEqual(select_winners(rows)['accuracy_boundary_constrained_temporal_best'],'A')

    def setUp(self):
        self.z=np.full((24,32),2.,np.float32)
        self.rgb=np.full((24,32,3),100,np.uint8)
        self.flow=np.zeros((24,32,2),np.float32)

    def test_B_recovers_scale_and_rejects_outlier(self):
        fit=fit_scale([1.25,2.5,3.75,5,100],[1,2,3,4,5])
        self.assertAlmostEqual(fit['scale'],.8)
        self.assertEqual(fit['inliers'],4)
        np.testing.assert_allclose(apply_scale(self.z,fit['scale']),1.6)

    def test_B_requires_sufficient_measurements(self):
        with self.assertRaises(ValueError):
            fit_scale([1,2],[1,2])

    def test_C_keeps_constant_depth_and_holes(self):
        depth=self.z.copy()
        depth[2,3]=np.nan
        out=refine_edges(depth,self.rgb)
        self.assertTrue(np.isnan(out[2,3]))
        np.testing.assert_allclose(out[np.isfinite(out)],2.,rtol=1e-6)

    def test_C_preserves_step_edge(self):
        z=self.z.copy()
        z[:,16:]=4
        rgb=self.rgb.copy()
        rgb[:,16:]=255
        out=refine_edges(z,rgb)
        np.testing.assert_allclose(out,z,atol=1e-4)

    def test_C_reduces_local_noise(self):
        rng=np.random.default_rng(7)
        noisy=self.z*np.exp(rng.normal(0,.04,self.z.shape)).astype(np.float32)
        out=refine_edges(noisy,self.rgb)
        self.assertLess(np.std(out),np.std(noisy))

    def test_D_high_current_uncertainty_uses_history(self):
        f=UncertaintyFusion()
        f.step(self.z,self.rgb,variance=np.full_like(self.z,.001))
        result,debug=f.step(self.z*1.05,self.rgb,self.flow,self.flow,variance=np.full_like(self.z,.1))
        self.assertGreater(float(debug['weight'].mean()),.6)
        self.assertLess(np.mean(abs(result-2)),.06)

    def test_D_occlusion_rejects_history(self):
        f=UncertaintyFusion()
        f.step(self.z,self.rgb)
        result,debug=f.step(self.z*1.05,self.rgb,self.flow+50,self.flow)
        np.testing.assert_allclose(result,self.z*1.05)
        self.assertEqual(float(debug['weight'].sum()),0)

    def test_D_preserves_invalid_current_pixels(self):
        f=UncertaintyFusion()
        f.step(self.z,self.rgb)
        depth=self.z.copy()
        depth[4,4]=np.nan
        with np.errstate(invalid='raise'):
            result,_=f.step(depth,self.rgb,self.flow,self.flow)
        self.assertTrue(np.isnan(result[4,4]))
        self.assertTrue(np.isfinite(result[np.isfinite(depth)]).all())

    def test_E_rescales_flow_vectors(self):
        f=ReducedFlow(.5)
        f.backend=lambda a,b:(np.ones((*a.shape[:2],2),np.float32),-np.ones((*a.shape[:2],2),np.float32))
        forward,backward=f(self.rgb,self.rgb)
        self.assertEqual(forward.shape,(24,32,2))
        np.testing.assert_allclose(forward,2.)
        np.testing.assert_allclose(backward,-2.)

    def test_E_adaptive_keeps_full_resolution_on_motion(self):
        f=ReducedFlow(.5,adaptive=True)
        f(self.rgb,np.full_like(self.rgb,255))
        self.assertEqual(f.used_scales,[1.])

    def test_F_hysteresis_and_missing_not_safe(self):
        f=ObstacleWarning(threshold_m=1.5,confirmations=2)
        self.assertFalse(f.step(self.z*.5)['warning'])
        self.assertTrue(f.step(self.z*.5)['warning'])
        self.assertTrue(f.step(self.z*.78)['warning'])
        self.assertIsNone(f.step(self.z*np.nan)['warning'])

    def test_F_metrics_count_missing(self):
        m=warning_metrics([True,False,None],[True,False,True])
        self.assertEqual(m['warning_recall'],.5)
        self.assertEqual(m['warning_unavailable'],1)

    def test_all_directions_preserve_identity(self):
        for method in ('RAW','A','B','C','D','E_half','E_adaptive','F','B+C+D'):
            p=DirectionProcessor(method,{'scale':1.})
            for _ in range(2):
                result,_=p.step(self.z,self.rgb)
                np.testing.assert_allclose(result,self.z,rtol=1e-6)

    def test_comparison_skips_B_without_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            writer=RunWriter(root/'raw',dict(model='test',synthetic=True,depth_type='metric',unit='meter',
                representation='depth',fps=10.,visualization_range=[0,5]))
            for i in range(2):
                writer.add(self.z,self.rgb,'test',i,{'sample_id':f'test{i}'})
            writer.finish({'frames':2})
            report=compare(root/'raw',root/'comparison',methods=['RAW','B'],repeats=1)
            self.assertIn('B',report['skipped'])
            self.assertEqual(len(report['methods']),1)

    def test_calibration_cannot_use_test_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            writer=RunWriter(root/'cal',dict(model='test',depth_type='metric',unit='meter',representation='depth'))
            writer.add(self.z,self.rgb,'test',0,{'sample_id':'overlap'})
            writer.finish({'frames':1})
            write_json(root/'calibration.json',dict(source='independent_measured_landmarks',run='cal',
                points=[dict(sample_id='overlap',x=5,y=5,depth_m=2.)]))
            with self.assertRaisesRegex(ValueError,'overlap'):
                load_calibration(root/'calibration.json',[{'sample_id':'overlap'}])

    def test_calibration_duplicate_landmarks_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            writer=RunWriter(root/'cal',dict(model='test',depth_type='metric',unit='meter',representation='depth'))
            writer.add(self.z,self.rgb,'test',0,{'sample_id':'dev'})
            writer.finish({'frames':1})
            write_json(root/'calibration.json',dict(source='independent_measured_landmarks',run='cal',
                points=[dict(sample_id='dev',x=5,y=5,depth_m=2.)]*3))
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                load_calibration(root/'calibration.json',[{'sample_id':'test'}])

    def test_application_scoring_skips_sparse_GT(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            writer=RunWriter(root/'raw',dict(model='test',synthetic=True,depth_type='metric',unit='meter',
                representation='depth',fps=10.,visualization_range=[0,5]))
            samples=[]
            for i in range(2):
                writer.add(self.z,self.rgb,'test',i,{'sample_id':f'test{i}'})
                target=self.z.copy()
                target[0,0]=0
                np.save(root/f'gt{i}.npy',target)
                samples.append(dict(id=f'test{i}',rgb=f'raw/rgb/{i:06d}.png',depth=f'gt{i}.npy',
                                    timestamp=float(i),pose_world_camera=[0,0,0,0,0,0,1]))
            writer.finish({'frames':2})
            write_json(root/'intrinsics.json',dict(fx=20.,fy=20.,cx=15.5,cy=11.5))
            write_json(root/'test.json',dict(protocol=dict(min_depth=.1,max_depth=10.),samples=samples,
                pose_convention='T_world_camera',intrinsics='intrinsics.json'))
            report=compare(root/'raw',root/'comparison',manifest=root/'test.json',methods=['RAW'],repeats=1)
            self.assertEqual(report['methods'][0]['AbsRel'],0.)
            self.assertIsNone(report['methods'][0]['warning_F1'])
            self.assertAlmostEqual(report['methods'][0]['pose_temporal_log_error'],0.)


if __name__=='__main__':
    unittest.main()
