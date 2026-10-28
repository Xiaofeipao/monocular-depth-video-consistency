from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.flow import warp,consistency
from src.geometry import disparity_to_depth,read_middlebury_calibration,read_pfm,backproject,measure_point,stereo_depth
from src.io import RunWriter,load_run,read_frames
from src.metrics import depth_metrics,boundary_metrics,temporal_metrics,paired_bootstrap
from src.temporal import FusionConfig,TemporalFusion


class GeometryTests(unittest.TestCase):
    def test_metric_units_and_offset(self):
        actual = disparity_to_depth(np.array([8.,18.,0.,np.nan,-1.]),100.,.2,2.)
        np.testing.assert_allclose(actual[:2],[2.,1.])
        self.assertTrue(np.isnan(actual[2:]).all())

    def test_calibration(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'calib.txt'
            path.write_text('cam0=[100 0 50; 0 100 40; 0 0 1]\nbaseline=200\ndoffs=2\nndisp=32\n')
            c = read_middlebury_calibration(path)
            self.assertEqual(c,dict(focal_px=100.,baseline_m=.2,doffs=2.,ndisp=32))

    def test_pfm_endianness_and_orientation(self):
        values = np.array([[1.,2.],[3.,4.]],np.float32)
        with tempfile.TemporaryDirectory() as root:
            for endian,scale in [('<',-1),('>',1)]:
                path = Path(root)/'disp.pfm'
                with path.open('wb') as f:
                    f.write(f'Pf\n2 2\n{scale}\n'.encode())
                    np.flipud(values).astype(endian+'f4').tofile(f)
                np.testing.assert_array_equal(read_pfm(path),values)

    def test_backprojection_and_robust_click(self):
        depth = np.full((5,5),2.,np.float32)
        depth[2,2] = 99
        k = dict(fx=2.,fy=2.,cx=2.,cy=2.)
        np.testing.assert_allclose(measure_point(depth,2,2,k),[0,0,2])
        np.testing.assert_allclose(backproject(np.ones((3,3)),2,2,1,1)[0,0],[-.5,-.5,1])
        with self.assertRaises(ValueError):
            measure_point(depth,-1,2,k)

    def test_sgbm_synthetic_rectified_shift(self):
        rng = np.random.default_rng(4)
        left = rng.integers(0,255,(96,256,3),dtype=np.uint8)
        right = np.zeros_like(left)
        right[:,:-8] = left[:,8:]
        for method in ('bm','sgbm'):
            disp,depth,valid = stereo_depth(left,right,100,.2,method=method,num_disparities=32)
            region = valid[20:-20,50:-50]
            self.assertGreater(region.mean(),.8)
            self.assertAlmostEqual(float(np.median(disp[20:-20,50:-50][region])),8,delta=.2)
            self.assertAlmostEqual(float(np.nanmedian(depth[20:-20,50:-50])),2.5,delta=.1)


class MetricTests(unittest.TestCase):
    def test_identity(self):
        gt = np.array([[1.,2.],[3.,4.]])
        m,_,_ = depth_metrics(gt,gt)
        self.assertEqual(m['AbsRel'],0.)
        self.assertEqual(m['RMSE_log'],0.)
        self.assertEqual(m['delta1'],1.)

    def test_known_scale(self):
        gt = np.ones((4,4))*2
        m,_,_ = depth_metrics(gt*2,gt)
        self.assertEqual(m['AbsRel'],1)
        self.assertEqual(m['SqRel'],2)
        self.assertEqual(m['RMSE'],2)
        self.assertAlmostEqual(m['RMSE_log'],np.log(2))
        self.assertEqual(m['delta3'],0)

    def test_mask_and_crop(self):
        gt = np.ones((4,4))
        gt[0,0] = np.nan
        pred = gt.copy()
        pred[0,1] = np.inf
        m,_,_ = depth_metrics(pred,gt,crop=[1,4,1,4])
        self.assertEqual(m['valid_pixels'],9)
        with self.assertRaises(ValueError):
            depth_metrics(pred,gt)

    def test_bad_predictions_not_silently_dropped(self):
        for value in (0,-1,np.nan,np.inf):
            pred = np.ones((3,3))
            pred[0,0] = value
            with self.assertRaises(ValueError):
                depth_metrics(pred,np.ones((3,3)))

    def test_metric_forbids_gt_alignment(self):
        with self.assertRaises(ValueError):
            depth_metrics(np.ones((3,3)),np.ones((3,3)),alignment='median')

    def test_relative_inverse_scale_shift(self):
        gt = np.linspace(1,5,25).reshape(5,5)
        relative_inverse = 3/gt+2
        m,_,_ = depth_metrics(relative_inverse,gt,depth_type='relative',representation='inverse_depth',alignment='scale_shift')
        self.assertLess(m['AbsRel'],1e-12)

    def test_boundary_holes_do_not_create_edges(self):
        gt = np.ones((20,20))
        gt[:,10:] = 2
        valid = np.ones_like(gt,bool)
        valid[3:6,3:6] = False
        m = boundary_metrics(gt,gt,valid)
        self.assertEqual(m['edge_F1'],1)
        wrong = np.ones_like(gt)
        self.assertEqual(boundary_metrics(wrong,gt,valid)['edge_recall'],0)

    def test_paired_bootstrap(self):
        r = paired_bootstrap([1,2,3],[0,1,2])
        self.assertEqual(r['mean_difference'],-1)
        self.assertEqual(r['ci95'],[-1,-1])


class FlowTests(unittest.TestCase):
    def test_backward_warp_direction(self):
        source = np.tile(np.arange(6,dtype=np.float32),(4,1))
        backward = np.zeros((4,6,2),np.float32)
        backward[...,0] = -1
        out,valid = warp(source,backward)
        np.testing.assert_array_equal(out[:,1:],source[:,:-1])
        self.assertFalse(valid[:,0].any())
        self.assertTrue(valid[:,1:].all())

    def test_forward_backward_occlusion(self):
        f = np.zeros((8,8,2),np.float32)
        b = f.copy()
        f[...,0],b[...,0] = 1,-1
        valid,error = consistency(f,b)
        self.assertTrue(valid[:,1:].all())
        self.assertFalse(valid[:,0].any())
        f[3:5,3:5] = 20
        valid,_ = consistency(f,b)
        self.assertFalse(valid[3:5,4:6].any())

    def test_temporal_metric_scale_and_flicker(self):
        z = np.ones((8,8),np.float32)*2
        flow = np.zeros((8,8,2),np.float32)
        result = temporal_metrics(z,z*2,flow,flow)
        self.assertAlmostEqual(result['temporal_log_error'],np.log(2),places=6)
        self.assertEqual(result['local_flicker'],0)
        self.assertEqual(result['coverage'],1)

    def test_empty_temporal_mask(self):
        z = np.zeros((4,4),np.float32)
        flow = np.zeros((4,4,2),np.float32)
        result = temporal_metrics(z,z,flow,flow)
        self.assertIsNone(result['temporal_log_error'])


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((24,32,3),100,np.uint8)
        self.z = np.full((24,32),2,np.float32)
        self.flow = np.zeros((24,32,2),np.float32)

    def test_all_ablations_identity(self):
        for i in range(7):
            fusion = TemporalFusion(FusionConfig(ablation=i))
            for _ in range(3):
                out,debug = fusion.step(self.z,self.rgb,self.flow,self.flow)
                np.testing.assert_allclose(out,self.z)
                self.assertTrue(np.isfinite(debug['weight']).all())

    def test_inverse_ema_hand_calculation(self):
        fusion = TemporalFusion(FusionConfig(ablation=1,alpha=.5))
        fusion.step(self.z,self.rgb)
        out,_ = fusion.step(self.z*2,self.rgb)
        np.testing.assert_allclose(out,8/3)

    def test_current_invalid_preserved(self):
        fusion = TemporalFusion()
        fusion.step(self.z,self.rgb)
        z = self.z.copy()
        z[5,5] = np.nan
        out,_ = fusion.step(z,self.rgb,self.flow,self.flow)
        self.assertTrue(np.isnan(out[5,5]))

    def test_occlusion_does_not_copy_history(self):
        fusion = TemporalFusion(FusionConfig(ablation=3))
        fusion.step(self.z,self.rgb)
        bad_forward = self.flow+30
        out,debug = fusion.step(self.z*2,self.rgb,bad_forward,self.flow)
        np.testing.assert_array_equal(out,self.z*2)
        self.assertEqual(debug['weight'].sum(),0)

    def test_scene_cut_resets(self):
        fusion = TemporalFusion()
        fusion.step(self.z,np.zeros_like(self.rgb))
        out,debug = fusion.step(self.z*2,np.full_like(self.rgb,255),self.flow,self.flow)
        self.assertTrue(debug['reset'])
        self.assertEqual(fusion.resets,1)
        np.testing.assert_array_equal(out,self.z*2)

    def test_scale_anchor_rescales_historical_inverse_correctly(self):
        no_anchor = TemporalFusion(FusionConfig(ablation=5))
        full = TemporalFusion(FusionConfig(ablation=6))
        for f in (no_anchor,full):
            f.step(self.z,self.rgb)
        a,_ = no_anchor.step(self.z*1.1,self.rgb,self.flow,self.flow)
        b,debug = full.step(self.z*1.1,self.rgb,self.flow,self.flow)
        self.assertGreater(debug['scale'],1)
        self.assertLess(np.mean(np.abs(b-2.2)),np.mean(np.abs(a-2.2)))

    def test_edge_protection(self):
        fusion = TemporalFusion(FusionConfig(ablation=5))
        fusion.step(self.z,self.rgb)
        current = self.z.copy()
        current[:,16:] = 2.3
        out,debug = fusion.step(current,self.rgb,self.flow,self.flow)
        self.assertTrue(debug['edge'][:,15:18].all())
        np.testing.assert_array_equal(out[:,15:18],current[:,15:18])

    def test_subpixel_warp_in_inverse_domain(self):
        fusion = TemporalFusion(FusionConfig(ablation=2,alpha=1.))
        first = self.z.copy()
        first[:,::2] = 4
        fusion.step(first,self.rgb)
        backward = self.flow.copy()
        backward[...,0] = .5
        out,_ = fusion.step(self.z,self.rgb,self.flow,backward)
        # Linear inverse-depth interpolation: 1 / ((1/2+1/4)/2).
        np.testing.assert_allclose(out[:,:-1],8/3)

    def test_nonfinite_flow_cannot_poison_output(self):
        fusion = TemporalFusion(FusionConfig(ablation=4))
        fusion.step(self.z,self.rgb)
        bad = self.flow.copy()
        bad[5,5] = np.nan
        out,debug = fusion.step(self.z,self.rgb,bad,bad)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(debug['weight'][5,5],0)


class IOTests(unittest.TestCase):
    def test_schema_and_frozen_output(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'run'
            meta = dict(depth_type='metric',unit='meter',representation='depth')
            writer = RunWriter(path,meta)
            writer.add(np.ones((8,8)),np.zeros((8,8,3),np.uint8),'synthetic',0,{'sample_id':'one'})
            writer.finish({'frames':1})
            metadata,records = load_run(path)
            self.assertEqual(metadata['alignment'],'none')
            self.assertEqual(len(records[0]['depth_sha256']),64)
            self.assertEqual(np.load(path/'depth/000000.npy').dtype,np.float32)
            with self.assertRaises(FileExistsError):
                RunWriter(path,meta)

    def test_noninteger_fps_sampling(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'video.avi'
            writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30,(32,24))
            self.assertTrue(writer.isOpened())
            for i in range(30):
                writer.write(np.full((24,32,3),i,np.uint8))
            writer.release()
            frames,indices,fps = read_frames(path,target_fps=12)
            self.assertEqual(len(frames),12)
            self.assertEqual(fps,12)
            self.assertEqual(indices[:4],[0,3,5,8])


if __name__ == '__main__':
    unittest.main()
