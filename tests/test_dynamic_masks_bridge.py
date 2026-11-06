"""Input/color/flow contracts for the additive dynamic-study bridge."""
import unittest
import numpy as np
from src.video.dynamic_compat import DepthModel,warp,consistency,rgb,FarnebackFlow

class BridgeTests(unittest.TestCase):
    def test_bgr_is_explicitly_converted(self):
        bgr=np.array([[[7,19,251]]],np.uint8)
        np.testing.assert_array_equal(rgb(bgr),[[[251,19,7]]])
        self.assertTrue(rgb(bgr).flags.c_contiguous)
    def test_frame_and_video_adapters_share_rgb_contract(self):
        class Native:
            def infer(self,value,*args):return value
        bridge=DepthModel.__new__(DepthModel);bridge.adapter=Native()
        bgr=np.zeros((4,6,3),np.uint8);bgr[:,:,0]=7;bgr[:,:,2]=251
        np.testing.assert_array_equal(bridge.image(bgr)[0],rgb(bgr))
        np.testing.assert_array_equal(bridge.video([bgr,bgr],1),np.stack([rgb(bgr),rgb(bgr)]))
    def test_depth_pro_tuple_preserves_metric_array(self):
        class Native:
            def infer(self,value):return np.full(value.shape[:2],3.,np.float32),100.,.1,128
        bridge=DepthModel.__new__(DepthModel);bridge.adapter=Native()
        z,_=bridge.image(np.zeros((4,6,3),np.uint8));np.testing.assert_array_equal(z,np.full((4,6),3.))
    def test_backward_warp_identity_and_outside_mask(self):
        z=np.arange(24,dtype=np.float32).reshape(4,6);back=np.zeros((4,6,2),np.float32)
        value,valid=warp(z,back);np.testing.assert_array_equal(value,z);self.assertTrue(valid.all())
        back[:,:,0]=1;value,valid=warp(z,back)
        np.testing.assert_array_equal(value[:,:-1],z[:,1:]);self.assertFalse(valid[:,-1].any())
    def test_consistency_directions(self):
        forward=np.zeros((4,6,2),np.float32);backward=forward.copy()
        forward[:,:,0]=1;backward[:,:,0]=-1
        mask,error=consistency(forward,backward,1)
        self.assertTrue(mask[:,1:].all());self.assertFalse(mask[:,0].any());self.assertTrue((error[:,1:]==0).all())
    def test_frozen_flow_is_bidirectional_and_finite(self):
        im=np.zeros((24,32,3),np.uint8)
        f,b=FarnebackFlow()(im,im)
        self.assertEqual(f.shape,(24,32,2));self.assertTrue(np.isfinite(f).all());np.testing.assert_allclose(f,0);np.testing.assert_allclose(b,0)

if __name__=='__main__':unittest.main()
