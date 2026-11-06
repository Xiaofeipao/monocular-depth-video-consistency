"""Dynamic-study bridge to the teammate's RGB model/flow APIs.

Inputs at this bridge are BGR (OpenCV); existing teammate APIs remain RGB.
Frozen Farneback settings are deliberately different from the general evaluator.
"""
from pathlib import Path
import hashlib
import cv2,numpy as np

def rgb(image):return np.ascontiguousarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

class DepthModel:
    def __init__(self,name,repo,checkpoint,device='cuda',input_size=518,fp16=True):
        import torch
        self.torch=torch;self.device=device;self.name=name
        repo=Path(repo)
        if not repo.is_dir():repo=repo.parent.parent/repo.name
        args=dict(repository=repo,checkpoint=Path(checkpoint),device=device,precision='float16' if fp16 else 'float32')
        if name=='dav2':
            from src.models.depth_anything_v2_adapter import DepthAnythingV2Adapter
            self.adapter=DepthAnythingV2Adapter(**args,input_size=input_size)
        elif name=='vda':
            from src.models.video_depth_anything_adapter import VideoDepthAnythingAdapter
            self.adapter=VideoDepthAnythingAdapter(**args,input_size=input_size)
        elif name=='depth-pro':
            from src.models.depth_pro_adapter import DepthProAdapter
            self.adapter=DepthProAdapter(**args)
        else:raise ValueError(name)
        self.metadata=dict(model=name,checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            precision='fp16' if fp16 else 'fp32',device='GPU-A',input_size=input_size if name!='depth-pro' else 1536,
            adapter='teammate native RGB adapter through explicit BGR conversion',depth_type='metric' if name=='depth-pro' else 'relative')
    def synchronize(self):
        if self.device.startswith('cuda'):self.torch.cuda.synchronize()
    def image(self,image):
        result=self.adapter.infer(rgb(image))
        return (result[0],{}) if isinstance(result,tuple) else (result,{})
    def video(self,images,fps):return self.adapter.infer(np.stack([rgb(im) for im in images]),fps)

def warp(source,backward):
    h,w=backward.shape[:2];y,x=np.indices((h,w),dtype=np.float32)
    mx=x+backward[...,0];my=y+backward[...,1]
    valid=np.isfinite(mx)&np.isfinite(my)&(mx>=0)&(mx<=w-1)&(my>=0)&(my<=h-1)
    value=cv2.remap(np.asarray(source,np.float32),np.nan_to_num(mx).astype(np.float32),np.nan_to_num(my).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return value,valid

def consistency(forward,backward,threshold=1.):
    sampled,valid=warp(forward,backward);error=np.linalg.norm(backward+sampled,axis=-1)
    return valid&np.isfinite(error)&(error<=threshold),error

class FarnebackFlow:
    def __call__(self,previous,current):
        a,b=(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) for x in (previous,current))
        args=(None,.5,3,21,3,5,1.2,0)
        return cv2.calcOpticalFlowFarneback(a,b,*args),cv2.calcOpticalFlowFarneback(b,a,*args)

class RAFTFlow:
    def __init__(self,device='cuda',checkpoint=None):
        from src.video.flow import FlowEstimator
        self.estimator=FlowEstimator(backend='raft_small',device=device,checkpoint=checkpoint,iterations=12)
    def __call__(self,previous,current):return self.estimator.estimate(rgb(previous),rgb(current))
