"""无GT的逐帧相对输出诊断；数值用于定位案例，不代表深度准确率。"""
from __future__ import annotations
import cv2
import numpy as np
from .temporal import warp_with_backward_flow, forward_backward_consistency_mask


def reference(sequence, percentiles=(1, 99)):
    values=np.asarray(sequence)
    if values.ndim!=3 or not np.isfinite(values).all():raise ValueError('需要完整有限的TxHxW原生预测')
    low,high=map(float,np.percentile(values,percentiles))
    return {'low':low,'high':high,'span':high-low if high>low else None,
            'scope':'whole_clip_once','percentiles':list(percentiles),'per_frame_normalization':False}


def edge_map(depth, span, threshold=.02):
    if span is None:return np.zeros(np.shape(depth),bool)
    image=cv2.GaussianBlur(np.asarray(depth,np.float32)/span,(0,0),.8)
    gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)/8
    return np.hypot(gx,gy)>=threshold


def difference(a,b,span,mask=None):
    a,b=np.asarray(a),np.asarray(b)
    if a.ndim!=2 or a.shape!=b.shape:raise ValueError('深度图形状不同')
    if mask is None:
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('预测必须完整有限，不能删掉困难像素')
        mask=np.ones(a.shape,bool)
    else:
        mask=np.asarray(mask,bool)
        if mask.shape!=a.shape:raise ValueError('mask形状不同')
        if not np.isfinite(a[mask]).all() or not np.isfinite(b[mask]).all():raise ValueError('有效域数值非有限')
    if span is not None and (not np.isfinite(span) or span<=0):raise ValueError('跨度必须为正有限值或None')
    if not mask.any():return {'mae':None,'rmse':None,'mae_normalized':None,'rmse_normalized':None,'pixels':0}
    d=b[mask].astype(np.float64)-a[mask].astype(np.float64)
    mae=float(np.abs(d).mean());rmse=float(np.sqrt(np.square(d).mean()))
    return {'mae':mae,'rmse':rmse,'mae_normalized':mae/span if span else None,
            'rmse_normalized':rmse/span if span else None,'pixels':int(mask.sum())}


def pair(previous,current,rgb_previous,rgb_current,span,settings):
    result={'pixel_'+k:v for k,v in difference(previous,current,span).items()}
    result.update(flow_mae_normalized=None,flow_rmse_normalized=None,flow_coverage=None,
                  edge_instability=None,flow_median_displacement_px=None)
    if not settings['farneback']:return result
    a=cv2.cvtColor(rgb_previous,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(rgb_current,cv2.COLOR_RGB2GRAY)
    args=(None,.5,3,21,3,7,1.5,0)
    f=cv2.calcOpticalFlowFarneback(a,b,*args);back=cv2.calcOpticalFlowFarneback(b,a,*args)
    return {**result,**flow_pair(previous,current,f,back,span,settings)}


def flow_pair(previous,current,forward,backward,span,settings):
    warped,valid=warp_with_backward_flow(previous,backward)
    valid &= forward_backward_consistency_mask(forward,backward,threshold_px=settings['fb_threshold_px'])
    score=difference(warped,current,span,valid)
    old_edge=edge_map(previous,span,settings['edge_threshold']).astype(np.float32)
    old_edge,support=warp_with_backward_flow(old_edge,backward)
    old_edge=np.nan_to_num(old_edge)>=.5
    current_edge=edge_map(current,span,settings['edge_threshold'])
    r=settings['edge_tolerance_px'];kernel=np.ones((2*r+1,2*r+1),np.uint8)
    # 收缩有效域，让容差窗口不能借用遮挡/越界区域的边缘。
    domain=cv2.erode((valid&support).astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    ea,eb=old_edge&domain,current_edge&domain
    instability=None
    if span and ea.any() and eb.any():
        ma=float((ea & (cv2.dilate(eb.astype(np.uint8),kernel)>0)).sum()/ea.sum())
        mb=float((eb & (cv2.dilate(ea.astype(np.uint8),kernel)>0)).sum()/eb.sum())
        instability=1-(ma+mb)/2
    return {'flow_mae_normalized':score['mae_normalized'],'flow_rmse_normalized':score['rmse_normalized'],
            'flow_coverage':float(valid.mean()),'edge_instability':instability,
            'flow_median_displacement_px':float(np.median(np.linalg.norm(backward,axis=-1))),
            'flow_pixels':score['pixels']}


def select_cases(rows,separation=5):
    """返回三个可视检查候选，分数高不自动等于模型失败。"""
    chosen=[]
    for metric,label in [('edge_instability','boundary_change'),('flow_rmse_normalized','correspondence_change'),
                          ('pixel_rmse_normalized','screen_coordinate_change')]:
        eligible=sorted((r for r in rows if r.get(metric) is not None),key=lambda r:(-r[metric],r['current_index']))
        distant=[r for r in eligible if all(abs(r['current_index']-c['current_index'])>=separation for c in chosen)]
        pool=distant or [r for r in eligible if r['current_index'] not in {c['current_index'] for c in chosen}]
        if pool:chosen.append({'current_index':pool[0]['current_index'],'metric':metric,'value':pool[0][metric],'role':label})
    return chosen
