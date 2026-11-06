"""相同GT支持下的边缘定位、剖面宽度和纹理伪边缘代理诊断。"""
import cv2
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, binary_dilation
from src.analysis.scale_boundary import (annotation_edges, choose_profiles, measure_profile,
                                         depth_edge_map, edge_comparison, error_statistics)


def texture_masks(gt,rgb,valid,gt_edges,config):
    radius=config['radius'];size=2*radius+1
    safe=np.where(valid,gt,1.).astype(np.float64)
    log=np.log(safe)
    flat=(maximum_filter(log,size=size,mode='nearest')-minimum_filter(log,size=size,mode='nearest')
          <=np.log(config['maximum_gt_ratio']))
    flat &= minimum_filter(valid.astype(np.uint8),size=size,mode='constant',cval=0)>0
    flat &= ~binary_dilation(gt_edges,structure=np.ones((size,size),bool))
    color=rgb.astype(np.float64)/255
    gx=cv2.Sobel(color,cv2.CV_64F,1,0,ksize=3,borderType=cv2.BORDER_REPLICATE)/8
    gy=cv2.Sobel(color,cv2.CV_64F,0,1,ksize=3,borderType=cv2.BORDER_REPLICATE)/8
    gradient=np.sqrt(gx*gx+gy*gy).max(axis=-1)
    return {'textured_flat':flat&(gradient>=config['minimum_rgb_gradient']),
            'untextured_flat':flat&(gradient<=config['maximum_control_gradient'])}


def prepare(gt,rgb,labels,instances,diagnostic,texture):
    valid=np.isfinite(gt)&(gt>.001)&(gt<10)
    semantic=annotation_edges(labels,instances,valid)
    support=maximum_filter(semantic,size=2*diagnostic['boundary']['object_annotation_support_px']+1)>0
    maps={ratio:depth_edge_map(gt,valid_mask=valid,log_threshold=np.log(ratio))
          for ratio in diagnostic['boundary']['ratios']}
    primary=diagnostic['boundary']['primary_ratio']
    return dict(gt=gt,valid=valid,support=support,gt_maps=maps,
                profiles=choose_profiles(gt,valid,support,diagnostic['profiles']),
                texture_masks=texture_masks(gt,rgb,valid,maps[primary],texture))


def analyze(depth,prepared,diagnostic,texture):
    gt,valid=prepared['gt'],prepared['valid']
    boundary=[];localization=[];profiles=[];textures=[];primary=None
    for ratio,edges in prepared['gt_maps'].items():
        pred=depth_edge_map(depth,valid_mask=valid,log_threshold=np.log(ratio))
        rows,distances=edge_comparison(pred,edges,valid,edges&prepared['support'],
            diagnostic['boundary']['tolerance_pixels'],diagnostic['boundary']['distance_cap_px'])
        boundary.extend(dict(ratio=ratio,**r) for r in rows)
        localization.append(dict(ratio=ratio,**distances))
        if ratio==diagnostic['boundary']['primary_ratio']:
            first=next(r for r in rows if r['tolerance_px']==1)
            primary=dict(primary_f1=first['f1'],primary_precision=first['precision'],primary_recall=first['recall'],
                         gt_edge_distance_capped_px=distances['gt_edge_distance_capped_px'],
                         prediction_edge_spurious_beyond_3px_fraction=distances['prediction_edge_spurious_beyond_3px_fraction'])
    pred=depth_edge_map(depth,valid_mask=valid,log_threshold=np.log(texture['prediction_edge_ratio']))
    for region,mask in prepared['texture_masks'].items():
        count=int(mask.sum());edges=int((pred&mask).sum())
        textures.append(dict(region=region,mask_pixels=count,predicted_edge_pixels=edges,
            spurious_edge_fraction=edges/count if count else None,**error_statistics(depth,gt,mask)))
    for number,profile in enumerate(prepared['profiles']):
        profiles.append(dict(profile_id=number,**profile,
                             **measure_profile(depth,gt,profile,diagnostic['profiles'])))
    primary.update(texture_spurious_fraction=textures[0]['spurious_edge_fraction'],
                   control_spurious_fraction=textures[1]['spurious_edge_fraction'],
                   selected_profiles=len(profiles),measurable_profiles=sum(r['status']=='measurable' for r in profiles))
    return primary,dict(boundary=boundary,localization=localization,profiles=profiles,texture=textures)
