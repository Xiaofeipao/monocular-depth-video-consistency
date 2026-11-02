"""Rigid-scene pose-compensated consistency on GT-defined visible pixels."""
import numpy as np


def pose_matrix(pose):
    pose=np.asarray(pose,float)
    if pose.shape!=(7,) or not np.isfinite(pose).all():
        raise ValueError('Pose must be tx ty tz qx qy qz qw')
    q=pose[3:]
    norm=np.linalg.norm(q)
    if norm<1e-8: raise ValueError('Zero quaternion')
    x,y,z,w=q/norm
    matrix=np.eye(4)
    matrix[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    matrix[:3,3]=pose[:3]
    return matrix


def pose_correspondences(gt_previous,gt_current,pose_previous,pose_current,intrinsics,
                         absolute_tolerance=.03,relative_tolerance=.03):
    """Cache GT-only visibility once, for exactly the same mask across methods."""
    if gt_previous.shape!=gt_current.shape: raise ValueError('GT grids differ')
    if min(intrinsics['fx'],intrinsics['fy'])<=0: raise ValueError('Invalid intrinsics')
    h,w=gt_previous.shape
    y,x=np.indices((h,w))
    rays=np.stack(((x-intrinsics['cx'])/intrinsics['fx'],(y-intrinsics['cy'])/intrinsics['fy'],np.ones((h,w))),axis=-1).reshape(-1,3)
    source=np.flatnonzero(np.isfinite(gt_previous.ravel())&(gt_previous.ravel()>0))
    transform=np.linalg.inv(pose_matrix(pose_current))@pose_matrix(pose_previous)
    xyz=(rays[source]*gt_previous.ravel()[source,None])@transform[:3,:3].T+transform[:3,3]
    front=np.isfinite(xyz).all(axis=1)&(xyz[:,2]>0)
    xyz,source=xyz[front],source[front]
    u=np.rint(xyz[:,0]*intrinsics['fx']/xyz[:,2]+intrinsics['cx']).astype(int)
    v=np.rint(xyz[:,1]*intrinsics['fy']/xyz[:,2]+intrinsics['cy']).astype(int)
    inside=(u>=0)&(u<w)&(v>=0)&(v<h)
    u,v,xyz,source=u[inside],v[inside],xyz[inside],source[inside]
    pixel=v*w+u
    # The nearest GT surface wins. Correspondences never depend on a method's
    # predicted depth, avoiding method-dependent projection masks.
    order=np.argsort(xyz[:,2],kind='stable')
    _,first=np.unique(pixel[order],return_index=True)
    selected=order[first]
    source,pixel,xyz=source[selected],pixel[selected],xyz[selected]
    gt_target=gt_current.ravel()[pixel]
    visible=np.isfinite(gt_target)&(gt_target>0)&(np.abs(xyz[:,2]-gt_target)<=absolute_tolerance+relative_tolerance*gt_target)
    source,pixel,xyz=source[visible],pixel[visible],xyz[visible]
    return dict(source=source,pixel=pixel,coefficient=rays[source]@transform[2,:3],offset=transform[2,3],
                gt_reprojection_MAE_m=float(np.mean(abs(xyz[:,2]-gt_current.ravel()[pixel]))) if len(source) else None,
                coverage=len(pixel)/(h*w),visible_pixels=len(pixel))


def score_pose_correspondences(previous,current,correspondences):
    """Evaluate positive metric predictions on precomputed GT-only visibility."""
    c=correspondences
    source,pixel=c['source'],c['pixel']
    if len(source)==0:
        return {'pose_temporal_log_error':None,'pose_temporal_MAE_m':None,'gt_reprojection_MAE_m':None,'coverage':0.,'visible_pixels':0}
    projected_prediction=c['coefficient']*previous.ravel()[source]+c['offset']
    target_prediction=current.ravel()[pixel]
    if not (np.isfinite(projected_prediction)&np.isfinite(target_prediction)&(projected_prediction>0)&(target_prediction>0)).all():
        raise ValueError('Invalid prediction on common GT-defined visible pixels')
    return {'pose_temporal_log_error':float(np.mean(abs(np.log(projected_prediction)-np.log(target_prediction)))),
            'pose_temporal_MAE_m':float(np.mean(abs(projected_prediction-target_prediction))),
            'gt_reprojection_MAE_m':c['gt_reprojection_MAE_m'],
            'coverage':c['coverage'],'visible_pixels':c['visible_pixels']}


def pose_temporal_metrics(previous,current,gt_previous,gt_current,pose_previous,pose_current,intrinsics,
                          absolute_tolerance=.03,relative_tolerance=.03):
    if previous.shape!=current.shape or previous.shape!=gt_previous.shape or previous.shape!=gt_current.shape:
        raise ValueError('Pose evaluation requires registered equal-size depth grids')
    return score_pose_correspondences(previous,current,pose_correspondences(
        gt_previous,gt_current,pose_previous,pose_current,intrinsics,absolute_tolerance,relative_tolerance))
