"""按解码索引保存连续原帧；不按FPS网格选帧，避免PTS小数舍入跳帧。"""
from pathlib import Path
import cv2
from src.video.frames import _probe_video, SCHEMA_VERSION, validate_manifest
from src.video.common import write_json,sha256_file,utc_now


def prepare(source,output, *, requested_frames=100,max_long_edge=640,fps,source_url):
    source,output=Path(source).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);(output/'rgb').mkdir()
    probe=_probe_video(source);pts=probe['timestamps'];count=min(requested_frames,len(pts))
    cap=cv2.VideoCapture(str(source));frames=[];mp=output/'manifest.json'
    manifest={'schema_version':SCHEMA_VERSION,'complete':False,'clip_id':'davis_rollercoaster','source_kind':'public',
              'camera_motion':'moving','split':'course_qualitative','fps':fps,'roi_path':None,'roi_sha256':None,
              'source':{'path':str(source),'sha256':sha256_file(source),'frame_count':len(pts),
                        'first_pts_seconds':float(pts[0]),'last_pts_seconds':float(pts[-1]),'stream':probe['stream'],
                        'provenance':{'source_url':source_url,'role':'course_provided_clip'}},
              'selection':{'rule':'consecutive_original_decoded_frames; no_grid_resampling','target_fps':fps,
                           'requested_frames':requested_frames,'selected_source_indices':list(range(count)),
                           'selected_timestamps_seconds':[float(t-pts[0]) for t in pts[:count]],'no_upsampling':True},
              'resize':{'max_long_edge':max_long_edge,'keep_aspect_ratio':True,'interpolation':'opencv_INTER_AREA_downsample'},
              'code_sha256':sha256_file(__file__),'created_at_utc':utc_now(),'frames':frames}
    write_json(mp,manifest)
    try:
        if not cap.isOpened():raise ValueError('无法解码指定视频')
        shape=None
        for i in range(count):
            ok,bgr=cap.read()
            if not ok:raise ValueError(f'视频提前结束：{i}/{count}')
            if shape is None:
                shape=bgr.shape[:2];factor=min(1,max_long_edge/max(shape))
                h,w=(int(round(s*factor)) for s in shape)
                manifest.update(height=h,width=w,decoded_source_hw=list(shape))
            if bgr.shape[:2]!=shape:raise ValueError('原视频中途改变分辨率')
            if bgr.shape[:2]!=(h,w):bgr=cv2.resize(bgr,(w,h),interpolation=cv2.INTER_AREA)
            relative=f'rgb/{i:06d}.png'
            if not cv2.imwrite(str(output/relative),bgr):raise IOError('保存RGB失败')
            frames.append({'index':i,'source_index':i,'timestamp_seconds':float(pts[i]-pts[0]),
                           'source_timestamp_seconds':float(pts[i]),'rgb_path':relative,'rgb_sha256':sha256_file(output/relative)})
        manifest['complete']=True;write_json(mp,manifest);validate_manifest(mp)
        return mp
    except BaseException as exc:
        manifest.update(complete=False,error={'type':type(exc).__name__,'message':str(exc)})
        write_json(mp,manifest);raise
    finally:cap.release()
