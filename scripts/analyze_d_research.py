"""Audit compact predictions and assess boundary/new-surface failure regions."""
import json
import csv
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.io import sha256,load_run,load_depth,read_bgr,write_json,write_csv,environment
from src.metrics.pose import pose_correspondences,score_pose_correspondences
from src.flow import warp,consistency


def project_path(value):
    path=(ROOT/str(value).replace('\\','/')).resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Export points outside project')
    return path


def load_predictions(folder,summary):
    if sha256(folder/'predictions.npz')!=summary['predictions_sha256']: raise ValueError('NPZ changed')
    data=np.load(folder/'predictions.npz'); arrays={}; cache={}
    for method,source in summary['prediction_locations'].items():
        key=json.dumps(source,sort_keys=True)
        if key in cache: arrays[method]=cache[key]; continue
        if 'npz_key' in source: z=data[source['npz_key']]
        else:
            run=project_path(source['run'])
            if sha256(run/'manifest.json')!=source['manifest_sha256']: raise ValueError('Referenced manifest changed')
            _,records=load_run(run); values=[]
            for r in records:
                if sha256(run/'rgb'/f'{r["id"]}.png')!=r['rgb_sha256']: raise ValueError('Referenced RGB changed')
                p=run/'depth'/f'{r["id"]}.npy'
                if sha256(p)!=r['depth_sha256']: raise ValueError('Referenced depth changed')
                values.append(np.load(p))
            z=np.stack(values)
        if z.dtype!=np.float32 or z.shape[0]!=summary['frames'] or not np.isfinite(z).all() or (z<=0).any(): raise ValueError('Invalid saved predictions')
        arrays[method]=z; cache[key]=z
    data.close()
    return arrays


def region_analysis(folder,summary,arrays):
    baseline=project_path(summary['baseline']); _,records=load_run(baseline)
    frames=[read_bgr(baseline/'rgb'/f'{r["id"]}.png') for r in records]
    gt=None; rows=[]; camera_rows=[]
    if summary['kind']!='phone':
        source=(ROOT/'outputs/extended_video_validation_gpu518/sample_inputs'/summary['name']/'manifest.json') if summary['kind']=='reference' else ROOT/'data/public_samples/d_research'/summary['name']/'manifest.json'
        data=json.loads(source.read_text(encoding='utf-8'))
        k=json.loads((source.parent/data['intrinsics']).read_text(encoding='utf-8'))
        gt=[]
        for s in data['samples']:
            if sha256(source.parent/s['depth'])!=s['depth_sha256']: raise ValueError('GT changed')
            if sha256(source.parent/s['rgb'])!=s['rgb_sha256']: raise ValueError('Input RGB changed')
            z=load_depth(source.parent/s['depth'],s.get('depth_scale',1))
            mask=load_depth(source.parent/s['mask'])>0
            if not np.array_equal(mask,np.isfinite(z)&(z>0)): raise ValueError('TUM validity mask changed')
            mask &= np.isfinite(z)&(z>=data['protocol']['min_depth'])&(z<=data['protocol']['max_depth'])
            gt.append(np.where(mask,z,np.nan))
    for i in range(len(frames)):
        masks={}
        if gt is not None:
            valid=np.isfinite(gt[i]); logz=np.log(np.where(valid,gt[i],1))
            edge=np.zeros(valid.shape,np.uint8)
            dx=(abs(np.diff(logz,axis=1))>.08)&valid[:,1:]&valid[:,:-1]
            dy=(abs(np.diff(logz,axis=0))>.08)&valid[1:]&valid[:-1]
            edge[:,1:]|=dx; edge[:,:-1]|=dx; edge[1:]|=dy; edge[:-1]|=dy
            band=cv2.dilate(edge,np.ones((5,5),np.uint8)).astype(bool)&valid
            masks['GT_boundary_band_2px']=band
            masks['GT_interior']=valid&~band
            if i:
                c=pose_correspondences(gt[i-1],gt[i],data['samples'][i-1]['pose_world_camera'],data['samples'][i]['pose_world_camera'],k)
                visible=np.zeros(valid.shape,bool); visible.ravel()[c['pixel']]=True
                masks['GT_not_rigidly_matched']=valid&~visible
                if data.get('sequence','').startswith('freiburg3'):
                    # Exploratory evaluation-only sensitivity to the official
                    # zero-distortion fr3 RGB calibration; never fed to VDA.
                    h,w=gt[i].shape
                    calibrated=dict(fx=535.4*w/640,fy=539.2*h/480,cx=(320.1+.5)*w/640-.5,cy=(247.6+.5)*h/480-.5)
                    c2=pose_correspondences(gt[i-1],gt[i],data['samples'][i-1]['pose_world_camera'],data['samples'][i]['pose_world_camera'],calibrated)
                    for method,zs in arrays.items():
                        camera_rows.append(dict(clip=summary['name'],method=method,current=i,camera='official_fr3_calibrated_RGB',**score_pose_correspondences(zs[i-1],zs[i],c2)))
        if i:
            cache=(ROOT/'outputs/extended_video_validation_gpu518'/summary['name']/'flow_raft') if summary['kind'] in ('phone','reference') else folder/'flow_raft'
            with np.load(cache/f'{i-1:06d}_{i:06d}.npz') as d: f,b=d['forward'],d['backward']
            common,_=consistency(f,b); old,_=warp(frames[i-1],b)
            photo=np.mean(abs(old-frames[i].astype(np.float32)),axis=-1)/255
            masks['RAFT_photometric_mismatch']=common&(photo>.12)
        for region,mask in masks.items():
            if not mask.any(): continue
            for method,zs in arrays.items():
                row=dict(clip=summary['name'],kind=summary['kind'],method=method,frame=i,region=region,pixels=int(mask.sum()))
                row['RAW_reference_log_change']=float(np.mean(abs(np.log(zs[i][mask])-np.log(arrays['RAW'][i][mask]))))
                row['GT_AbsRel']=float(np.mean(abs(zs[i][mask]-gt[i][mask])/gt[i][mask])) if gt is not None and region.startswith('GT_') else None
                rows.append(row)
    if rows:
        write_csv(folder/'regions.csv',rows)
    else:
        # A genuinely static phone excerpt can have no pixels exceeding the
        # photometric mismatch threshold. Preserve an empty, typed table.
        with (folder/'regions.csv').open('w',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=['clip','kind','method','frame','region','pixels','RAW_reference_log_change','GT_AbsRel']).writeheader()
    write_json(folder/'region_status.json',dict(scored_rows=len(rows),has_GT=gt is not None,
        empty_reason='No valid pixels in the requested diagnostic regions; not a zero-error result' if not rows else None))
    if camera_rows: write_csv(folder/'camera_sensitivity.csv',camera_rows)
    if summary['kind']=='phone': montage(folder,frames,arrays)
    return rows


def montage(folder,frames,arrays):
    from src.visualization.video import colorize
    # Select the temporal window from RGB differences only, not depth scores.
    changes=[float(np.mean(abs(b.astype(float)-a.astype(float)))) for a,b in zip(frames,frames[1:])]
    center=int(np.clip(np.argmax(changes)+1,2,len(frames)-3)); indices=list(range(center-2,center+3))
    h,w=frames[0].shape[:2]
    # Fixed central upper region includes person and chair bars in the phone clips.
    roi=(int(h*.02),int(h*.72),int(w*.05),int(w*.9))
    methods=['RGB','RAW','D_temporal','D']+(['D_tuned'] if 'D_tuned' in arrays else [])+['E_half']
    strips=[]
    for method in methods:
        tiles=[]
        for i in indices:
            img=frames[i] if method=='RGB' else colorize(arrays[method][i],.1,10)
            img=img[roi[0]:roi[1],roi[2]:roi[3]]
            factor=180/max(img.shape[:2]); img=cv2.resize(img,(round(img.shape[1]*factor),round(img.shape[0]*factor)))
            tile=np.zeros((img.shape[0]+24,img.shape[1],3),np.uint8); tile[24:]=img
            cv2.putText(tile,f'{method} f{i}',(3,17),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1); tiles.append(tile)
        strips.append(np.hstack(tiles))
    cv2.imwrite(str(folder/'motion_review.jpg'),np.vstack(strips))
    write_json(folder/'motion_review_selection.json',dict(indices=indices,roi_top_bottom_left_right=roi,selection='Maximum RGB difference; fixed normalized upper-central ROI; no GT for phone'))


def main():
    root=ROOT/'outputs/d_research'; evidence=ROOT/'docs/evidence/d_research'
    study=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if len(study['cases'])!=9: raise ValueError('Need all 3 heldout + 4 phone + 2 reference cases')
    expected=study['protocol']['heldout']+study['protocol']['phone_demonstration']+study['protocol']['reference_only']
    if {c['name'] for c in study['cases']}!=set(expected): raise ValueError('Wrong case identities')
    if set(study['protocol']['development'])&set(study['protocol']['heldout']): raise ValueError('Development/test overlap')
    if study['selection']['protocol_sha256']!=sha256(root/'protocol.json'): raise ValueError('Protocol changed')
    if study['selection']['development_summary_sha256']!=sha256(root/study['selection']['development']/'summary.json'):
        raise ValueError('Development scores changed after selection')
    dev=json.loads((root/study['selection']['development']/'summary.json').read_text(encoding='utf-8'))
    all_cases=[dev]+study['cases']; all_regions=[]; hashes=[]; counts=dict(clips=0,RGB_frames=0,logical_predictions=0,videos=0,flow_pairs=0)
    lines=['# D 多序列研究与消融','',
        '新增四条 TUM 序列：desk 为开发组，rpy、sitting_static、walking_xyz 为冻结测试组，各 64 帧。',
        '另检查四段自录视频与已有 xyz 两段参考片段。既有参考不是新测试集，不参与参数选择。',
        '这些是少量序列，fr1/fr3 内也可能共享物理场景；不报告帧级显著性或跨域最优。','',
        'VDA-Small / 518 / FP32 / RGB 长边 448，固定 5 FPS 抽样。原生米制，不拟合测试 GT 尺度。',
        '相机使用 TUM 建议的注册 PNG/ROS 默认约定，深度除以 5000，不重复应用深度校正。',
        '来源：[TUM 下载与序列描述](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)、',
        '[TUM 格式和相机约定](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)。','',
        '## 消融定义与参数冻结','',
        '- D_temporal：固定历史权重，关闭置信度与全局重置；保留相同的光流、光度和创新门限。',
        '- D_confidence：加入局部 log-depth 残差方差权重，仍关闭全局重置。',
        '- D：原始完整方法，加入全局场景切换重置；默认数值已与旧版逐值对照。',
        '- D_tuned：仅在 desk 开发组选择 alpha 上限、边缘保护与低覆盖重置；随后冻结。',
        '- A、E_half、C 和 RAW 均保留；C 用于检查边界反例。','',
        f'开发组选择：`{study["selection"]["selected"]}`，参数 `{json.dumps(study["selection"]["params"],ensure_ascii=False)}`。',
        '选择约束：AbsRel ≤ RAW+0.01、δ1 ≥ RAW−0.005、边界 F1 ≥ RAW−0.01；合格项中最小 pose-log-error。',
        'selection.json 在任何新测试组推理前保存；未根据测试结果重新选择参数。','',
        '## 新测试组结果','']
    fresh=[c for c in study['cases'] if c['kind']=='heldout']
    headline=['## 主要结论','']
    for method in ('D_temporal','D','E_half','C'):
        rows=[next(r for r in c['methods'] if r['method']==method) for c in fresh]
        headline.append(f'- {method}：通过预设精度/边界约束 {sum(r["guard_passed"] for r in rows)}/{len(rows)}。')
    same=sum(c['prediction_locations'].get('D_tuned')==c['prediction_locations']['D'] for c in fresh) if not study['selection']['fallback_RAW'] else 0
    headline+= [f'- 冻结优化版与原 D 在 {same}/{len(fresh)} 条新测试序列预测完全相同，本轮未获得新增优化收益。',
               '- 没有一种方法在每段的时序、边界和速度上同时最好；不据此宣称超过基础模型论文。','']
    lines[2:2]=headline
    for case in all_cases:
        folder=root/case['name']; arrays=load_predictions(folder,case)
        image_rows=list(csv.DictReader((folder/'per_image.csv').open(encoding='utf-8')))
        pair_rows=list(csv.DictReader((folder/'per_pair.csv').open(encoding='utf-8')))
        if len(image_rows)!=case['frames']*len(arrays) or len(pair_rows)!=(case['frames']-1)*2*len(arrays):
            raise ValueError('Incomplete per-frame or per-pair table')
        for method in arrays:
            report=next(r for r in case['methods'] if r['method']==method)
            for key in ('AbsRel','RMSE','delta1','edge_F1','RAW_reference_edge_F1','RAW_log_change'):
                if report[key] is None: continue
                mean=float(np.mean([float(r[key]) for r in image_rows if r['method']==method]))
                if abs(mean-report[key])>1e-12: raise ValueError('Image means differ from CSV')
            for backend in ('farneback','raft'):
                mean=float(np.mean([float(r['warp_proxy']) for r in pair_rows if r['method']==method and r['backend']==backend]))
                if abs(mean-report[backend+'_warp_proxy'])>1e-12: raise ValueError('Flow means differ from CSV')
            if report['pose_temporal_log_error'] is not None:
                mean=float(np.mean([float(r['pose_temporal_log_error']) for r in pair_rows if r['method']==method and r.get('pose_temporal_log_error')]))
                if abs(mean-report['pose_temporal_log_error'])>1e-12: raise ValueError('Pose means differ from CSV')
        all_regions+=region_analysis(folder,case,arrays)
        for file in ('summary.json','predictions.npz','per_image.csv','per_pair.csv','regions.csv','diagnostics.json'):
            hashes.append(dict(path=(folder/file).relative_to(ROOT).as_posix(),sha256=sha256(folder/file)))
        for backend in ('farneback','raft'):
            cache=(ROOT/'outputs/extended_video_validation_gpu518'/case['name']/f'flow_{backend}') if case['kind'] in ('phone','reference') else folder/f'flow_{backend}'
            for i in range(1,case['frames']):
                file=cache/f'{i-1:06d}_{i:06d}.npz'
                with np.load(file) as d:
                    mask,_=consistency(d['forward'],d['backward'])
                    if not np.array_equal(mask,d['common_mask']): raise ValueError('Cached common mask differs')
                selected=[r for r in pair_rows if r['backend']==backend and int(r['current'])==i]
                if len(selected)!=len(arrays) or any(int(r['valid_pixels'])!=int(mask.sum()) for r in selected):
                    raise ValueError('Methods do not share the same flow mask')
                for r in selected:
                    # The local raw export inherited pose's "coverage" during
                    # dict merging. Give both coverages explicit names here.
                    r['flow_coverage']=float(mask.mean())
                    r['pose_coverage']=float(r['coverage']) if r.get('pose_temporal_log_error') else None
                    del r['coverage']
                hashes.append(dict(path=file.relative_to(ROOT).as_posix(),sha256=sha256(file)))
                counts['flow_pairs']+=1
        cap=cv2.VideoCapture(str(folder/'comparison.mp4')); decoded=0
        while cap.read()[0]: decoded+=1
        cap.release()
        if decoded!=case['frames']: raise ValueError('Incomplete comparison video')
        counts['clips']+=1; counts['videos']+=1; counts['RGB_frames']+=case['frames']; counts['logical_predictions']+=case['frames']*len(arrays)
        dest=evidence/case['name']; dest.mkdir(exist_ok=True)
        baseline=project_path(case['baseline'])
        for name in ('metadata.json','manifest.json','config_resolved.json','timing.json'):
            (dest/('baseline_'+name)).write_bytes((baseline/name).read_bytes())
        env=json.loads((baseline/'environment.json').read_text(encoding='utf-8'))
        write_json(dest/'software_environment.json',{k:env[k] for k in ('python','packages')})
        config=json.loads((baseline/'config_resolved.json').read_text(encoding='utf-8'))
        source=project_path(config['input'])
        (dest/'input_manifest.json').write_bytes(source.read_bytes())
        input_data=json.loads(source.read_text(encoding='utf-8'))
        if input_data.get('source_sha256'):
            if sha256(project_path(input_data['source']))!=input_data['source_sha256']: raise ValueError('Phone capture changed')
        if input_data.get('intrinsics'):
            (dest/'intrinsics.json').write_bytes((source.parent/input_data['intrinsics']).read_bytes())
        receipt=ROOT/'data/public_samples/d_research'/f'{case["name"]}_receipt.json'
        if receipt.exists(): (dest/'download_receipt.json').write_bytes(receipt.read_bytes())
        write_csv(dest/'per_pair.csv',pair_rows)
        hashes.append(dict(path=(dest/'per_pair.csv').relative_to(ROOT).as_posix(),sha256=sha256(dest/'per_pair.csv')))
        (dest/'regions.csv').write_bytes((folder/'regions.csv').read_bytes())
        if (folder/'camera_sensitivity.csv').exists():
            (dest/'camera_sensitivity.csv').write_bytes((folder/'camera_sensitivity.csv').read_bytes())
        if case['kind']=='heldout':
            raw=next(r for r in case['methods'] if r['method']=='RAW')
            lines += [f'### {case["name"]}','',f'RAW：AbsRel {raw["AbsRel"]:.6f}，δ1 {raw["delta1"]*100:.3f}%，edge F1 {raw["edge_F1"]:.6f}，pose-log-error {raw["pose_temporal_log_error"]:.6f}。','']
            for r in case['methods']:
                if r['method']=='RAW': continue
                gain=100*(1-r['pose_temporal_log_error']/raw['pose_temporal_log_error'])
                lines.append(f'- {r["method"]}：pose 变化 {gain:+.2f}%（正值为改善），AbsRel {r["AbsRel"]:.6f}，edge F1 {r["edge_F1"]:.6f}；约束'+('通过。' if r['guard_passed'] else '未通过。'))
            lines.append('')
    lines += ['## 机制、边界与遮挡诊断','']
    heldout=[c for c in study['cases'] if c['kind']=='heldout']
    for method in ('D_temporal','D_confidence','D','D_tuned','E_half','C'):
        rows=[next((r for r in c['methods'] if r['method']==method),None) for c in heldout]
        rows=[r for r in rows if r is not None]
        lines.append(f'- {method}：测试组通过约束 {sum(r["guard_passed"] for r in rows)}/{len(rows)}；重置次数 {sum(r["resets"] for r in rows)}。')
    identical=[]
    for c in study['cases']:
        m={r['method']:r for r in c['methods']}
        if c['prediction_locations']['D']==c['prediction_locations']['D_confidence']: identical.append(c['name'])
    lines += ['',f'D 与 D_confidence 预测完全相同的片段：{", ".join(identical) or "无"}。',
        '固定权重基线为 0.4，而置信度版产生可变权重；比较也包含实际平滑强度差异，不能把全部变化归因于方差排序质量。',
        '若自然片段没有触发重置，不能以分数相同证明重置有效；场景切换与低覆盖的作用另由机制测试验证。',
        'regions.csv 报告 GT 边界两像素带、内部和未被刚性 GT 重投影匹配区域的 AbsRel。',
        '最后一种区域混合了新显露表面、动态物体、视野变化和传感器误差，不是精确运动分割。',
        '另保存 fr3 官方 RGB 内参的探索性评分敏感性检查（camera_sensitivity.csv）；只改变评测投影，不给模型额外相机信息，也不用于参数选择。',
        '光度不匹配区报告相对 RAW 的改动量，自录视频的 RAW 边界保留量也只是结构诊断，不能当作 GT 精度或无拖影证明。','',
        '## 自录视频与观感检查','']
    for c in study['cases']:
        if c['kind']!='phone': continue
        m={r['method']:r for r in c['methods']}; raw=m['RAW']
        desc=[]
        for key in ('A','D','D_tuned','E_half'):
            if key in m: desc.append(f'{key} {100*(1-m[key]["raft_warp_proxy"]/raw["raft_warp_proxy"]):.2f}%')
        lines.append(f'- {c["name"]}，RAFT proxy 降幅：'+ '；'.join(desc)+f'。[动态区域连续帧](../outputs/d_research/{c["name"]}/motion_review.jpg)，[对比视频](../outputs/d_research/{c["name"]}/comparison.mp4)。')
    lines += ['', '移动相机/人物会改变真实 z；光流 proxy 不是纯闪烁或论文 TAE。','',
        '## 计时范围','',
        'summary.json 中 fusion_seconds 不含光流；postprocess_seconds 加入重新计算光流的时间。',
        '二者均不含 VDA、评测和 IO，使用 1 次预热与 3 次重复。','']
    timing=root/'pipeline_timing/summary.json'
    if not timing.exists(): raise RuntimeError('Complete pipeline timing before publishing report')
    t=json.loads(timing.read_text(encoding='utf-8'))
    expected_methods=4 if study['selection']['fallback_RAW'] else 5
    if len(t['methods'])!=2*expected_methods: raise ValueError('Pipeline timing incomplete')
    lines += ['另外直接测量“读取保存的 RGB → 离线 VDA → CPU 光流/融合 → 写压缩深度”的暖进程管线：','']
    for r in t['methods']: lines.append(f'- {r["clip"]} / {r["method"]}：中位吞吐 {r["median_fps"]:.2f} FPS。')
    for clip in sorted({r['clip'] for r in t['methods']}):
        rates={r['method']:r['median_fps'] for r in t['methods'] if r['clip']==clip}
        lines.append(f'{clip}：E_half 的文件管线吞吐比 A 高 {100*(rates["E_half"]/rates["A"]-1):.1f}%，仍比 RAW 低 {100*(1-rates["E_half"]/rates["RAW"]):.1f}%。')
    observation=evidence/'mixed_shape_timing_observation.json'
    if observation.exists():
        obs=json.loads(observation.read_text(encoding='utf-8'))
        before=float(np.median([float(v) for _,v in obs['observed_phone_RAW_fps']]))
        after=next(r['median_fps'] for r in t['methods'] if r['clip']=='moving' and r['method']=='RAW')
        lines += ['',f'混合宽高比的初次暖进程计时出现显存占满，手机 RAW 约 {before:.2f} FPS；',
                  f'释放无用 CUDA 缓存并重测后为 {after:.2f} FPS。异常观察另存，不混入主要方法速度比较。',
                  '主计时在片段/方法边界、计时区间之外清理无用缓存。此处是运行内存管理，不是 D/E 的算法加速。']
    lines += ['', '该管线不含初次模型加载、原视频抽帧、显示、告警和评分；VDA 为整段离线推理，不能把吞吐叫作流式实时帧率。','',
        '## 复现和证据','',
        '`scripts/prepare_d_research_data.py` → `scripts/research_direction_d.py` →',
        '`scripts/benchmark_d_pipeline.py` → `scripts/analyze_d_research.py`。',
        '先运行项目缓存脚本，使用项目既有兼容环境；模型/数据前提同 518 扩展验证。',
        '所有数值、逐帧/逐对表、开发选择与校验见 `docs/evidence/d_research/`。',
        '压缩 NPY 数组无损保存；prediction_locations 明确指向 NPZ 键或经 SHA256 验证的既有冻结运行，避免重复占用空间。','']
    aggregate=[]
    keys=sorted({(r['clip'],r['method'],r['region']) for r in all_regions})
    for clip,method,region in keys:
        rows=[r for r in all_regions if (r['clip'],r['method'],r['region'])==(clip,method,region)]
        total=sum(r['pixels'] for r in rows)
        aggregate.append(dict(clip=clip,method=method,region=region,frames=len(rows),total_pixels=total,
            pixel_weighted_GT_AbsRel=sum(r['GT_AbsRel']*r['pixels'] for r in rows)/total if rows[0]['GT_AbsRel'] is not None else None,
            macro_frame_GT_AbsRel=float(np.mean([r['GT_AbsRel'] for r in rows])) if rows[0]['GT_AbsRel'] is not None else None,
            macro_RAW_reference_log_change=float(np.mean([r['RAW_reference_log_change'] for r in rows]))))
    write_csv(evidence/'region_aggregates.csv',aggregate)
    write_csv(evidence/'region_scores.csv',all_regions); write_csv(evidence/'checksums.csv',hashes)
    write_json(evidence/'validation.json',dict(all_passed=True,counts=counts,selection_sha256=sha256(root/'selection.json')))
    write_json(evidence/'evaluation_settings.json',dict(
        metric_depth=dict(alignment='none',prediction_clipping=False,GT_range_m=[.1,10],crop=None,aggregation='frame macro means then clip macro means'),
        fusion_flow=dict(name='Farneback',pyr_scale=.5,levels=3,winsize=21,iterations=3,poly_n=5,poly_sigma=1.2,flags=0),
        E_half_flow_scale=.5,
        independent_flow=dict(name='torchvision RAFT Small C_T_V2',checkpoint_sha256=sha256(ROOT/'checkpoints/raft_small_C_T_V2-01064c6d.pth'),updates=12,min_side=128,multiple_of=8),
        flow_mask=dict(forward_backward_error_max_px=1.,common_across_methods=True,source='RGB flow and finite positive identical prediction support',warp='current-to-previous bilinear sampling'),
        pose=dict(visibility='GT-only z-buffer, not method-dependent',absolute_tolerance_m=.03,relative_tolerance=.03,model_receives_camera=False),
        boundary=dict(name='log-depth jump tolerant F1, not Depth Pro SI F1',log_threshold=.08,tolerance_px=2),
        region_diagnostics=dict(GT_boundary_band_radius_px=2,RAFT_photo_mismatch_threshold=.12),
        fr3_calibration_sensitivity=dict(exploratory=True,evaluation_only=True,full_RGB=dict(fx=535.4,fy=539.2,cx=320.1,cy=247.6),resize='pixel-center-aware',used_for_parameter_selection=False)))
    sources=('scripts/research_direction_d.py','scripts/benchmark_d_pipeline.py','scripts/analyze_d_research.py',
        'scripts/prepare_d_research_data.py','scripts/run_d_research.ps1','scripts/check_d_legacy_parity.py',
        'scripts/run_extended_video_validation.py','pipelines/part3/research_d.py',
        'src/exploration/uncertainty.py','src/metrics/pose.py','configs/part3/d_research.json',
        'src/models/foundation.py','src/exploration/efficient.py','src/exploration/edge_aware.py',
        'src/flow.py','src/temporal.py','src/metrics/video_project.py','src/data/tum.py')
    write_json(evidence/'source_scripts.json',{p:sha256(ROOT/p) for p in sources})
    env=environment(); write_json(evidence/'software_environment.json',{k:env[k] for k in ('python','packages')})
    (ROOT/'docs/d_research.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(counts),flush=True)


if __name__=='__main__': main()
