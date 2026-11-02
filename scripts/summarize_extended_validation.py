"""Validate, publish numerical evidence, and refresh an extended video report."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.io import sha256,write_json,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-root',default='outputs/extended_video_validation_cpu')
    parser.add_argument('--depth-pro-root',default='outputs/ibims_refinement_gpu/depth-pro')
    parser.add_argument('--unidepth-root',default='outputs/ibims_refinement_cpu/unidepth-v2')
    parser.add_argument('--evidence',default='docs/evidence/extended_validation_cpu')
    parser.add_argument('--report',default='docs/extended_validation.md')
    a=parser.parse_args()
    root=ROOT/a.video_root
    summary=json.loads((root/'summary.json').read_text())
    if len(summary['clips'])!=6: raise ValueError('All six video jobs must finish')
    evidence=ROOT/a.evidence; evidence.mkdir(parents=True,exist_ok=True)
    counts=dict(baseline_predictions=0,derived_predictions=0,flow_pairs=0,videos=0)
    csv_rows=[]; checksum_rows=[]
    lines=['# 扩展验证：自录视频与有真值 sample','',
           '四段自录视频各 64 帧（约 12.6 秒），TUM fr1/xyz 两段不重叠 sample 为 64/54 帧。',
           f'所有视频用同一 metric VDA-Small、{summary["settings"]["device"]} FP32、input-size {summary["settings"]["input_size"]}、RGB 长边 448、5 FPS 名义播放率，',
           '固定 RAW/A/C/D/E_half 参数；TUM 评分使用实际同步时间戳和 GT 位姿。没有训练或测试集调参。','',
           '64 帧覆盖多个官方推理窗口，但仍非完整视频或超长视频基准。四段手机视频是同一物理场景，',
           '两段 TUM 也来自同一序列，不应视为六个独立场景。所有颜色固定为 0.1–10 m。','',
           '## 自录视频：独立 RAFT 光流代理评分','',
           '以下为相对于原始 metric VDA 的 log-warp proxy 降幅，顺序 A / C / D / E_half。',
           '移动相机/人物会改变真实 z，不能把这些降幅等同纯闪烁或真实尺度漂移。','']
    gt_clips=[]
    for clip in summary['clips']:
        folder=root/clip['name']; n=clip['frames']
        manifest=ROOT/clip['input_manifest']
        if sha256(manifest)!=clip['input_manifest_sha256']: raise ValueError('Input manifest changed')
        data=json.loads(manifest.read_text())
        if len(data['samples'])!=n: raise ValueError('Input frame count mismatch')
        if data.get('source_sha256') and sha256(ROOT/data['source'])!=data['source_sha256']: raise ValueError('Phone video changed')
        baseline=folder/'baseline'; base_records=json.loads((baseline/'manifest.json').read_text())
        for r in base_records:
            p=baseline/'depth'/f'{r["id"]}.npy'
            if sha256(p)!=r['depth_sha256']: raise ValueError('Baseline changed')
            checksum_rows.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=r['depth_sha256']))
            counts['baseline_predictions']+=1
        for method in summary['settings']['methods']:
            run=folder/'directions'/method
            records=json.loads((run/'manifest.json').read_text())
            if len(records)!=n: raise ValueError('Derived frame count mismatch')
            meta=json.loads((run/'metadata.json').read_text())
            if meta['parent_manifest_sha256']!=sha256(baseline/'manifest.json'): raise ValueError('Wrong parent')
            for r,b in zip(records,base_records):
                p=run/'depth'/f'{r["id"]}.npy'; digest=sha256(p)
                if digest!=r['depth_sha256'] or r['rgb_sha256']!=b['rgb_sha256'] or sha256(run/'rgb'/f'{r["id"]}.png')!=r['rgb_sha256']: raise ValueError('Derived hash/RGB mismatch')
                z=np.load(p)
                if not np.isfinite(z).all() or (z<=0).any(): raise ValueError('Invalid prediction')
                if method=='RAW' and not np.array_equal(z,np.load(baseline/'depth'/p.name)): raise ValueError('RAW changed')
                checksum_rows.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=digest)); counts['derived_predictions']+=1
        windows={}
        for backend in ('farneback','raft'):
            rows=list(csv.DictReader((folder/f'per_pair_{backend}.csv').open()))
            if len(rows)!=(n-1)*5: raise ValueError('Missing pair scores')
            for i in range(1,n):
                p=folder/f'flow_{backend}'/f'{i-1:06d}_{i:06d}.npz'; f=np.load(p)
                coverage=float(f['common_mask'].mean())
                if not f['common_mask'].any(): raise ValueError('Empty common mask')
                selected=[r for r in rows if int(r['current_id'])==i]
                if len(selected)!=5 or any(abs(float(r['coverage'])-coverage)>1e-10 for r in selected): raise ValueError('Masks differ between methods')
                counts['flow_pairs']+=1
            for m in summary['settings']['methods']:
                values=[r for r in rows if r['method']==m]
                mean=float(np.mean([float(r['log_warp_proxy']) for r in values]))
                if abs(mean-clip['flow_scores'][backend][m])>1e-12: raise ValueError('Summary differs from CSV')
            near={i for join in (32,54) for i in range(join-8,join+1) if 1<=i<n}
            windows[backend]={m:{scope:float(np.mean([float(r['log_warp_proxy']) for r in rows if r['method']==m and ((int(r['current_id']) in near)==flag)])) for scope,flag in [('near_overlap_or_join',True),('outside_join_region',False)]} for m in summary['settings']['methods']}
        write_json(folder/'window_diagnostics.json',dict(metrics=windows,note='Near official overlap/join regions 24..32 and 46..54. Scene motion also varies across time; not a causal stitch-artifact metric.'))
        cap=cv2.VideoCapture(str(folder/'comparison.mp4')); decoded=0
        while cap.read()[0]: decoded+=1
        cap.release()
        if decoded!=n: raise ValueError('Comparison video incomplete')
        counts['videos']+=1
        base=next(r for r in clip['methods'] if r['method']=='RAW')
        for row in clip['methods']:
            m=row['method']; scores=clip['flow_scores']['raft']
            guard=(row['AbsRel']<=base['AbsRel']+.01 and row['delta1']>=base['delta1']-.005 and row['edge_F1']>=base['edge_F1']-.01) if base['AbsRel'] is not None else None
            csv_rows.append(dict(clip=clip['name'],track=clip['track'],frames=n,method=m,AbsRel=row['AbsRel'],RMSE=row['RMSE'],delta1=row['delta1'],log_edge_F1=row['edge_F1'],RAFT_warp_proxy=scores[m],RAFT_proxy_reduction_pct=100*(1-scores[m]/scores['RAW']),pose_temporal_log_error=row['pose_temporal_log_error'],postprocess_fps=row['postprocess_fps'],accuracy_boundary_guard_passed=guard))
        if clip['track']=='self-recorded':
            scores=clip['flow_scores']['raft']; gains=[100*(1-scores[m]/scores['RAW']) for m in ('A','C','D','E_half')]
            lines.append(f'- {clip["name"]}：'+ ' / '.join(f'{g:.2f}%' for g in gains)+f'。[对比视频](../{a.video_root}/{clip["name"]}/comparison.mp4)。')
        else: gt_clips.append(clip)
        dest=evidence/clip['name']; dest.mkdir(exist_ok=True)
        for name in ('config_resolved.json','metadata.json','timing.json','manifest.json'):
            shutil.copyfile(baseline/name,dest/f'baseline_{name}')
        env=json.loads((baseline/'environment.json').read_text())
        write_json(dest/'software_environment.json',{k:env[k] for k in ('python','packages')})
        for name in ('per_pair_farneback.csv','per_pair_raft.csv','window_diagnostics.json'):
            shutil.copyfile(folder/name,dest/name)
        shutil.copyfile(folder/'directions/comparison.csv',dest/'depth_and_pose.csv')
    lines+=['','## TUM sample：深度真值与相机运动补偿','',
            '以下 pose-log-error 在共同的 GT 可见表面上补偿相机旋转/平移与 z 变化，不是光流 proxy，',
            '也不是原始 TUM 论文的轨迹 ATE/RPE。边界指标为 log-depth edge F1，不是 Depth Pro SI F1。','']
    for clip in gt_clips:
        base=next(r for r in clip['methods'] if r['method']=='RAW')
        lines += [f'### {clip["name"]}，{clip["frames"]} 帧','',f'原始 VDA：AbsRel {base["AbsRel"]:.6f}，δ1 {base["delta1"]*100:.3f}%，pose-log-error {base["pose_temporal_log_error"]:.6f}。','']
        for r in clip['methods']:
            if r['method']=='RAW': continue
            reduction=100*(1-r['pose_temporal_log_error']/base['pose_temporal_log_error'])
            passed=r['AbsRel']<=base['AbsRel']+.01 and r['delta1']>=base['delta1']-.005 and r['edge_F1']>=base['edge_F1']-.01
            lines.append(f'- {r["method"]}：AbsRel {r["AbsRel"]:.6f}；δ1 {r["delta1"]*100:.3f}%；pose-log-error {r["pose_temporal_log_error"]:.6f}（降低 {reduction:.2f}%）；edge F1 {r["edge_F1"]:.6f}；约束'+('通过。' if passed else '未通过。'))
        lines += ['',f'通过精度/边界约束的 pose 指标最优方法：**{clip["guard"]["accuracy_boundary_constrained_pose_best"]}**。','']
    lines += ['原生 VDA 在这两段上仍有米制误差，后处理的距离精度改善幅度很小。',
              '通过相对基线的退化约束不代表绝对精度已经良好；D 在本轮通过，但此前短片段的负结果仍需保留。','',
              '## iBims 跨数据集检查','']
    for model in ('depth-pro','unidepth-v2'):
        folder=ROOT/(a.depth_pro_root if model=='depth-pro' else a.unidepth_root)
        if not (folder/'summary.json').exists(): raise ValueError(f'{model} image audit must finish')
        image=json.loads((folder/'summary.json').read_text())
        if image['samples']!=100: raise ValueError('Need all 100 images')
        image_rows=list(csv.DictReader((folder/'per_image.csv').open()))
        if len(image_rows)!=200 or len({r['sample_id'] for r in image_rows})!=100:
            raise ValueError('Incomplete image pairs')
        for r in image_rows:
            path=folder/r['method']/f'{r["sample_id"]}.npy'
            if sha256(path)!=r['prediction_sha256']: raise ValueError('Image prediction changed')
            checksum_rows.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=r['prediction_sha256']))
        for method in ('RAW','C'):
            for metric,value in image['means'][method].items():
                if abs(float(np.mean([float(r[metric]) for r in image_rows if r['method']==method]))-value)>1e-12:
                    raise ValueError('Image summary differs from CSV')
        raw,new=(image['means'][m] for m in ('RAW','C'))
        lines.append(f'- {model}，100 图：AbsRel {raw["AbsRel"]:.6f} → {new["AbsRel"]:.6f}；masked ratio SI F1 {raw["masked_ratio_SI_boundary_F1"]:.6f} → {new["masked_ratio_SI_boundary_F1"]:.6f}；精度 `{image["settings"]["precision"]}`。')
        dest=evidence/f'ibims_{model}'; dest.mkdir(exist_ok=True)
        env=json.loads((folder/'environment.json').read_text())
        write_json(dest/'software_environment.json',{k:env[k] for k in ('python','packages')})
        for name in ('per_image.csv','summary.json','settings.json','weights.json','inference_times.csv'):
            shutil.copyfile(folder/name,dest/name)
    lines += ['', 'iBims 为研究作者公开的 MoGe 处理版：JPEG RGB、log-quantized depth，缺原始完整掩码。',
              '边界评分明确扩展为“相邻两 GT 像素均有效”的 masked ratio SI F1；全有效时已验证与官方公式一致，',
              '但不能直接与论文的原始 MAT/unmasked 表格排名。Depth Pro 使用 GPU FP32，',
              'UniDepthV2 使用 CPU FP32；仅在各自模型内部做 RAW/C 配对，不把精度/硬件差异当作方法优势。','',
              '## 复现与限制','',
              '运行入口：`scripts/run_extended_video_validation.py`、`scripts/download_ibims_core.py --processed-mirror`、',
              '`scripts/run_ibims_refinement_cpu.py --model depth-pro --device cuda --output outputs/ibims_refinement_gpu` 与 CPU `--model unidepth-v2`；',
              '环境和下载/临时缓存先通过 `. ./scripts/project_cache.ps1` 限制在项目目录。',
              'GPU 完整入口：`./scripts/run_extended_gpu_validation.ps1 -Python python`；它按顺序执行，避免多个 GPU 作业争用。',
              'Depth Pro 需使用兼容其官方 NumPy 约束的依赖组合；本地 `.runtime_depthpro` 在 `.runtime` 前。',
              f'数值副本、逐对表与校验在 `{a.evidence}/`；原始 RGB/NPY/valid/flow/视频在 outputs 对应目录。',
              '论文逐项协议与仍缺少的直接对照见 [论文审查](paper_comparison.md)。这些实验不构成 12 篇论文所有原任务的完整复现。','']
    write_csv(evidence/'summary_methods.csv',csv_rows); write_csv(evidence/'prediction_checksums.csv',checksum_rows)
    write_json(evidence/'validation.json',dict(all_passed=True,counts=counts))
    for name in ('summary.json','settings.json','commands.json'): shutil.copyfile(root/name,evidence/name)
    receipt=ROOT/'data/downloads/ibims_mirror_receipt.json'
    if receipt.exists(): shutil.copyfile(receipt,evidence/'ibims_download_receipt.json')
    write_json(evidence/'source_scripts.json',{name:sha256(ROOT/name) for name in (
        'scripts/run_extended_video_validation.py','scripts/run_ibims_refinement_cpu.py',
        'scripts/summarize_extended_validation.py','scripts/compare_video_resolutions.py',
        'src/infer.py','src/models/foundation.py','src/flow.py','src/exploration/compare.py',
        'src/exploration/edge_aware.py')})
    (ROOT/a.report).write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(counts),flush=True)


if __name__=='__main__': main()
