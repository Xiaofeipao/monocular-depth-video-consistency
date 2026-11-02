"""Compare frozen video results; input size and device change together."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.io import sha256,write_json


def main():
    roots=[ROOT/'outputs'/name for name in ('extended_video_validation_cpu','extended_video_validation_gpu518')]
    reports=[json.loads((r/'summary.json').read_text(encoding='utf-8')) for r in roots]
    if any(len(r['clips'])!=6 for r in reports): raise ValueError('Incomplete six-clip validation')
    rows=[]
    lines=['# 同帧视频输入分辨率检查','',
           '比较 CPU FP32 / 252 与 GPU FP32 / 518；分辨率与设备同时变化，不能单独归因于其中一个因素。',
           '四段 self-recorded 与两个 TUM sample 均使用同帧 RGB、同权重、同后处理参数。',
           '后处理仍使用 CPU 实现，GPU 用于 VDA 与独立 RAFT 评分。速度不作跨设备方法优势比较。','']
    for old,new in zip(*(r['clips'] for r in reports)):
        if old['name']!=new['name'] or old['frames']!=new['frames']: raise ValueError('Clips differ')
        manifests=[json.loads((root/old['name']/'baseline/manifest.json').read_text(encoding='utf-8')) for root in roots]
        if [r['rgb_sha256'] for r in manifests[0]]!=[r['rgb_sha256'] for r in manifests[1]]:
            raise ValueError('RGB frames differ')
        meta=json.loads((roots[1]/new['name']/'baseline/metadata.json').read_text(encoding='utf-8'))
        if meta['device']!='cuda' or meta['input_size']!=518: raise ValueError('Not requested GPU run')
        lines += [f'## {new["name"]}，{new["frames"]} 帧','']
        raw=next(r for r in new['methods'] if r['method']=='RAW')
        if raw['AbsRel'] is not None:
            before=next(r for r in old['methods'] if r['method']=='RAW')
            lines.append(f'原始 VDA AbsRel：{before["AbsRel"]:.6f} → {raw["AbsRel"]:.6f}；δ1：{before["delta1"]*100:.3f}% → {raw["delta1"]*100:.3f}%。')
            lines.append(f'518 下通过精度/边界约束的 pose 指标最优：{new["guard"]["accuracy_boundary_constrained_pose_best"]}。')
        for method in ('A','C','D','E_half'):
            gain=[]
            for clip in (old,new):
                s=clip['flow_scores']['raft']; gain.append(100*(1-s[method]/s['RAW']))
            row=dict(clip=new['name'],method=method,frames=new['frames'],proxy_gain_252_pct=gain[0],proxy_gain_518_pct=gain[1])
            if raw['AbsRel'] is not None:
                m=next(r for r in new['methods'] if r['method']==method)
                row.update(AbsRel_518=m['AbsRel'],delta1_518=m['delta1'],edge_F1_518=m['edge_F1'],pose_gain_518_pct=100*(1-m['pose_temporal_log_error']/raw['pose_temporal_log_error']))
            rows.append(row)
            lines.append(f'- {method} 的 RAFT warp proxy 降幅：{gain[0]:.2f}% → {gain[1]:.2f}%'+(f'；518 pose-log-error 降幅 {row["pose_gain_518_pct"]:.2f}%。' if 'pose_gain_518_pct' in row else '。'))
        lines += ['','运动相机和人物的 warp proxy 包含真实 z 变化；有 GT 时另看 pose-log-error。','']
    lines += ['## 解释范围','',
              '这是已复现 Part 2 与固定 Part 3 的同帧消融；没有训练、GT 尺度拟合或根据测试集重选参数。',
              '四段手机来自同一场景，两个 sample 来自同一 TUM 序列；不能据此宣布跨场景最优或超过 12 篇论文。',
              '完整指标、逐对表、预测 SHA256、视频及限制分别见 [252 验证](extended_validation.md) 与 [518 验证](extended_validation_gpu518.md)。','']
    dest=ROOT/'docs/evidence/extended_validation_gpu518'
    write_json(dest/'resolution_comparison.json',dict(rows=rows,source_summary_sha256=[sha256(r/'summary.json') for r in roots],same_RGB_verified=True,confound='input resolution and device both change'))
    (ROOT/'docs/video_resolution_comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Verified identical RGB for all six clips')


if __name__=='__main__': main()
