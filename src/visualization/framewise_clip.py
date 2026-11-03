"""课程1B：同一整段色标、逐帧边缘叠加和连续案例；不加载模型。"""
from pathlib import Path
import cv2
import numpy as np
from src.metrics.video_diagnostics import edge_map
from src.video.report import colorize_relative,_panels,_write_video,_figure


def save(path,rgb):
    if not cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise IOError(path)


def report(output,raw,rgb,frames,pairs,summary,config,fps):
    output=Path(output);out=output/'visualization';out.mkdir()
    ref=summary['reference'];span=ref['span'];limits=[ref['low'],ref['high'] if span else ref['low']+1]
    def colored(i):return colorize_relative(raw[i],limits)[0]
    def overlay(i):
        image=rgb[i].copy();image[edge_map(raw[i],span,config['diagnostics']['edge_threshold'])]=[255,40,40]
        return image
    def panels(i):
        return _panels([rgb[i],colored(i),overlay(i)],['RGB','DA-V2-S relative (not meters)','Depth edges (red; no GT)'],
                       f'Frame {i} | PTS {frames[i]["timestamp_seconds"]:.6f}s | fixed [{limits[0]:.4f}, {limits[1]:.4f}]')
    samples=np.unique(np.linspace(0,len(raw)-1,min(8,len(raw)),dtype=int))
    thumbnails=[]
    for i in samples:
        image=panels(int(i));ratio=min(1,1200/image.shape[1]);image=cv2.resize(image,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
        thumbnails.append(image)
    save(out/'contact.png',np.concatenate(thumbnails,axis=0))
    if config['visualization']['video']:
        _write_video(out/'rgb_relative_edges.mp4',(panels(i) for i in range(len(raw))),fps)
    candidates=out/'cases';candidates.mkdir()
    for j,case in enumerate(summary['case_candidates'],1):
        i=case['current_index'];indices=sorted({max(0,i-2),i-1,i,min(len(raw)-1,i+2)})
        save(candidates/f'case_{j:02d}_{case["role"]}_{i:04d}.png',np.concatenate([panels(k) for k in indices],axis=0))
    f=_figure(out/'diagnostics.png',figsize=(11,11));axes=f.subplots(4,1)
    axes[0].plot([r['timestamp_seconds'] for r in frames],[r['median'] for r in frames]);axes[0].set_ylabel('Raw relative median')
    ts=[r['timestamp_seconds'] for r in pairs]
    for key,label in [('pixel_rmse_normalized','Same pixel'),('flow_rmse_normalized','Farneback + FB')]:
        axes[1].plot(ts,[r.get(key) for r in pairs],label=label)
    axes[1].legend();axes[1].set_ylabel('Pair RMSE / fixed S')
    axes[2].plot(ts,[r.get('edge_instability') for r in pairs]);axes[2].set_ylabel('Edge change proxy')
    axes[3].plot(ts,[r.get('flow_coverage') for r in pairs]);axes[3].set_ylabel('Flow valid fraction');axes[3].set_xlabel('Original PTS (seconds)')
    for ax in axes:ax.grid(alpha=.25)
    f.suptitle('No depth/temporal GT: motion, occlusion and flow errors affect these diagnostics')
    f.savefig(out/'diagnostics.png',dpi=150);f.savefig(out/'diagnostics.pdf')
    f=_figure(out/'fixed_colorbar.png',figsize=(8,1.8));ax=f.subplots()
    ramp=np.tile(np.linspace(limits[0],limits[1],512),(25,1))
    ax.imshow(colorize_relative(ramp,limits)[0],extent=[limits[0],limits[1],0,1],aspect='auto');ax.set_yticks([])
    ax.set_xlabel('DA-V2 relative value; large = near; colors have no meter calibration')
    f.savefig(out/'fixed_colorbar.png',dpi=150)
    saturation={'below':int((raw<limits[0]).sum()),'above':int((raw>limits[1]).sum()),'total':int(raw.size)}
    from src.video.common import write_json
    write_json(out/'display.json',{'limits':limits,'palette':'TURBO','per_frame_normalization':False,
                                 'scope':'whole_clip_once','saturation_pixels':saturation,'prediction_clipped':False,
                                 'frames':len(raw),'fps':fps,'depth_unit':'arbitrary'})
    lines=['# DA-V2逐帧基线诊断','',f'真实输入{len(raw)}帧，原FPS={fps:g}。输出为relative，不是米；不读取GT，不运行VDA。','',
           '整段固定一次P1/P99显示范围，不逐帧归一化；显示范围外颜色截断不修改原生预测。','',
           '## 尺度约定','',
           '原生预测是近处数值大的相对量。正尺度/偏移演示保持相同显示，说明可视层次不能确定米制尺度；没有视频GT，不能报告视频AbsRel或“错了多少米”。',
           '', '## 边界与时间诊断','',
           '红色是固定阈值的预测深度梯度，不是GT轮廓。案例按变化分数定位，须人工看连续画面后解释，不自动宣称每个候选是模型失败。',
           '相同像素差含相机/物体运动；Farneback/FB对应误差和边缘变化受光流、遮挡影响，都是可选诊断，不是时序真值误差或米制尺度漂移。','',
           '主展示见visualization/rgb_relative_edges.mp4；每个案例列出前后真实帧。大范围变化、细结构和遮挡处须区分真实运动与预测不稳定。','',
           '## 辅助数值（帧对等权）','']
    for k,v in summary['aggregate'].items():lines.append(f'- {k}: {v}')
    lines+=['','## 可视检查候选','']+[f'- frame {c["current_index"]}: {c["role"]}, {c["metric"]}={c["value"]}' for c in summary['case_candidates']]
    lines+=['','100帧要求是否满足：'+str(summary['frame_scope']['requirement_100_frames_satisfied'])+'；素材不足时保留待课程确认，不能把70帧改写成100帧。','']
    (output/'analysis.md').write_text('\n'.join(lines))
    return [output/'frame_diagnostics.csv',output/'pair_diagnostics.csv',output/'analysis.md',*sorted(p for p in out.rglob('*') if p.is_file())]
