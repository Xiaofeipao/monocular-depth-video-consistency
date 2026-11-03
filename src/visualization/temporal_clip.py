"""VDA独立报告与同帧DA/VDA报告；不加载模型，不修改原生预测。"""
from pathlib import Path

import cv2
import numpy as np

from src.metrics.video_diagnostics import edge_map
from src.video.common import write_json
from src.video.report import _panels, _write_video, _figure, colorize_relative


def save(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(path)


def limits(reference):
    return [reference['low'], reference['high'] if reference['span'] else reference['low'] + 1]


def overlay(rgb, raw, reference, config):
    result = rgb.copy()
    result[edge_map(raw, reference['span'], config['diagnostics']['edge_threshold'])] = [255, 40, 40]
    return result


def stack_panels(output, indices, panel):
    save(output, np.concatenate([panel(int(i)) for i in indices], axis=0))


def neighborhood(index, count):
    start = min(max(0, index - 2), max(0, count - 5))
    return list(range(start, min(count, start + 5)))


def review(output, cases, panel, summary):
    output.mkdir()
    for case in cases:
        stack_panels(output / (case['id'] + '.png'), case['frames'],
                     lambda i: panel(i, case['bbox_xyxy']))
    write_json(output / 'review_sources.json', {
        'runs': summary.get('sources', summary.get('run')),
        'fixed_reference': summary.get('reference'), 'cases': cases,
        'models_executed': False, 'same_screen_coordinates_not_flow_aligned': True,
        'role': 'display-only continuous fixed crops; no ROI metric masks',
    })


def _crop(images, box):
    if box is None:
        return images
    x1, y1, x2, y2 = box
    return [im[y1:y2, x1:x2] for im in images]


def standalone_report(output, raw, rgb, frames, pairs, summary, config, fps):
    out = output / 'visualization'
    out.mkdir()
    ref = summary['reference']; display_limits = limits(ref)

    def panel(i, box=None):
        images = [rgb[i], colorize_relative(raw[i], display_limits)[0], overlay(rgb[i], raw[i], ref, config)]
        return _panels(_crop(images, box), ['RGB', 'VDA-S relative (not meters)', 'Depth edges (red; no GT)'],
                       f'Original frame {i} | PTS {frames[i]["timestamp_seconds"]:.6f}s | fixed {display_limits} | crop {box}')

    samples = np.unique(np.linspace(0, len(raw) - 1, min(8, len(raw)), dtype=int))
    thumbnails = []
    for i in samples:
        im = panel(int(i)); ratio = min(1, 1200 / im.shape[1])
        thumbnails.append(cv2.resize(im, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA))
    save(out / 'contact.png', np.concatenate(thumbnails, axis=0))
    if config['visualization']['video']:
        _write_video(out / 'rgb_relative_edges.mp4', (panel(i) for i in range(len(raw))), fps)
    cases = out / 'cases'; cases.mkdir()
    for j, case in enumerate(summary['case_candidates'], 1):
        i = case['current_index']
        stack_panels(cases / f'case_{j:02d}_{case["role"]}_{i:04d}.png', neighborhood(i, len(raw)), panel)
    review(output / 'review', config.get('review', {}).get('cases', []), panel, summary)
    fig = _figure(out / 'diagnostics.png', figsize=(11, 11)); axes = fig.subplots(4, 1)
    axes[0].plot([r['timestamp_seconds'] for r in frames], [r['median'] for r in frames])
    axes[0].set_ylabel('Raw relative median')
    ts = [r['timestamp_seconds'] for r in pairs]
    for k, label in [('pixel_rmse_normalized', 'Same pixel'), ('flow_rmse_normalized', 'Farneback + FB')]:
        axes[1].plot(ts, [r.get(k) for r in pairs], label=label)
    axes[1].legend(); axes[1].set_ylabel('Pair RMSE / fixed S')
    axes[2].plot(ts, [r.get('edge_instability') for r in pairs]); axes[2].set_ylabel('Edge change proxy')
    axes[3].plot(ts, [r.get('flow_coverage') for r in pairs]); axes[3].set_ylabel('Flow valid fraction')
    axes[3].set_xlabel('Original PTS (seconds)')
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle('VDA offline relative: no depth/temporal GT; motion, occlusion and flow errors affect diagnostics')
    fig.savefig(out / 'diagnostics.png', dpi=150); fig.savefig(out / 'diagnostics.pdf')
    fig = _figure(out / 'fixed_colorbar.png', figsize=(8, 1.8)); ax = fig.subplots()
    ramp = np.tile(np.linspace(*display_limits, 512), (25, 1))
    ax.imshow(colorize_relative(ramp, display_limits)[0], extent=[*display_limits, 0, 1], aspect='auto')
    ax.set_yticks([]); ax.set_xlabel('VDA relative value; large = near; no meter calibration')
    fig.savefig(out / 'fixed_colorbar.png', dpi=150)
    write_json(out / 'display.json', {'limits': display_limits, 'reference': ref, 'palette': 'TURBO',
                                     'scope': 'whole_clip_once', 'per_frame_normalization': False,
                                     'saturation_pixels': {'below': int((raw < display_limits[0]).sum()),
                                                           'above': int((raw > display_limits[1]).sum()), 'total': int(raw.size)},
                                     'prediction_clipped': False, 'depth_unit': 'arbitrary', 'fps': fps, 'frames': len(raw)})
    text = ['# VDA-S离线相对深度分析', '', f'真实输入{len(raw)}帧，原FPS={fps:g}；离线非因果、relative/arbitrary，无视频GT，不运行米制或streaming。', '',
            '## 尺度、边界与时间诊断', '',
            '原生数值通常越大越近，不能读取成米。2D+5演示只说明固定正尺度/偏移不能由相对显示确定，不改写预测。',
            '整段P1/P99确定一次TURBO色标，不逐帧归一化。红色是平滑预测深度的梯度阈值区域，不是GT物体轮廓；粗细不是置信度。',
            '同坐标变化混入真实运动；Farneback及FB有效域是近似对应，边缘变化允许2像素邻域匹配。没有视频GT，不能将它们称为准确率、纯闪烁误差或真实米制尺度漂移。', '',
            '## 完整精度数值（相邻帧对等权）', '']
    text += [f'- {k}: {v}' for k, v in summary['aggregate'].items()]
    text += ['', '## 图像阅读与实际观察', '',
             'diagnostics.png从上到下为原生中位数、同坐标/对应后RMSE/S、边缘变化、光流覆盖。横坐标为原始PTS，帧对放在后一帧时间；MAE/S另见CSV。',
             '先看visualization/rgb_relative_edges.mp4，再看review中的连续局部图。三个固定窗口仅用于展示，不作为数值评价ROI；自动高变化候选另在visualization/cases。',
             '真实视觉结论需要观看当前预测后记录，不能仅凭较低曲线宣称无闪烁、边界准确或困难区域恢复成功。', '',
             '## 官方窗口与效率', '',
             '官方32帧窗口、10帧重叠、22帧步长、8帧混合；预测间scale/shift衔接属于官方算法，不使用GT。内部尾部补齐不算新增真实帧。',
             '计时为完整离线序列的均摊成本，包含窗口和衔接；不是在线首帧延迟。分位数是多次整段耗时分位数除以真实帧数，不是逐帧延迟分布。', '',
             f'课程100真实帧门禁：{summary["frame_scope"]["requirement_100_frames_satisfied"]}；短素材保持待课程确认，不补帧凑数。', '']
    (output / 'analysis.md').write_text('\n'.join(text))


def comparison_report(output, inputs, summary):
    out = output / 'visualization'; out.mkdir()
    names = ('dav2', 'vda'); rgb = inputs['rgb']; manifest = inputs['manifest']
    count, fps = len(rgb), manifest['fps']
    refs = summary['reference']

    def image(name, i, edges=False):
        item = inputs[name]
        if edges:
            return overlay(rgb[i], item['raw'][i], refs[name], item['run']['config'])
        return colorize_relative(item['raw'][i], limits(refs[name]))[0]

    def panel(i, box=None, edges=False):
        images = [rgb[i], image('dav2', i, edges), image('vda', i, edges)]
        suffix = 'edges (red; no GT)' if edges else 'relative (not meters)'
        return _panels(_crop(images, box), ['RGB', f'DA-V2-S {suffix}', f'VDA-S {suffix}'],
                       f'Original frame {i} | PTS {manifest["frames"][i]["timestamp_seconds"]:.6f}s | own whole-clip fixed ranges | crop {box}')

    if inputs['vda']['run']['config']['visualization']['video']:
        _write_video(out / 'rgb_dav2_vda.mp4', (panel(i) for i in range(count)), fps)
        _write_video(out / 'rgb_dav2_vda_edges.mp4', (panel(i, edges=True) for i in range(count)), fps)
    sample = np.unique(np.linspace(0, count - 1, min(8, count), dtype=int))
    thumbs = []
    for i in sample:
        im = panel(int(i)); ratio = min(1, 1200 / im.shape[1])
        thumbs.append(cv2.resize(im, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA))
    save(out / 'contact.png', np.concatenate(thumbs, axis=0))
    review(output / 'review', inputs['vda']['run']['config'].get('review', {}).get('cases', []), panel, summary)
    cases = out / 'cases'; cases.mkdir()
    for j, case in enumerate(summary['case_candidates'], 1):
        stack_panels(cases / f'{j:02d}_{case["metric"]}_{case["role"]}_{case["current_index"]:04d}.png',
                     neighborhood(case['current_index'], count), panel)
    fig = _figure(out / 'diagnostics.png', figsize=(11, 11)); axes = fig.subplots(4, 1)
    ts = [f['timestamp_seconds'] for f in inputs['vda']['frames']]
    pair_ts = [p['timestamp_seconds'] for p in inputs['vda']['pairs']]
    for name in names:
        item = inputs[name]
        axes[0].plot(ts, [f['median'] for f in item['frames']], label=name.upper())
        for k, style in [('pixel_rmse_normalized', '--'), ('flow_rmse_normalized', '-')]:
            axes[1].plot(pair_ts, [p[k] for p in item['pairs']], style, label=f'{name.upper()} {k.split("_")[0]}')
        axes[2].plot(pair_ts, [p['edge_instability'] for p in item['pairs']], label=name.upper())
    axes[0].set_ylabel('Raw median (own arbitrary units)')
    axes[1].set_ylabel('Pair RMSE / own fixed S')
    axes[2].set_ylabel('Edge change proxy')
    axes[3].plot(pair_ts, [p['flow_coverage'] for p in inputs['vda']['pairs']])
    axes[3].set_ylabel('Common flow valid fraction'); axes[3].set_xlabel('Original PTS (seconds)')
    for i, ax in enumerate(axes):
        ax.grid(alpha=.25)
        if i < 3:
            ax.legend()
    fig.suptitle('Same RGB frames; own fixed whole-clip units; no depth/temporal GT or shared meter scale')
    fig.savefig(out / 'diagnostics.png', dpi=150); fig.savefig(out / 'diagnostics.pdf')
    write_json(out / 'display.json', {'palette': 'TURBO', 'per_frame_normalization': False,
                                     'scope': 'whole_clip_once', 'reference': refs, 'fps': fps, 'frames': count,
                                     'shared_meter_scale': False, 'prediction_clipped': False,
                                     'saturation_pixels': {name: {'below': int((inputs[name]['raw'] < refs[name]['low']).sum()),
                                                                   'above': int((inputs[name]['raw'] > limits(refs[name])[1]).sum()),
                                                                   'total': int(inputs[name]['raw'].size)} for name in names}})
    text = ['# 同帧DA-V2-S与VDA-S比较', '',
            '两条保存的原生预测只读参与比较，不重新推理，不改写1B/2B文件。相同RGB、源帧号、PTS、尺寸、精度、调色板和评价规则已核对。',
            '每模型使用自己的整段固定P1/P99跨度；颜色不是共同米制距离。没有视频GT，以下是变化诊断，不是深度准确率排名。', '',
            '| 指标 | DA-V2-S | VDA-S | DA−VDA | 相对降低率（%） | 共同帧对数 |',
            '|---|---:|---:|---:|---:|---:|']
    for k, v in summary['metrics'].items():
        def fmt(x):
            return 'N/A' if x is None else f'{x:.8f}'
        text.append(f'| {k} | {fmt(v["dav2"])} | {fmt(v["vda"])} | {fmt(v["dav2_minus_vda"])} | {fmt(v["reduction_percent"])} | {v["common_pair_count"]} |')
    text += ['', '正差表示本项VDA变化较小，负差原样保留；零基线或不可评价项不计算降低率。每项仅使用双方均可评价的共同帧对，不删去负结果。', '',
             '## 效率', '', '| 模型 | 平均ms/真实帧 | FPS | 峰值allocated MiB |', '|---|---:|---:|---:|']
    for name in names:
        timing = summary['efficiency'][name]['timing']; peak = timing['peak_gpu_allocated_bytes']
        text.append(f'| {name} | {timing["mean_seconds_per_frame"] * 1000:.8f} | {timing["fps"]:.8f} | {peak / 2**20 if peak is not None else "N/A"} |')
    text += ['', 'DA是独立单帧调用，VDA为整段离线时间除以真实帧数；不比较两者P95为同一延迟分布。硬件、尺寸、参数、权重、显存和详细范围见summary.json。', '',
             '## 逐项观察', '',
             '观看visualization/rgb_dav2_vda.mp4及预测边缘视频，再检查review中相同连续帧和相同坐标的三个窗口。',
             '分别记录颜色跳动、边界粗细/偏移/断裂、运动拖影与遮挡显露，以及是否丢失细结构。平滑但细节消失不能直接称全面改善。',
             '高低差候选在visualization/cases；文件名使用最大/最小DA−VDA，不预设一定存在改善或恶化。它们仍需实际观看后解释。',
             '中位数受视野组成与任意零点影响，不是米制尺度漂移；同坐标项含真实运动，光流项含对应误差，低边缘变化不证明边缘位置正确。', '',
             f'课程100真实帧门禁：{summary["frame_scope"]["requirement_100_frames_satisfied"]}；内部补齐不替代原始素材。', '']
    (output / 'analysis.md').write_text('\n'.join(text))
