"""统一生成单模型/双模型报告；逐像素差分与光流差分分开显示。"""
from pathlib import Path
import cv2
import numpy as np

from .common import write_json
from .clip_workflow import identity
from .report import _figure, _panels, _write_video, colorize_relative
from .evaluation import _safe_warp
from .clip_analysis import METRICS

TYPE_NOTES={
    'fixed_static':'无人固定：全图背景std与逐像素变化是闪烁主诊断，光流项作共同评价；轻微相机抖动、照明和自动曝光也可能影响结果。',
    'fixed_dynamic':'固定有人：光流对应后的变化是主诊断，人物活动/遮挡仍会产生真实变化；逐像素项混合人物运动与模型闪烁，全图背景std不适用。',
    'moving':'移动相机：光流误差与逐帧中位数曲线为主诊断；逐像素项含视差，不能单独评价闪烁；中位数同时受真实距离、视野组成和遮挡影响，全图背景std不适用。',
}


def generate(output, summary, sources, pairs, frames):
    output=Path(output);report=output/'report';report.mkdir()
    names=sources['names'];raw=sources['raw'];manifest=sources['manifest'];config=summary['config']['visualization']
    if config.get('colormap') != 'turbo':
        raise ValueError('当前报告仅支持明确冻结的TURBO色标')
    ranges={'raw':config['raw_range'],'normalized':config['normalized_range'],'difference':config['difference_range']}
    assets=[];saturation={};labels={'dav2':'DA-V2-S','vda':'VDA-S offline'}
    def add(path,role,**extra):
        assets.append({**identity(path),'path':str(path.relative_to(output)),'role':role,**extra})
    def rgb(i):
        value=cv2.imread(str(sources['manifest_path'].parent/manifest['frames'][i]['rgb_path']))
        if value is None:raise ValueError('RGB图片无法读取')
        return cv2.cvtColor(value,cv2.COLOR_BGR2RGB)
    def values(name,i,view):
        value=np.asarray(raw[name][i],np.float64)
        if view=='normalized':
            reference=summary['models'][name]['reference'];span=reference['span']
            value=(value-reference['percentile_low_value'])/span if span else np.full(value.shape,np.nan)
        return value
    def depth_panel(i,view):
        frame=manifest['frames'][i]
        return _panels([rgb(i)]+[colorize_relative(values(n,i,view),ranges[view])[0] for n in names],
                       ['RGB']+[labels[n]+' | '+view for n in names],
                       f"frame={i} source={frame['source_index']} t={frame['timestamp_seconds']:.4f}s | fixed {ranges[view]}")
    def difference_panel(i,kind):
        images=[rgb(i)];titles=['Current RGB']
        mask=None
        for name in names:
            previous=np.asarray(raw[name][i-1],np.float64)
            if kind=='flow':
                previous,_=_safe_warp(raw[name][i-1],sources['flows'][i-1][1])
                mask=np.load(output/name/f'masks/pair_{i-1:06d}_{i:06d}.npy',allow_pickle=False)
            span=summary['models'][name]['reference']['span']
            delta=np.abs(np.asarray(raw[name][i],np.float64)-previous)/span if span else np.full(previous.shape,np.nan)
            images.append(colorize_relative(delta,ranges['difference'],mask)[0]);titles.append(labels[name]+' | '+kind+' delta / S')
        if kind=='flow':
            images.append(np.repeat(mask[...,None],3,axis=2).astype(np.uint8)*255);titles.append('Common FB mask')
        return _panels(images,titles,f"pair={i-1}->{i} t={pairs[i-1]['timestamp_seconds']:.4f}s | {kind}; fixed {ranges['difference']}")
    for view in ('raw','normalized'):
        saturation[view]={}
        for name in names:
            totals={k:0 for k in ('valid_pixels','invalid_pixels','below_range_pixels','above_range_pixels')}
            for i in range(len(frames)):
                _,stats=colorize_relative(values(name,i,view),ranges[view])
                for key in totals:totals[key]+=stats[key]
            saturation[view][name]={**totals,**{k+'_fraction':totals[k+'_pixels']/totals['valid_pixels'] if totals['valid_pixels'] else None for k in ('below_range','above_range')}}
        indices=sorted({0,len(frames)//2,len(frames)-1})
        path=report/f'contact_{view}.png'
        if not cv2.imwrite(str(path),cv2.cvtColor(np.concatenate([depth_panel(i,view) for i in indices]),cv2.COLOR_RGB2BGR)):
            raise RuntimeError('保存首中尾图失败')
        add(path,'contact',indices=indices)
        if config.get('generate_video',True):
            path=report/f'depth_{view}.mp4';add(path,'depth_video',**_write_video(path,(depth_panel(i,view) for i in range(len(frames))),manifest['fps']))
    for kind in ('pixel','flow'):
        if config.get('generate_video',True):
            path=report/f'{kind}_difference.mp4'
            add(path,kind+'_difference',**_write_video(path,(difference_panel(i,kind) for i in range(1,len(frames))),manifest['fps']))
        selected=set()
        for name in names:
            valid=[p for p in pairs if p[name+'_'+kind+'_mae'] is not None]
            if valid:selected.add(max(valid,key=lambda p:p[name+'_'+kind+'_mae'])['current_index'])
        if len(names)==2:
            valid=[p for p in pairs if p[kind+'_mae_absolute_reduction'] is not None]
            if valid:
                selected.add(max(valid,key=lambda p:p[kind+'_mae_absolute_reduction'])['current_index'])
                selected.add(min(valid,key=lambda p:p[kind+'_mae_absolute_reduction'])['current_index'])
        if selected:
            path=report/f'{kind}_cases.png'
            if not cv2.imwrite(str(path),cv2.cvtColor(np.concatenate([difference_panel(i,kind) for i in sorted(selected)]),cv2.COLOR_RGB2BGR)):
                raise RuntimeError('保存差分案例失败')
            add(path,kind+'_cases',current_indices=sorted(selected))
    figure=_figure(report/'curves.png',(12,17));axes=figure.subplots(8,1,sharex=True)
    t=[f['timestamp_seconds'] for f in frames];pt=[p['timestamp_seconds'] for p in pairs]
    def curve(rows,key):return [np.nan if r[key] is None else r[key] for r in rows]
    for name in names:
        for axis,key in [(0,'median_ratio'),(1,'normalized_median_delta')]:axes[axis].plot(t,curve(frames,name+'_'+key),label=labels[name])
        for axis,key in [(2,'pixel_mae'),(3,'pixel_rmse'),(4,'flow_mae'),(5,'flow_rmse')]:axes[axis].plot(pt,curve(pairs,name+'_'+key),label=labels[name])
    axes[6].plot(pt,[p['coverage'] for p in pairs],label='common flow valid');axes[6].plot(pt,[p['pixel_coverage'] for p in pairs],label='pixel full frame',linestyle=':')
    axes[6].set_ylim(-.02,1.05)
    if sources['record']['clip']['type']=='fixed_static':
        for name in names:
            path=output/name/'background_normalized_std.npy'
            if path.exists():
                value=np.load(path);im=axes[7].imshow(value,cmap='turbo',vmin=0,vmax=.1,aspect='auto',extent=[t[0],t[-1],value.shape[0],0]);figure.colorbar(im,ax=axes[7]);break
        axes[7].set_ylabel('First model BG std map')
    else:axes[7].text(.03,.5,'Background std: N/A (real motion)',transform=axes[7].transAxes)
    for ax,label in zip(axes[:7],('Median / first median','Median change / fixed S','Pixel MAE / S','Pixel RMSE / S','Flow MAE / S','Flow RMSE / S','Coverage')):
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend()
        if 'vda' in sources['runs']:
            for i in sources['runs']['vda']['model'].get('window_starts',[]):
                if 0 < i < len(t):ax.axvline(t[i],color='grey',linestyle=':',alpha=.25)
    axes[-1].set_xlabel('Source time (s); dotted lines: input window starts, not exact output seams')
    figure.suptitle(summary['clip']['clip_id']+' | '+summary['clip']['type']+' | no GT / no per-frame alignment')
    for extension in ('png','pdf'):
        path=report/f'curves.{extension}';figure.savefig(path,dpi=140);add(path,'curves')
    figure.clear()
    if sources['record']['clip']['type']=='fixed_static':
        fig=_figure(report/'background_std.png',(6*len(names),5));axs=np.atleast_1d(fig.subplots(1,len(names)))
        for name,ax in zip(names,axs):
            path=output/name/'background_normalized_std.npy'
            value=np.load(path) if path.exists() else np.full(sources['shape'],np.nan)
            im=ax.imshow(value,cmap='turbo',vmin=0,vmax=.1);ax.set_title(labels[name]+' | full-frame BG std / S');fig.colorbar(im,ax=ax)
        path=report/'background_std.png';fig.savefig(path,dpi=140);add(path,'full_frame_background');fig.clear()
    def num(value):return 'N/A' if value is None else f'{value:.8f}'
    notes=['# '+summary['clip']['clip_id']+'：视频诊断','','类型：'+summary['clip']['type']+'。'+TYPE_NOTES[summary['clip']['type']],
           '',f"共{len(frames)}帧、{len(pairs)}对；无深度GT，原生relative、arbitrary、alignment:none。",
           '', '| 指标 | '+' | '.join(labels[n] for n in names)+' |','|---|'+'---:|'*len(names)]
    labels_cn={'flow_mae':'光流MAE/S','flow_rmse':'光流RMSE/S','pixel_mae':'逐像素MAE/S','pixel_rmse':'逐像素RMSE/S','background_std':'全图背景std/S','median_delta':'平均绝对中位数变化/S','median_log_ratio':'平均绝对log中位数比'}
    for metric,(section,field) in METRICS.items():notes.append('| '+labels_cn[metric]+' | '+' | '.join(num(summary['models'][n][section][field]) for n in names)+' |')
    if len(names)==2:
        notes += ['', '| 指标 | 降低率（正数为变化减少） |','|---|---:|']
        for metric in METRICS:
            change=summary['improvement'][metric]['relative_reduction'];notes.append('| '+labels_cn[metric]+' | '+('N/A' if change is None else f'{change:+.2%}')+' |')
    notes += ['', '| 模型 | 参数量 | 权重MiB | FPS | 均摊ms/输出帧 | 峰值allocated MiB | 峰值reserved MiB |','|---|---:|---:|---:|---:|---:|---:|']
    for n in names:
        timing=summary['timing'][n];model=summary['model_metadata'][n]
        size=model.get('checkpoint_bytes')
        notes.append('| '+labels[n]+' | '+str(model.get('parameters','N/A'))+' | '+num(size/1024**2 if size is not None else None)+' | '+num(timing.get('model_fps'))+' | '+num(timing.get('amortized_ms_per_output_frame'))+' | '+' | '.join(num(timing.get(k)/1024**2) if timing.get(k) is not None else 'N/A' for k in ('peak_gpu_allocated_bytes','peak_gpu_reserved_bytes'))+' |')
    for n in names:
        m=summary['model_metadata'][n]
        notes += ['',labels[n]+'实际网络H×W='+str(m.get('network_input_hw'))+'，输出H×W='+str(m.get('actual_output_hw'))+'，设备='+str(m.get('device'))+'，精度='+str(m.get('precision'))+'。']
    notes += ['', 'S为各模型第0帧全图P95−P5，整段固定；不逐帧调整预测。零跨度N/A，有限零值合法，非有限预测明确失败。',
              '逐像素指标使用相同屏幕坐标、不用光流或FB mask；光流指标使用相同对应域。两者有效域不同，应分别比较同项两模型，不能只比较两项数值大小判断算法好坏。',
              'raw和normalized均固定色标，后者只用首帧常数。饱和比例见report.json，截断只影响显示、不影响指标。',
              '中位数变化是尺度proxy，受视野、真实运动与任意零点影响，不等于真实米制漂移。VDA离线使用未来帧及官方预测间scale/shift拼接，无GT。',
              'FPS来自实际模型计时，不含解码/RAFT/保存/绘图；离线均摊ms不是在线首帧延迟。DA/VDA可能是不同调用时刻，结合实际环境记录解释。',
              'pixel_cases/flow_cases保留各模型高变化、最大改善与最小改善（可能恶化）案例。自动说明不替代人工检查遮挡、拖影、细结构、镜面与运动模糊；稳定不等于准确。']
    path=report/'analysis.md';path.write_text('\n'.join(notes)+'\n');add(path,'chinese_notes')
    metadata={'ranges':ranges,'saturation':saturation,'per_frame_normalization':False,'manual_roi':False,
              'pixel_domain':'full_frame_no_warp_no_FB_mask','type_note':TYPE_NOTES[summary['clip']['type']],
              'official_window_starts_are_not_exact_output_seams':True}
    path=report/'report.json';write_json(path,metadata);add(path,'display_metadata')
    return {**metadata,'assets':assets}
