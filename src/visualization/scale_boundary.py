"""Standalone figures, local object contours, profiles and a Chinese numeric report."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

from src.analysis.scale_boundary import annotation_edges, depth_edge_map, extract_profile, chebyshev_distance
from src.data.nyuv2 import NYUv2LabeledDataset

LABELS={'depth_pro':'Depth Pro','unidepth_v2':'UniDepthV2-S'}
COLORS={'depth_pro':'#2673ad','unidepth_v2':'#c84949'}


def finish(figure,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,dpi=160,bbox_inches='tight')
    plt.close(figure)


def overview_figures(c,tables,groups,summary,output):
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,model in zip(axes,LABELS):
        rows=[r for r in tables['per_image'] if r['model']==model]
        raw=np.array([r['raw_scale_ratio'] for r in rows]);clipped=np.array([r['clipped_scale_ratio'] for r in rows])
        bins=np.linspace(min(.5,raw.min()),max(1.5,raw.max()),45)
        ax.hist(raw,bins=bins,alpha=.55,label='raw prediction',color=COLORS[model])
        ax.hist(clipped,bins=bins,histtype='step',linewidth=1.7,label='scoring-clipped prediction',color='black')
        ax.axvline(1,color='black',ls=':');ax.set(title=LABELS[model],xlabel='per-image median prediction / GT',ylabel='image count');ax.legend(fontsize=8)
    finish(fig,output/'figures/scale_distribution.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for model in LABELS:
        rows=[r for r in groups['depth_bins'] if r['model']==model and r['track']=='native_clipped']
        x=np.array([(r['lower_m']+r['upper_m'])/2 for r in rows])
        for ax,key in zip(axes,('relative_bias','rmse_m')):
            vals=np.array([r[key+'_mean'] if r[key+'_mean'] is not None else np.nan for r in rows]);lo=np.array([r[key+'_lower'] if r[key+'_lower'] is not None else np.nan for r in rows]);hi=np.array([r[key+'_upper'] if r[key+'_upper'] is not None else np.nan for r in rows])
            ax.plot(x,vals,'o-',color=COLORS[model],label=LABELS[model]);ax.fill_between(x,lo,hi,color=COLORS[model],alpha=.15);ax.grid(alpha=.2)
    axes[0].axhline(0,color='black',lw=.8);axes[0].set(ylabel='signed relative bias',xlabel='GT distance bin center (m)',title='Bias versus true distance (clipped)')
    axes[1].set(ylabel='image-macro RMSE (m)',xlabel='GT distance bin center (m)',title='Absolute error versus true distance');axes[1].legend()
    finish(fig,output/'figures/distance_error.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ms=list(LABELS);bias=[summary['models'][m]['log_bias_squared']['mean'] for m in ms];shape=[summary['models'][m]['centered_log_mse']['mean'] for m in ms]
    axes[0].bar(ms,bias,label='squared mean log bias');axes[0].bar(ms,shape,bottom=bias,label='centered log-error variance');axes[0].set(ylabel='unclipped mean squared log error',title='Exact log-error identity');axes[0].legend(fontsize=8)
    for m in ms:
        rows=[r for r in tables['per_image'] if r['model']==m]
        axes[1].scatter([r['raw_abs_log_scale'] for r in rows],[r['native_clipped_abs_rel'] for r in rows],s=9,alpha=.35,label=LABELS[m],color=COLORS[m])
    axes[1].set(xlabel='absolute log of raw median scale ratio',ylabel='native clipped AbsRel',title='Association is not causation');axes[1].legend()
    finish(fig,output/'figures/scale_and_shape.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,m in zip(axes,ms):
        for ratio in c['boundary']['ratios']:
            rows=[r for r in groups['boundary'] if r['model']==m and r['track']=='native_raw' and r['ratio']==ratio]
            ax.plot([r['tolerance_px'] for r in rows],[r['f1_mean'] for r in rows],'o-',label=f'ratio {ratio:g}: all depth-edge F1')
        rows=[r for r in groups['boundary'] if r['model']==m and r['track']=='native_raw' and r['ratio']==c['boundary']['primary_ratio']]
        ax.plot([r['tolerance_px'] for r in rows],[r['object_edge_recall_mean'] for r in rows],'k--',label='annotation-supported edge recall')
        ax.set(title=LABELS[m],xlabel='Chebyshev tolerance (px)',ylabel='score',xticks=[1,2,3]);ax.legend(fontsize=7);ax.grid(alpha=.2)
    finish(fig,output/'figures/boundary_sensitivity.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    regions=('edge_le3px','interior_gt5px','object_edge_le3px')
    x=np.arange(3)
    for ax,m in zip(axes,ms):
        for track,offset,label in [('native_clipped',-.18,'native metric'),('oracle_raw_median_clipped',.18,'GT-median diagnostic')]:
            rows=[next(r for r in groups['regions'] if r['model']==m and r['track']==track and r['region']==name) for name in regions]
            ax.bar(x+offset,[r['abs_rel_mean'] for r in rows],width=.34,label=label)
        ax.set(title=LABELS[m],ylabel='image-macro AbsRel',xticks=x,xticklabels=['depth edge <=3 px','interior >5 px','object edge <=3 px']);ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8)
    finish(fig,output/'figures/scale_boundary_connection.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    disjoint=('edge_0_1px','edge_1_3px','edge_3_5px','interior_gt5px')
    for ax,key,title in zip(axes,('pixel_fraction','squared_error_share'),('Valid-pixel distribution','Clipped squared-error distribution')):
        bottom=np.zeros(2)
        for region in disjoint:
            rows=[next(r for r in groups['regions'] if r['model']==m and r['track']=='native_clipped' and r['region']==region) for m in ms]
            values=np.array([r[key+'_mean'] for r in rows]);ax.bar(ms,values,bottom=bottom,label=region);bottom+=values
        ax.set(title=title,ylabel='image-macro fraction',ylim=(0,1.01));ax.legend(fontsize=7)
    finish(fig,output/'figures/error_contributions.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    statuses=sorted({r['status'] for r in tables['profiles']});bottom=np.zeros(2)
    for status in statuses:
        values=np.array([summary['profiles'][m]['status_counts'].get(status,0) for m in ms])
        axes[0].bar(ms,values,bottom=bottom,label=status);bottom+=values
    axes[0].set(title='GT-selected profile outcomes',ylabel='profile count');axes[0].legend(fontsize=7)
    for m in ms:
        rows=[r for r in tables['profiles'] if r['model']==m and r['status']=='measurable']
        axes[1].scatter([r['gt_width_10_90_px'] for r in rows],[r['width_10_90_px'] for r in rows],s=10,alpha=.35,label=LABELS[m],color=COLORS[m])
    axes[1].plot([0,c['profiles']['maximum_gt_width_px']],[0,c['profiles']['maximum_gt_width_px']],'k:');axes[1].set(xlabel='GT 10-90% transition width (px)',ylabel='predicted width (px)',title='Measurable profiles only; coverage shown separately');axes[1].legend(fontsize=8)
    finish(fig,output/'figures/profile_geometry.png')


def select_cases(c,tables):
    images={r['test_position']:r for r in tables['per_image'] if r['model']=='depth_pro'}
    ud={r['test_position']:r for r in tables['per_image'] if r['model']=='unidepth_v2'}
    edge={m:{r['test_position']:r for r in tables['boundary'] if r['model']==m and r['track']=='native_raw' and r['ratio']==c['boundary']['primary_ratio'] and r['tolerance_px']==3} for m in LABELS}
    chosen=[dict(test_position=i,reason='fixed_illustrative_case') for i in c['cases']['fixed_test_positions'] if i in images]
    used={r['test_position'] for r in chosen};scenes={images[i]['scene'] for i in used}
    scores=[
        ('depth_pro_low_scale',lambda i:-images[i]['raw_scale_ratio']),
        ('unidepth_high_scale',lambda i:ud[i]['raw_scale_ratio']),
        ('clipping_exposure',lambda i:max(images[i]['raw_out_of_range_fraction'],ud[i]['raw_out_of_range_fraction'])),
        ('depth_pro_edge_advantage',lambda i:edge['depth_pro'][i]['f1']-edge['unidepth_v2'][i]['f1']),
        ('unidepth_edge_advantage',lambda i:edge['unidepth_v2'][i]['f1']-edge['depth_pro'][i]['f1']),
        ('low_object_recall',lambda i:-np.mean([edge[m][i]['object_edge_recall'] or 0 for m in LABELS])),
        ('false_edge_failure',lambda i:-np.mean([edge[m][i]['precision'] for m in LABELS])),
        ('large_residual_after_scale',lambda i:np.mean([images[i]['oracle_raw_median_clipped_abs_rel'],ud[i]['oracle_raw_median_clipped_abs_rel']])),
        ('large_native_metric_error',lambda i:np.mean([images[i]['primary_abs_rel'],ud[i]['primary_abs_rel']])),
    ]
    for reason,score in scores[:c['cases']['automatically_selected']]:
        ranking=sorted(images,key=lambda i:(-score(i),i))
        eligible=[i for i in ranking if reason!='low_object_recall' or all(edge[m][i]['object_edge_recall'] is not None for m in LABELS)]
        candidates=[i for i in eligible if i not in used and images[i]['scene'] not in scenes] or [i for i in eligible if i not in used]
        if not candidates:break
        i=candidates[0];chosen.append(dict(test_position=i,reason=reason));used.add(i);scenes.add(images[i]['scene'])
    return chosen


def edge_overlay(predicted,gt,valid):
    pd,gd=chebyshev_distance(predicted),chebyshev_distance(gt)
    rgb=np.zeros((*gt.shape,3),float)
    rgb[predicted&(gd<=3)]=[.2,.85,.35]
    rgb[predicted&(gd>3)]=[1.,.25,.15]
    rgb[gt&(pd>3)]=[.2,.5,1.]
    rgb[~valid]=[.5,.5,.5]
    return rgb


def local_figures(c,tables,output):
    from pipelines.part2.analyze_scale_boundary import ROOT,load_depth,dump_json
    choices=select_cases(c,tables);case_records=[]
    top,bottom,left,right=c['evaluation']['crop'];crop=np.s_[top:bottom,left:right]
    with NYUv2LabeledDataset(ROOT/c['dataset']['labeled_mat'],ROOT/c['dataset']['split_file']) as ds:
        for choice in choices:
            sample=ds.get_test_sample(choice['test_position']);stem=f'test_{sample.test_position:04d}_mat_{sample.matlab_index_1based:04d}'
            rgb=sample.rgb[crop];gt=sample.depth_m[crop]
            valid=np.isfinite(gt)&(gt>.001)&(gt<10)
            depths={m:load_depth(c,m,sample)[crop] for m in LABELS}
            labels=np.asarray(ds.file['labels'][sample.matlab_index_1based-1]).T[crop];instances=np.asarray(ds.file['instances'][sample.matlab_index_1based-1]).T[crop]
            contour=annotation_edges(labels,instances,valid)
            edge=depth_edge_map(gt,valid_mask=valid,log_threshold=np.log(c['boundary']['primary_ratio']))
            object_edge=edge&(maximum_filter(contour,size=2*c['boundary']['object_annotation_support_px']+1)>0)
            rgb_overlay=rgb.astype(float)/255;rgb_overlay[contour]=[1,.9,0]
            fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
            axes[0,0].imshow(rgb_overlay);axes[0,0].set_title('RGB + class/instance contour')
            im=axes[0,1].imshow(gt,cmap='turbo_r',vmin=.5,vmax=10);axes[0,1].set_title('GT depth (m)')
            for j,m in enumerate(LABELS):
                axes[0,j+2].imshow(depths[m],cmap='turbo_r',vmin=.5,vmax=10);axes[0,j+2].set_title(LABELS[m])
                err=axes[1,j].imshow(depths[m]-gt,cmap='coolwarm',vmin=-1,vmax=1);axes[1,j].set_title(LABELS[m]+' signed error (m)')
                pred_edge=depth_edge_map(depths[m],valid_mask=valid,log_threshold=np.log(c['boundary']['primary_ratio']))
                axes[1,j+2].imshow(edge_overlay(pred_edge,edge,valid));axes[1,j+2].set_title(LABELS[m]+' depth-edge matches')
            selected=[r for r in tables['profiles'] if r['model']=='depth_pro' and r['test_position']==sample.test_position]
            for r in selected:
                y,x=r['y'],r['x'];rad=c['cases']['roi_radius_px'];x0,x1=max(0,x-rad),min(gt.shape[1],x+rad+1);y0,y1=max(0,y-rad),min(gt.shape[0],y+rad+1)
                axes[0,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='cyan',lw=1));axes[0,0].text(x,y,str(r['profile_id']),color='cyan',fontsize=9)
            for ax in axes.flat:ax.axis('off')
            fig.colorbar(im,ax=list(axes[0,1:]),shrink=.7,label='m');fig.colorbar(err,ax=list(axes[1,:2]),shrink=.7,label='prediction - GT (m)')
            fig.suptitle(f'{stem} | {choice["reason"]}\nEdges: green=matched; red=unsupported prediction; blue=missed GT (3 px)')
            relative=f'cases/{stem}/overview.png';finish(fig,output/relative)
            local=[]
            for r in selected:
                y,x,rad=r['y'],r['x'],c['cases']['roi_radius_px'];x0,x1=max(0,x-rad),min(gt.shape[1],x+rad+1);y0,y1=max(0,y-rad),min(gt.shape[0],y+rad+1);roi=np.s_[y0:y1,x0:x1]
                fig,axes=plt.subplots(1,6,figsize=(20,3.7),layout='constrained')
                for ax,array,title in zip(axes[:4],[rgb,gt,depths['depth_pro'],depths['unidepth_v2']],['RGB','GT (m)','Depth Pro (m)','UniDepthV2-S (m)']):
                    if array.ndim==3:ax.imshow(array[roi])
                    else:ax.imshow(array[roi],cmap='turbo_r',vmin=.5,vmax=10)
                    if r['axis']=='x':ax.plot([x-r['radius_px']-x0,x+r['radius_px']-x0],[y-y0,y-y0],'c-',lw=1)
                    else:ax.plot([x-x0,x-x0],[y-r['radius_px']-y0,y+r['radius_px']-y0],'c-',lw=1)
                    ax.set_title(title);ax.axis('off')
                lines={'GT':extract_profile(gt,r),**{LABELS[m]:extract_profile(depths[m],r) for m in LABELS}}
                offsets=np.arange(-r['radius_px'],r['radius_px']+1)
                pair=[p for p in tables['profiles'] if p['sample_id']==stem and p['profile_id']==r['profile_id']]
                failures=[]
                for name,line in lines.items():
                    color='black' if name=='GT' else COLORS[next(m for m in LABELS if LABELS[m]==name)]
                    axes[4].plot(offsets,line,label=name,color=color)
                    status='measurable' if name=='GT' else next(p['status'] for p in pair if LABELS[p['model']]==name)
                    if status!='measurable':
                        failures.append(f'{name}: N/A ({status})')
                        continue
                    side=c['profiles']['plateau_samples'];a,b=np.median(line[:side]),np.median(line[-side:]);norm=(line-a)/(b-a) if abs(b-a)>1e-8 else np.full(len(line),np.nan)
                    axes[5].plot(offsets,norm,label=name,color=color)
                if failures:axes[5].text(.02,.02,'\n'.join(failures),transform=axes[5].transAxes,fontsize=7,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
                axes[4].set(title='Native depth cross-section',xlabel='offset (px)',ylabel='m');axes[4].legend(fontsize=7)
                axes[5].set(title='Own-plateau shape (diagnostic)',xlabel='offset (px)',ylabel='normalized within each profile',ylim=(-.5,1.5));axes[5].axhline(.1,c='gray',ls=':');axes[5].axhline(.9,c='gray',ls=':')
                for ax in axes[4:]:ax.grid(alpha=.2)
                pair=[p for p in tables['profiles'] if p['sample_id']==stem and p['profile_id']==r['profile_id']]
                fig.suptitle(f'{stem} | full-image center (x={x+left}, y={y+top}), {r["axis"]}-section | '+', '.join(f'{LABELS[p["model"]]}: {p["status"]}' for p in pair))
                path=f'cases/{stem}/profile_{r["profile_id"]}.png';finish(fig,output/path)
                local.append(dict(profile_id=r['profile_id'],figure=path,full_image_roi_xyxy=[x0+left,y0+top,x1+left,y1+top],full_image_center_xy=[x+left,y+top],axis=r['axis'],values_m={name:line.tolist() for name,line in lines.items()},measurements=pair))
            if not selected:
                ys,xs=np.where(object_edge)
                y,x=(int(np.median(ys)),int(np.median(xs))) if len(xs) else (gt.shape[0]//2,gt.shape[1]//2)
                rad=c['cases']['roi_radius_px'];roi=np.s_[max(0,y-rad):y+rad+1,max(0,x-rad):x+rad+1]
                fig,axes=plt.subplots(1,4,figsize=(12,3.4),layout='constrained')
                for ax,array,title in zip(axes,[rgb,gt,depths['depth_pro'],depths['unidepth_v2']],['RGB','GT','Depth Pro','UniDepthV2-S']):
                    ax.imshow(array[roi],**({} if array.ndim==3 else dict(cmap='turbo_r',vmin=.5,vmax=10)));ax.set_title(title);ax.axis('off')
                fig.suptitle(f'{stem}: no eligible stable GT cross-section; no width/offset assigned')
                finish(fig,output/f'cases/{stem}/object_zoom.png')
            record={**choice,'sample_id':stem,'scene':sample.scene,'overview':relative,'profiles':local,'gt_profile_count':len(selected)}
            dump_json(output/f'cases/{stem}/case.json',record);case_records.append(record)
    # Contact sheet uses only newly generated overviews.
    if case_records:
        thumbs=[]
        for r in case_records:
            with Image.open(output/r['overview']) as im:
                im=im.convert('RGB');im.thumbnail((720,400));thumbs.append(im.copy())
        cols=2;rows=(len(thumbs)+1)//2;sheet=Image.new('RGB',(cols*720,rows*425),'white')
        for i,im in enumerate(thumbs):sheet.paste(im,((i%cols)*720,(i//cols)*425))
        sheet.save(output/'figures/case_contact_sheet.jpg',quality=92)
    dump_json(output/'case_selection.json',{'selection_is_gt_or_metric_based_not_random':True,'cases':[{k:v for k,v in r.items() if k!='profiles'} for r in case_records]})
    return case_records


def write_report(c,summary,groups,cases,output):
    def fmt(x):return 'N/A' if x is None else f'{x:.8f}'
    s='''# NYUv2：原生米制深度的绝对尺度与物体边界\n\n本报告从保存的两条完整float32预测重新计算。标准654 test、固定crop和有效GT、逐图等权平均、正式`alignment:none`。不运行模型、不训练；GT辅助调整和剖面归一化只作诊断。区间为按scene成簇重采样、保持图像权重的10,000次bootstrap 95% CI；同场景的重复帧不当成完全独立样本。\n\n## 正式米制与裁剪前诊断\n\n| 模型 | AbsRel | RMSE m | δ1 | raw尺度比中位数 | raw超范围像素宏平均 | 整图log偏差平方占logMSE |\n|---|---:|---:|---:|---:|---:|---:|\n'''
    for m in LABELS:
        r=summary['models'][m]
        s+=f'| {LABELS[m]} | {fmt(r["primary_abs_rel"]["mean"])} | {fmt(r["primary_rmse"]["mean"])} | {fmt(r["primary_delta1"]["mean"])} | {fmt(r["scale_distribution"]["median"])} | {fmt(r["raw_out_of_range_fraction"]["mean"])} | {fmt(r["scale_distribution"]["global_log_bias_squared_share"])} |\n'
    s+='''\nraw范围诊断只统计固定crop内有效GT像素，不代表完整原图的分布。逐图`r=median(raw/GT)`；正式评分仍clip预测到[0.001,10]。log误差满足`mean(e²)=mean(e)²+mean((e-mean(e))²)`，这是未裁剪log误差的精确恒等式，不是AbsRel的因果分解。GT辅助轨道为`clip(raw/r)`；不作部署或主表成绩。\n\n## 距离误差\n\n| 模型 | 距离 m | 有支持图像数 | clipped AbsRel | clipped RMSE m | 有符号相对偏差 |\n|---|---:|---:|---:|---:|---:|\n'''
    for r in groups['depth_bins']:
        if r['track']=='native_clipped':s+=f'| {LABELS[r["model"]]} | {r["lower_m"]:g}–{r["upper_m"]:g} | {r["supported_image_count"]} | {fmt(r["abs_rel_mean"])} | {fmt(r["rmse_m_mean"])} | {fmt(r["relative_bias_mean"])} |\n'
    s+='''\n每个桶先逐图计算，再对有支持的图像平均；空桶为N/A，保留图像/像素覆盖。raw及oracle桶结果与区间见`summary/depth_bins.csv`。\n\n## 边界、物体轮廓与误差位置\n\n全部深度边缘采用四邻接深度比≥1.05/1.10/1.25，容差1/2/3 px为Chebyshev距离。定位距离单独用Euclidean距离、截断10 px，防止把两种距离混写。无预测边缘时GT距离计10 px；无GT边缘时相应GT统计N/A。\n\n物体支持边缘为GT深度边缘中、距类别/实例轮廓≤2 px的部分。标签0排除，实例号与类别号组合，不因实例号0直接排除。只报告该子集召回，不把不完整物体标注当作所有预测边缘的负例。其余米制主评价域保持原样。字段依据[NYUv2官方说明](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)。\n\n| 模型 | raw P@3 | raw R@3 | raw F1@3 | 物体支持边缘R@3 |\n|---|---:|---:|---:|---:|\n'''
    for r in groups['boundary']:
        if r['track']=='native_raw' and r['ratio']==c['boundary']['primary_ratio'] and r['tolerance_px']==3:s+=f'| {LABELS[r["model"]]} | {fmt(r["precision_mean"])} | {fmt(r["recall_mean"])} | {fmt(r["f1_mean"])} | {fmt(r["object_edge_recall_mean"])} |\n'
    s+='''\nraw/clipped分别计算，不将裁剪造成的边缘变化解释为模型结构变化。1/2/3 px及三阈值全部见`summary/boundary.csv`。互斥区域[0,1]、(1,3]、(3,5]、>5 px的像素比例和平方误差份额可逐图相加；边缘≤3与物体边缘≤3是另列的重叠区域，不能重复相加。误差份额为0的空区域保留0；误差均值在空域为N/A。\n\n## 局部剖面：位置、模糊与漏检\n\n只按GT和物体标注选择水平/垂直剖面，每图最多3条、半径20 px、两侧各5像素稳定、GT比值≥1.10、GT宽度≤6 px。不是所有曲线的连续精确法线，不覆盖所有细结构。预测各用自身两侧中位数归一化，仅比较形状；位置/宽度不是米制评价对齐。10/50/90交点必须唯一，弱对比、深度次序错误或多次交点均明确报告，几何量为N/A；不同模型的可测量覆盖不能忽略。先对一图内剖面平均，再对图像平均；配对比较只用双方都可测的相同剖面，完整失败计数另报。\n\n| 模型 | GT候选剖面 | 有剖面图像 | 可测比例 | 位置绝对误差 px | 相对GT宽度增量 px |\n|---|---:|---:|---:|---:|---:|\n'''
    for m in LABELS:
        r=summary['profiles'][m];s+=f'| {LABELS[m]} | {r["profile_count"]} | {r["images_with_profiles"]} | {fmt(r["measurable_fraction"])} | {fmt(r["absolute_midpoint_error_px"]["mean"])} | {fmt(r["width_excess_px"]["mean"])} |\n'
    s+='''\n具体失败状态、配对区间及覆盖见`summary.json`，不把无法测量的预测宽度写为0，不用条件子集的低误差替代总体质量。填充GT与语义轮廓存在对齐/物理表面局限，不能把每个RGB纹理当成真实深度边界。\n\n## 图像与复核材料\n\n'''
    for r in cases:s+=f'- [{r["sample_id"]}：{r["reason"]}]({r["overview"]})；可选GT剖面{r["gt_profile_count"]}条，放大图、实际坐标和原生深度曲线见同目录。\n'
    s+='''\n候选按固定示例和全体指标排序选择，并尽量使用不同scene；不是随机或无偏代表集，不用于选参。固定显示范围统一为0.5–10 m；显示饱和不修改深度数组，误差色图为±1 m。\n\n原始输入及源码SHA在`inputs.json`、`provenance/`；全部逐图数值在`tables/`，完整精度汇总和区间在`summary/`、`summary.json`。`manifest.json`绑定产物。原始基线、key_analysis及其执行记录不在本目录重写。\n'''
    (output/'report.md').write_text(s,encoding='utf-8')


def make_report(c,tables,groups,summary,output):
    overview_figures(c,tables,groups,summary,output)
    cases=local_figures(c,tables,output)
    write_report(c,summary,groups,cases,output)
