"""Full-population descriptive distributions; Matplotlib only, no seaborn."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
import json,warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ft2font import FT2Font
from matplotlib.ticker import ScalarFormatter
from PIL import Image
warnings.filterwarnings('error',message='Glyph .* missing from font')
FONT=ROOT/'fonts/NotoSansCJKsc-Regular.otf';fm.fontManager.addfont(str(FONT))
name=fm.FontProperties(fname=str(FONT)).get_name()
# Verify actual glyph availability; do not hide missing-glyph warnings.
charmap=FT2Font(str(FONT)).get_charmap()
assert all(ord(c) in charmap for c in '明确前景背景四类分布重建误差所有平均诊断分组样本量仅因规则未进入高分值方向物体性')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[name,'DejaVu Sans'],'axes.unicode_minus':False,'font.size':12,'axes.titlesize':15,'axes.labelsize':13,'legend.fontsize':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
z=np.load(ROOT/'paired_query_data.npz');group=z['group'];iou=z['max_iou_gt']
assert len(group)==911900
LABELS={0:'Known',1:'Unknown',2:'Background',3:'Ambiguous'}
# Stable default Matplotlib colors, reused across every method.
COLORS={0:'C0',1:'C1',2:'C2',3:'C3',4:'C0'}
STYLES={0:'-',1:'--',2:'-',3:'-.',4:'-'}
MODELS={'R0_S2':dict(key='r0_error',title='R0 S2 | trained73 | overall | B0 Query',caption='Encoder S2；四尺度误差算术平均；全部 B0 Query'),
        'R1_L6':dict(key='r1_error',title='R1 Decoder L6 | L2 | Mask 0 | seed91',caption='formal_20261112；full_error；全部 B0 Query'),
        'Gaussian':dict(key='gaussian',title='B0 Gaussian | 同一最终 Query 池',caption='Objectness 高分表示模型赋予较高物体性；不是重建偏差')}
allstats=[];ioustats=[];shift=[];diagnostics=[];catalog=[];curve_arrays={};density_audit=[];cdf_audit=[]
amb=group==3
bin_names=['[0, 0.1)','[0.1, 0.3)','[0.3, 0.5)','[0.5, 1] (difficult)']
bin_masks=[amb&(iou<.1),amb&(iou>=.1)&(iou<.3),amb&(iou>=.3)&(iou<.5),amb&(iou>=.5)]
assert np.array_equal(np.sum(bin_masks,axis=0),amb.astype(int))

def stat(values,model,group_name,mask):
    v=values[mask]
    row=dict(model=model,group=group_name,n=len(v),image_count=len(np.unique(z['image_id'][mask])),mean=float(v.mean()),std=float(v.std(ddof=1)),min=float(v.min()),p01=float(np.quantile(v,.01)),p05=float(np.quantile(v,.05)),p25=float(np.quantile(v,.25)),median=float(np.median(v)),p75=float(np.quantile(v,.75)),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),max=float(v.max()))
    row['iqr']=row['p75']-row['p25'];return row

def layout(title,subtitle,xlabel,ylabel='Probability Density'):
    fig,ax=plt.subplots(figsize=(10.8,5.9))
    fig.subplots_adjust(left=.105,right=.735,bottom=.18,top=.80)
    fig.text(.105,.955,title,ha='left',va='top',fontsize=15,weight='medium')
    fig.text(.105,.890,subtitle,ha='left',va='top',fontsize=10.5,color='.30')
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.16,linewidth=.6)
    ax.tick_params(direction='out')
    return fig,ax

def save(fig,ax,filename,footer,entry):
    if ax.get_ylabel()=='Probability Density':
        peak=max(float(np.max(line.get_ydata())) for line in ax.lines)
        ax.set_ylim(0,peak*1.08)
        assert all(np.max(line.get_ydata())<ax.get_ylim()[1] for line in ax.lines)
        entry['max_density']=peak
        entry['y_limit']=ax.get_ylim()[1]
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.0),frameon=False,borderaxespad=0,handlelength=2.6,labelspacing=.9)
    fig.text(.105,.045,footer,ha='left',va='bottom',fontsize=9,color='.33')
    path=ROOT/filename;fig.savefig(path,dpi=360)
    plt.close(fig)
    with Image.open(path) as im:assert min(im.info['dpi'])>=300 and im.size[0]>3000
    catalog.append(dict(file=filename,**entry))

for model,config in MODELS.items():
    values=z[config['key']];gaussian=model=='Gaussian'
    # Shared robust Silverman bandwidth: fixed for all groups and plots of this
    # method. Full population (no subsampling), same grid across groupings.
    robust=min(values.std(ddof=1),(np.quantile(values,.75)-np.quantile(values,.25))/1.349)
    bandwidth=float(.9*robust*len(values)**(-.2))
    upper=1. if gaussian else float(values.max()+4*bandwidth)
    edges=np.linspace(0,upper,4097);dx=edges[1]-edges[0];x=(edges[:-1]+edges[1:])/2
    xlabel='Gaussian Objectness' if gaussian else 'Reconstruction Error'
    cache={}
    def curve(mask,label):
        v=values[mask];counts,_=np.histogram(v,bins=edges)
        assert int(counts.sum())==len(v)
        d=gaussian_filter1d(counts.astype(float)/(len(v)*dx),bandwidth/dx,mode='reflect',truncate=4.)
        assert abs(d.sum()*dx-1)<1e-10
        density_audit.append(dict(model=model,group=label,n=len(v),integral=float(d.sum()*dx),bandwidth=bandwidth,grid_bins=4096,bin_width=float(dx),histogram_count=int(counts.sum()),min=float(v.min()),max=float(v.max())))
        curve_arrays[model+'__'+label]=d;return d
    curve_arrays[model+'__x']=x
    for code,label in LABELS.items():
        mask=group==code;allstats.append(stat(values,model,label,mask));cache[label]=curve(mask,label)
    fg=group<2;allstats.append(stat(values,model,'Foreground',fg));cache['Foreground']=curve(fg,'Foreground')
    bgmed=float(np.median(values[group==2]));fgmed=float(np.median(values[fg]));ambmed=float(np.median(values[amb]));bgiqr=float(np.quantile(values[group==2],.75)-np.quantile(values[group==2],.25))
    for label,mask in [('Foreground',fg),('Ambiguous',amb)]:
        diff=float(np.median(values[mask])-bgmed)
        shift.append(dict(model=model,comparison=label+'_minus_Background',median_difference=diff,background_iqr=bgiqr,median_difference_in_bg_iqr=diff/bgiqr))
    bg75=float(np.quantile(values[group==2],.75));fg25=float(np.quantile(values[fg],.25))
    for label,mask in [('Background',group==2),('Known',group==0),('Unknown',group==1),('Ambiguous',amb),('Foreground',fg)]:
        diagnostics.append(dict(model=model,group=label,reference_background_p75=bg75,fraction_le_background_p75=float(np.mean(values[mask]<=bg75)),reference_foreground_p25=fg25,fraction_lt_foreground_p25=float(np.mean(values[mask]<fg25))))
    subtitle=config['caption']+'\n四类 Query 的条件概率密度'
    fig,ax=layout(config['title'],subtitle,xlabel)
    for code,label in LABELS.items():ax.plot(x,cache[label],color=COLORS[code],ls=STYLES[code],lw=2,label=f'{label}\n(n={int((group==code).sum()):,})')
    ax.set_xlim(0,upper)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True));ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4))
    footer='每条曲线面积为 1；样本数见图例；完整取值范围；共同带宽的平滑密度。'
    save(fig,ax,f'{model}_四类分布.png',footer,dict(model=model,kind='four_groups',purpose='定位 Known、Unknown、Background、Ambiguous 的相对分布',bandwidth=bandwidth))
    fig,ax=layout(config['title'],config['caption']+'\n明确前景、Ambiguous 与明确背景',xlabel)
    for label,chinese,mask,code in [('Foreground','明确前景',fg,4),('Ambiguous','Ambiguous',amb,3),('Background','明确背景',group==2,2)]:
        ax.plot(x,cache[label],color=COLORS[code],ls=STYLES[code],lw=2,label=f'{chinese}\n(n={int(mask.sum()):,})')
    ax.set_xlim(0,upper);ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True));ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4))
    save(fig,ax,f'{model}_前景Ambiguous背景三组分布.png',footer+'\n明确前景 = Known + Unknown；三组均保留各自全部样本。',dict(model=model,kind='three_groups',purpose='简洁表达明确前景、Ambiguous、背景的位置和重叠',bandwidth=bandwidth))
    if gaussian:continue
    # C: four diagnostic IoU subgroups, with foreground/background references.
    fig,ax=layout(config['title'],config['caption']+'\nAmbiguous 按最大 IoU 诊断分组',xlabel)
    for j,(name,mask) in enumerate(zip(bin_names,bin_masks)):
        row=stat(values,model,name,mask);row['fraction_of_ambiguous']=float(mask.sum()/amb.sum());ioustats.append(row)
        density=curve(mask,'Ambiguous_IoU_'+str(j))
        label=['0 ≤ IoU < 0.1','0.1 ≤ IoU < 0.3','0.3 ≤ IoU < 0.5','IoU ≥ 0.5 (difficult)'][j]
        ax.plot(x,density,color=f'C{j}',lw=2,ls=['-','--','-.',':'][j],label=f'{label}\n(n={int(mask.sum()):,})')
    ax.plot(x,cache['Background'],color='.43',ls='--',lw=1.35,alpha=.85,label=f'Background 参照\n(n={int((group==2).sum()):,})')
    ax.plot(x,cache['Foreground'],color='.15',ls=':',lw=1.5,alpha=.85,label=f'明确前景参照\n(n={int(fg.sum()):,})')
    ax.set_xlim(0,upper);ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True));ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4))
    save(fig,ax,f'{model}_Ambiguous按IoU分组分布.png','每组面积为 1；同一方法内共用带宽；完整取值范围。\nIoU 仅用于诊断分组，不作为物体性强弱的严格标签；高 IoU 组均为 difficult。',dict(model=model,kind='iou_groups',purpose='检验 Ambiguous 内部异质性；观察随重叠变化的分布',bandwidth=bandwidth))
    # Exact empirical CDF: keep all duplicate multiplicities; no smoothing.
    for kind in ['four','iou']:
        fig,ax=layout(config['title'],config['caption']+'\n'+('四类 Query 累积分布（全样本）' if kind=='four' else 'Ambiguous IoU 子组累积分布（全样本）'),xlabel,'Cumulative Probability')
        seq=[(label,group==code,COLORS[code],STYLES[code]) for code,label in LABELS.items()] if kind=='four' else [(bin_names[j],mask,f'C{j}',['-','--','-.',':'][j]) for j,mask in enumerate(bin_masks)]
        for label,mask,color,ls in seq:
            unique,counts=np.unique(values[mask],return_counts=True);p=np.cumsum(counts)/counts.sum()
            xx=np.r_[0,unique,upper];yy=np.r_[0,p,1]
            ax.step(xx,yy,where='post',lw=1.8,color=color,ls=ls,label=f'{label}\n(n={int(mask.sum()):,})')
            cdf_audit.append(dict(model=model,kind=kind,group=label,n=int(mask.sum()),unique_values=len(unique),endpoint=float(p[-1])))
        ax.set_xlim(0,upper);ax.set_ylim(0,1.02);ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True));ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4))
        file=f'{model}_四类CDF.png' if kind=='four' else f'{model}_Ambiguous按IoU分组CDF.png'
        save(fig,ax,file,'经验 CDF，使用全部样本；不平滑、不截尾；同一误差值处越高表示低于该值的比例越大。',dict(model=model,kind='cdf_'+kind,purpose='核验分布曲线中的位置差异及长尾，避免只比较主峰'))

pd.DataFrame(allstats).to_csv(ROOT/'summary_statistics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(ioustats).to_csv(ROOT/'ambiguous_iou_group_stats.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(shift).to_csv(ROOT/'median_shift_diagnostics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(diagnostics).to_csv(ROOT/'distribution_mass_diagnostics.csv',index=False,encoding='utf-8-sig')
np.savez_compressed(ROOT/'density_curve_data.npz',**curve_arrays)
(ROOT/'plot_metadata.json').write_text(json.dumps(dict(method='full-population histogram-binned Gaussian KDE approximation with reflected boundaries',n_per_method=911900,bins=4096,bandwidth='0.9 * min(pooled sample std, pooled IQR/1.349) * pooled_N^(-1/5); shared within method',boundary='reflection at zero; Gaussian baseline also at one',range='full data range plus 4 bandwidths for reconstruction errors; Gaussian [0,1]',weighting='each query equally weighted within group; each group density integrates to one',font=str(FONT),font_family=name,dpi=360,figures=catalog,density_checks=density_audit,cdf_checks=cdf_audit),ensure_ascii=False,indent=2))
print('SUMMARY\n'+pd.DataFrame(allstats)[['model','group','n','mean','p25','median','p75']].to_string(index=False))
print('AMBIGUOUS IOU\n'+pd.DataFrame(ioustats)[['model','group','n','mean','p25','median','p75']].to_string(index=False))
print('MEDIAN SHIFTS\n'+pd.DataFrame(shift).to_string(index=False))
print('FIGURES',len(catalog),flush=True)
plan=ROOT/'task_plan.md';plan.write_text(plan.read_text().replace('- [ ] 计算描述统计与绘制 Matplotlib 分布曲线','- [x] 计算描述统计与绘制 Matplotlib 分布曲线').replace('计算描述统计与绘图。','图像检查与报告。'))
