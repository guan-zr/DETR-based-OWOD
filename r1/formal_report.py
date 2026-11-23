"""Evidence-checked R1 report and plots. All conclusions refer to frozen B0 Queries."""
import argparse,json,platform,sys,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from r1.reuse import write_json,file_sha256
from r1.formal_stats import main as analyze,SCORES

COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']

def table(df,columns=None):
    if columns is not None:df=df[columns]
    def value(v):
        if v is None or (isinstance(v,float) and not np.isfinite(v)):return '未定义'
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(value(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

def savefig(fig,path):
    fig.tight_layout();fig.savefig(str(path)+'.png',dpi=220,bbox_inches='tight');fig.savefig(str(path)+'.svg',bbox_inches='tight');plt.close(fig)

def plots(root):
    out=root/'analysis';figs=out/'figures';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.prop_cycle':plt.cycler(color=COLORS)})
    ci=pd.read_csv(out/'unknown_cluster_intervals.csv');metrics=pd.read_csv(out/'all_metrics.csv');coverage=pd.read_csv(out/'unknown_coverage.csv');paired=pd.read_csv(out/'paired_error_minus_baseline.csv');catalog=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,pool in zip(axes,['all','selected']):
        for score,label in [('raw_norm','Post-LN norm'),('pre_norm','Pre-LN norm'),('gaussian_objectness','B0 Gaussian'),('classifier_unknown','B0 unknown classifier')]:
            d=ci.loc[ci.run.str.startswith('diagnostic_')&(ci.pool==pool)&(ci.score==score)].sort_values('layer');x=d.layer.to_numpy();y=d.auroc.to_numpy();ax.errorbar(x,y,yerr=np.vstack([y-d.auroc_ci_low.to_numpy(),d.auroc_ci_high.to_numpy()-y]),marker='o',capsize=2,label=label)
        ax.axhline(.5,c='gray',ls='--',lw=.8);ax.set(xlabel='Decoder layer',ylabel='Unknown vs background AUROC',title=pool+' Query pool',xticks=range(1,7),ylim=(0,1));ax.legend(fontsize=7)
    savefig(fig,figs/'01_decoder_separability');catalog.append(('01_decoder_separability','各层 Query 简单统计/B0 分数的 Unknown 区分力；范数曲线为高值方向，反向AUROC见完整表；误差条为200次图像聚类95%区间，非训练seed不确定性。'))
    fig,axes=plt.subplots(1,2,figsize=(9,4.5))
    for ax,rep in zip(axes,['l2','raw']):
        matrix=[];labels=[]
        for mask in ['0.6','0','0.3']:
            for direction in ['higher','lower']:
                row=[]
                for layer in [1,3,6]:
                    d=metrics.loc[(metrics.run==f'{rep}_L{layer}_mask{mask}')&(metrics.pool=='all')&(metrics.score=='full_error')&(metrics.direction==direction)&(metrics.comparison=='unknown_vs_background')];row.append(float(d.auroc.iloc[0]))
                matrix.append(row);labels.append(f'mask {mask} / {direction}')
        matrix=np.asarray(matrix);im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set(xticks=range(3),xticklabels=['L1','L3','L6'],yticks=range(6),yticklabels=labels,title=rep+' full MSE: both signs')
        for i in range(6):
            for j in range(3):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]<.65 else 'black')
    fig.colorbar(im,ax=list(axes),label='Unknown AUROC',fraction=.03,pad=.03);fig.subplots_adjust(wspace=.65,right=.87);fig.savefig(figs/'02_mask_normalization.png',dpi=220,bbox_inches='tight');fig.savefig(figs/'02_mask_normalization.svg',bbox_inches='tight');plt.close(fig);catalog.append(('02_mask_normalization','完整层×mask×表示及误差方向；两方向同时展示，不选择更好方向。AUROC可跨层比较，MSE量纲不用于层优劣排名。'))
    fig,ax=plt.subplots(figsize=(7,4))
    for k in ['03','05','07']:ax.plot(coverage.layer,coverage['coverage_'+k+'_fraction'],marker='o',label='IoU '+str(int(k)/10))
    ax.plot(coverage.layer,coverage.final_unknown_geometric_hit_fraction,ls='--',label='Final unknown geometry');ax.set(xlabel='Decoder layer',ylabel='Fraction of Unknown GT',ylim=(0,1),xticks=range(1,7));ax.legend();savefig(fig,figs/'03_unknown_coverage');catalog.append(('03_unknown_coverage','每层全部 Query 的 Unknown GT 几何覆盖，分母为非difficult Unknown GT；最终 Unknown 几何命中不等于官方 U-Recall。'))
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    names=[f'{rep}_L{layer}_mask{mask}' for rep in ['l2','raw'] for mask in ['0.6','0','0.3'] for layer in [1,3,6]]
    for ax,direction in zip(axes,['higher','lower']):
        d=paired.loc[(paired.pool=='all')&(paired.baseline=='gaussian_objectness')&(paired.direction==direction)].set_index('run').loc[names];y=d.auroc_difference.to_numpy();ax.errorbar(range(18),y,yerr=np.maximum(0,np.vstack([y-d.auroc_ci_low.to_numpy(),d.auroc_ci_high.to_numpy()-y])),fmt='o',capsize=3);ax.axhline(0,c='gray',ls='--');ax.set(ylabel='Error - Gaussian AUROC',title=direction+' error direction: pointwise paired 95% image CI')
    axes[-1].set_xticks(range(18),names,rotation=65,ha='right',fontsize=7);savefig(fig,figs/'04_paired_difference');catalog.append(('04_paired_difference','同一 Query 上的误差−Gaussian AUROC差及成对图像区间；点态区间不是多重比较校正后的显著性。'))
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for row,rep in enumerate(['l2','raw']):
        for ax,layer in zip(axes[row],[1,3,6]):
            folder=out/'arrays'/f'{rep}_L{layer}_mask0.6';m=np.load(folder/'metadata.npy',mmap_mode='r');scores=np.load(folder/'scores.npy',mmap_mode='r')[:,5]
            for group,index,color in [('Known',0,'#009E73'),('Unknown',1,'#D55E00'),('Background',2,'#0072B2')]:
                v=np.asarray(scores[m[:,1]==index]);ax.hist(v,bins=50,density=True,histtype='step',label=f'{group} n={len(v)}',color=color)
            ax.set(title=f'{rep} L{layer}, mask 0.6',xlabel='Full reconstruction MSE',ylabel='Density');ax.legend(fontsize=6)
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            if rep=='l2':ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)
    savefig(fig,figs/'05_reconstruction_distributions');catalog.append(('05_reconstruction_distributions','mask0.6主实验的完整三组误差分布，不截取有利目标；raw/l2轴量纲不同，不能直接跨图用MSE大小判断性能。'))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,rep in zip(axes,['l2','raw']):
        for mask in ['0.6','0','0.3']:
            for layer,style in zip([1,3,6],['-','--',':']):
                rows=[json.loads(l) for l in (root/f'{rep}_L{layer}_mask{mask}'/'train.jsonl').read_text().splitlines()];d=pd.DataFrame(rows).drop_duplicates('step',keep='last');ax.plot(d.step,d.full_mse.rolling(50,min_periods=1).mean(),ls=style,lw=.9,label=f'L{layer} m{mask}')
        ax.set(xlabel='Optimizer step',ylabel='Full MSE, rolling mean 50',title=rep+' independent AEs',yscale='log');ax.legend(fontsize=6,ncol=3)
    savefig(fig,figs/'06_training');catalog.append(('06_training','全部18个AE的训练轨迹与预定2000步预算；平滑仅用于显示，不能根据测试指标选择checkpoint。'))
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    cond=pd.read_csv(out/'conditional_metrics.csv')
    for ax,layer in zip(axes,[1,3,6]):
        d=cond.loc[(cond.run==f'l2_L{layer}_mask0.6')&(cond.pool=='all')&(cond.gaussian_band!='bottom25')]
        for score,direction,label in [('gaussian_objectness','higher','Gaussian'),('full_error','higher','Higher MSE'),('full_error','lower','Lower MSE')]:
            a=d.loc[(d.score==score)&(d.direction==direction)].copy();a['bin']=a.gaussian_band.astype(int);a=a.sort_values('bin');ax.plot(a.bin,a.auroc,marker='o',label=label)
        ax.axhline(.5,c='gray',ls='--');ax.set(xlabel='Gaussian quantile bin',title=f'L{layer} l2 mask0.6',ylim=(0,1));ax.legend(fontsize=7)
    axes[0].set_ylabel('Unknown vs background AUROC');savefig(fig,figs/'07_conditional_information');catalog.append(('07_conditional_information','固定Gaussian分位层内的区分力；切点只用全Query分数计算。空层保留未定义，零分并列可导致非均匀分箱；此图不证明因果或融合增益。'))
    for figure,score,title in [('08_postLN_norm','raw_norm','Post-LN Query norm'),('09_preLN_norm','pre_norm','Pre-LN Query norm'),('10_gaussian_distribution','gaussian_objectness','B0 Gaussian objectness')]:
        fig,axes=plt.subplots(2,3,figsize=(11,6))
        for ax,layer in zip(axes.flat,range(1,7)):
            folder=out/'arrays'/f'diagnostic_L{layer}';m=np.load(folder/'metadata.npy',mmap_mode='r');values=np.load(folder/'scores.npy',mmap_mode='r')[:,SCORES.index(score)]
            for group,index,color in [('Known',0,'#009E73'),('Unknown',1,'#D55E00'),('Background',2,'#0072B2')]:
                v=np.asarray(values[m[:,1]==index])
                if len(v):ax.hist(v,bins=50,density=True,histtype='step',label=f'{group} n={len(v)}',color=color)
            ax.set(title=f'L{layer}: {title}',xlabel=title,ylabel='Density');ax.legend(fontsize=6)
        savefig(fig,figs/figure);catalog.append((figure,'R1-0六层全部原始Query的Known/Unknown/Background完整经验分布；不含Ambiguous，图例列出真实样本数；范数接近常数不表示高维表示没有信息。'))
    return catalog

def verify_protected(root):
    old=json.loads((root/'protected_file_stats.json').read_text());changed=[]
    for name,expected in old.items():
        p=Path(name)
        if not p.is_file():changed.append(dict(path=name,reason='missing'));continue
        s=p.stat()
        if [s.st_size,s.st_mtime_ns]!=expected:changed.append(dict(path=name,reason='size_or_mtime_changed'))
    reviewed=json.loads((root/'reviewed_external_bookkeeping.json').read_text()) if (root/'reviewed_external_bookkeeping.json').exists() else {}
    external=[];unexpected=[]
    for row in changed:
        path=Path(row['path']);record=reviewed.get(str(path))
        if record and path.is_file() and file_sha256(path)==record['sha256']:external.append(dict(row,review=record))
        else:unexpected.append(row)
    write_json(root/'protected_artifact_check.json',dict(passed=not unexpected,critical_inputs_unchanged=not unexpected,checked_files=len(old),reviewed_external_bookkeeping_updates=external,unexpected_changes=unexpected))
    if unexpected:raise RuntimeError('Protected artifact changed; report requires investigation')

def main(root):
    root=Path(root);analyze(root);a=root/'analysis';catalog=plots(root)
    meta=json.loads((root/'evaluation_cache/manifest.json').read_text());diag=json.loads((root/'diagnostic/summary.json').read_text());metrics=pd.read_csv(a/'all_metrics.csv');ci=pd.read_csv(a/'unknown_cluster_intervals.csv');paired=pd.read_csv(a/'paired_error_minus_baseline.csv');coverage=pd.read_csv(a/'unknown_coverage.csv');conditional=pd.read_csv(a/'conditional_metrics.csv');counts=pd.read_csv(a/'sample_counts.csv');train=pd.read_csv(a/'training_summary.csv')
    checkpoint=meta['baseline']['path'];sha=file_sha256(checkpoint)
    if sha!=meta['baseline']['sha256']:raise RuntimeError('Baseline checkpoint changed')
    verify_protected(root)
    write_json(root/'environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),baseline_sha256=sha))
    diagtable=[]
    for layer in range(1,7):
        row=dict(layer=layer)
        for score in ['raw_norm','pre_norm','gaussian_objectness','classifier_unknown']:
            d=metrics.loc[(metrics.run=='diagnostic')&(metrics.layer==layer)&(metrics.pool=='all')&(metrics.score==score)&(metrics.comparison=='unknown_vs_background')];row[score+'_AUROC']=float(d.auroc.iloc[0]);row[score+'_AP']=float(d.auprc_average_precision.iloc[0])
        diagtable.append(row)
    dt=pd.DataFrame(diagtable);dt['raw_norm_lower_AUROC']=1-dt.raw_norm_AUROC;dt['pre_norm_lower_AUROC']=1-dt.pre_norm_AUROC;dt.to_csv(a/'R1_0_layer_summary.csv',index=False)
    mainrows=[]
    for rep in ['l2','raw']:
        for mask in ['0.6','0','0.3']:
            for layer in [1,3,6]:
                run=f'{rep}_L{layer}_mask{mask}'
                for direction in ['higher','lower']:
                    row=dict(representation=rep,mask=mask,layer=layer,direction=direction)
                    for comparison,name in [('known_vs_background','Known'),('unknown_vs_background','Unknown'),('foreground_vs_background','Foreground')]:
                        v=metrics.loc[(metrics.run==run)&(metrics.pool=='all')&(metrics.score=='full_error')&(metrics.direction==direction)&(metrics.comparison==comparison)].iloc[0];row[name+'_AUROC']=v.auroc;row[name+'_AP']=v.auprc_average_precision
                    mainrows.append(row)
    mt=pd.DataFrame(mainrows);mt.to_csv(a/'R1_1_complete_comparison.csv',index=False)
    selected=metrics.loc[(metrics.run!='diagnostic')&(metrics.pool=='selected')&(metrics.score=='full_error')];selected.to_csv(a/'R1_1_selected_comparison.csv',index=False)
    selected_main=selected.pivot_table(index=['run','layer','direction'],columns='comparison',values=['auroc','auprc_average_precision']).reset_index()
    selected_main.columns=['_'.join(str(v) for v in col if v) if isinstance(col,tuple) else col for col in selected_main.columns]
    selected_main.to_csv(a/'R1_1_selected_summary.csv',index=False)
    primary=paired.loc[(paired.pool=='all')&(paired.baseline=='gaussian_objectness')];primarycols=['run','direction','auroc_difference','auroc_ci_low','auroc_ci_high','auroc_holm_p'];delta_table=table(primary[primarycols])
    low=conditional.loc[(conditional.pool=='all')&(conditional.gaussian_band=='bottom25')&(conditional.score.isin(['gaussian_objectness','full_error']))];low.to_csv(a/'gaussian_bottom25_comparison.csv',index=False)
    # Every narrative comparison is generated from the complete predeclared family.
    descriptions=[]
    for layer in [1,3,6]:
        text=[]
        for direction in ['higher','lower']:
            r=primary.loc[(primary.run==f'l2_L{layer}_mask0.6')&(primary.direction==direction)].iloc[0]
            status='区间全高于0' if r.auroc_ci_low>0 else ('区间全低于0' if r.auroc_ci_high<0 else '区间跨0')
            text.append(f"{direction}误差方向 ΔAUROC={r.auroc_difference:.6f}，95%区间[{r.auroc_ci_low:.6f},{r.auroc_ci_high:.6f}]，{status}")
            point=mt.loc[(mt.representation=='l2')&(mt['mask']=='0.6')&(mt.layer==layer)&(mt.direction==direction)].iloc[0]
            agreement='Known与Unknown在此方向的点估计均高于0.5，仍须限定为固定B0候选诊断' if point.Known_AUROC>.5 and point.Unknown_AUROC>.5 else ('Known/Unknown的方向不一致，不能认定统一前景排序' if (point.Known_AUROC-.5)*(point.Unknown_AUROC-.5)<0 else '此方向未同时区分Known与Unknown前景')
            text.append(f'同方向Known AUROC={point.Known_AUROC:.6f}、Unknown={point.Unknown_AUROC:.6f}、合并Foreground={point.Foreground_AUROC:.6f}；{agreement}')
        descriptions.append(f"- L{layer}、l2、mask0.6、全部Query："+'；'.join(text)+'。')
    sensitivity=[]
    for (rep,layer,direction),group in mt.groupby(['representation','layer','direction']):
        sensitivity.append(dict(representation=rep,layer=layer,direction=direction,unknown_AUROC_min=float(group.Unknown_AUROC.min()),unknown_AUROC_max=float(group.Unknown_AUROC.max()),mask_range=float(group.Unknown_AUROC.max()-group.Unknown_AUROC.min())))
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(a/'mask_sensitivity.csv',index=False)
    first=dt.loc[dt.layer==1].iloc[0];last=dt.loc[dt.layer==6].iloc[0]
    depth_text=f"Gaussian Unknown AUROC从L1的{first.gaussian_objectness_AUROC:.6f}变为L6的{last.gaussian_objectness_AUROC:.6f}（变化{last.gaussian_objectness_AUROC-first.gaussian_objectness_AUROC:+.6f}）；pre-LN范数从{first.pre_norm_AUROC:.6f}变为{last.pre_norm_AUROC:.6f}。层间候选框也在变化，这些是整体候选诊断的描述，不构成同框表示的因果比较。"
    gaussian_sequence=dt.sort_values('layer').gaussian_objectness_AUROC.to_numpy();monotonic=bool((np.diff(gaussian_sequence)>=0).all());depth_text+=' 六层Gaussian AUROC序列为'+', '.join(f'{v:.6f}' for v in gaussian_sequence)+('，呈单调非降。' if monotonic else '，并非随层深单调改善。')
    coverage_text=f"IoU≥0.5的Unknown GT覆盖比例从L1的{coverage.loc[coverage.layer==1,'coverage_05_fraction'].iloc[0]:.6f}变为L6的{coverage.loc[coverage.layer==6,'coverage_05_fraction'].iloc[0]:.6f}；低阈值/高阈值与最终Unknown命中完整列出，不择优采用单个阈值。"
    conditional_notes=[]
    for rep in ['l2','raw']:
        for layer in [1,3,6]:
            for direction in ['higher','lower']:
                d=conditional.loc[(conditional.run==f'{rep}_L{layer}_mask0.6')&(conditional.pool=='all')&(conditional.gaussian_band!='bottom25')&(conditional.score=='full_error')&(conditional.direction==direction)]
                valid=d.auroc.notna();conditional_notes.append(f"- {rep} L{layer}、mask0.6、{direction}方向：{int(valid.sum())}个有效Gaussian分位层中，{int((d.loc[valid,'auroc']>.5).sum())}层误差AUROC>0.5。这是完整探索性计数，不表示这些层达到统计显著。")
    feature_notes=[]
    for _,r in dt.iterrows():feature_notes.append(f"L{int(r.layer)} 的 Gaussian Unknown AUROC={r.gaussian_objectness_AUROC:.6f}，pre-LN范数={r.pre_norm_AUROC:.6f}，post-LN范数={r.raw_norm_AUROC:.6f}；低范数方向AUROC为1−上述高值方向（表中同时列出），这仅描述排序可分性，不据此选定得分方向；AP不能用1−AP推导。范数结果不等价于穷尽高维特征可分性。")
    interpretation='\n'.join(descriptions)
    analysis_text='# R1 严格分析记录\n\n主比较：在同一层、同一Query候选与同一GT分组中比较full reconstruction MSE（两个方向均保留）与B0 Gaussian的Unknown/Background AUROC/AP。次比较包括classifier_unknown、pre/post-LN范数、all/selected、mask与raw/l2。\n\n'+interpretation+'\n\n'+table(dt)+'\n\n## Claim Candidates\n\n- Claim：冻结B0 Query的既有范数、分数和重建误差在指定候选池中具有表中区分能力。\n  - Source evidence：all_metrics.csv、unknown_cluster_intervals.csv、paired_error_minus_baseline.csv。\n  - Allowed wording：在固定B0、seed91和当前候选及标注规则下观察到指定排序信号。\n  - Forbidden stronger wording：学得独立于Known监督的通用Objectness，或已改善官方检测指标。\n  - Uncertainty：训练单seed；背景依赖现有标注；点态图像区间不包含训练seed/数据域变化。\n  - Next check：另行预登记多seed及独立检测评价。\n  - Decision：保留为机制诊断。\n\n- Claim：条件排序表可检查误差是否只是复现Gaussian排序。\n  - Source evidence：conditional_metrics.csv与score_correlations.csv；所有方法同Query。\n  - Allowed wording：仅当固定Gaussian分层内仍出现区分时，视为局部剩余排序信息的证据。\n  - Forbidden stronger wording：条件区分等于独立因果信息、融合收益或无监督检测改善。\n  - Uncertainty：分箱较粗、并列分数和样本保留差异；没有拟合融合器或反事实实验。\n  - Next check：另行设计独立验证集上的增量预测实验。\n  - Decision：弱化为诊断证据。\n'
    (a/'analysis-report.md').write_text(analysis_text)
    (a/'stats-appendix.md').write_text('# 统计附录\n\n- 单位：9119个唯一评价图像，Query嵌套于图像，不能把Query当独立重复。\n- 18个配置均单一训练seed91，掩码设置的变化不是seed重复；每配置2000 optimizer steps，训练图像序列完全一致。训练/测试ID无交集。\n- 主指标：Unknown/Background AUROC及阶梯average precision；完整导出Known和Known+Unknown/Background及正例比例。AP随候选池比例变化，all与selected不可直接归因比较。\n- 200次图像聚类bootstrap，seed91；所有配置共享相同图像权重，重采样包含每图全部Query。点态95% percentile区间；不采用正态假设，不含训练seed不确定性。\n- 差值为同Query、同图像抽样下的成对AUROC/AP差；简单统计及B0分数逐项一致才允许配对。\n- 探索性检验为以观察差值居中的bootstrap残差双侧尾概率，使用+1有限抽样修正，不是精确随机化检验。Holm校正覆盖所有表示、层、掩码、候选池、两个方向及四个baseline的完整比较族，各指标单独校正。200次抽样的最小未校正p=1/201，不能用此精度承诺严格校正后的显著性。所有强度判断以效应大小和区间为主，禁止择优方向。\n- Gaussian分箱及bottom25切点从池内全部Query分数计算，不使用GT标签；并列分数会使实际保留比例不等于25%，实际比例及切点全部输出。条件表为探索性，不报告未校正的显著性。\n- mask=0无masked维度，masked MSE未定义；所有配置仍有full MSE。未定义指标不是0。\n- 只有一种训练seed，不能就层/掩码/表示作训练随机性总体显著性结论。无额外参数检验、无训练结果选择、无测试阈值拟合。\n')
    (a/'figure-catalog.md').write_text('# 图表目录与解释边界\n\n'+'\n\n'.join(f'## {name}\n\n- 文件：figures/{name}.png 与 .svg。\n- 目的与解释：{desc}\n- 数据：all_metrics.csv、unknown_cluster_intervals.csv、paired_error_minus_baseline.csv、unknown_coverage.csv或训练日志；图1与4使用bootstrap区间，其余为完整配置/经验分布描述，无伪造误差条。\n- 检查：候选池、方向、表示和层均须对应；不能从范数或重建单个统计推断全高维信息或因果。' for name,desc in catalog))
    report='# R1 / Round 1 / Decoder Query机制验证 / 2026-11-12\n\n'
    report+='## 执行摘要\n\n完成六层R1-0与18个R1-1配置（L1/3/6 × mask0.6/0/0.3 × l2/raw）。全部使用冻结B0 checkpoint0040和相同9119唯一测试图像，seed91、batch1、2000 optimizer steps。R1不引入R0重建模型或融合模块。\n\n'+interpretation+'\n\n这些是固定B0表示上的排序诊断，不能据此宣称独立于Known分类监督的通用Objectness，也未验证端到端检测增益。\n\n'
    report+='## 实验身份与判定问题\n\n本轮是R1首次正式CUDA机制验证。R1-0检查Decoder中的简单统计与既有分数；R1-1检查独立共享MLP的无标签Query重建是否给出不同的前景排序。主问题是相同Query上误差相对B0 Gaussian的Unknown/Background区分差异及条件排序，而非选择表现最好的层、掩码或符号。\n\n'
    report+=f'## 环境、权重与协议\n\n- B0 checkpoint：{checkpoint}\n- SHA256：`{sha}`；epoch40完整，原始args见B0_original_args.json。加载strict=True并使用原官方推理backend。\n- 数据：M-OWODB Task1，VOC20 Known、其余60 Unknown；训练16551唯一图，测试10246行去重为9119图，训练测试ID交集为0。\n- B0结构：6层Decoder × 100 Query × 256维；own-layer特征/框/logits/Objectness对应经真实接口检查，最终筛选与原postprocessor完全一致。\n- Query raw为Decoder post-norm3表示；pre_layernorm为norm3输入；l2逐Query归一化。模型已学习B0的Known监督，raw范数也受LayerNorm约束。\n- 正例：与最近可用GT的IoU≥0.5且非difficult；背景：与包括Unknown/difficult在内全部可用GT交集面积为0。有交集而未达正例规则的框单列Ambiguous，非法框不静默跳过。0.1仅保留旧IoU规则审计参数，不是主背景标准。\n- 全部原始Query与Top100 query/class pair去重后selected独立报告；中间层selected为相同规则模拟，第6层为实际最终筛选。没有人为构造Query的GT-reference表示，GT仅用于离线分组/覆盖。\n- AE：512→128→32→128→256，GELU，107040参数，输入拼接masked Query与二值mask；全维MSE，AdamW lr1e-4/wd1e-4。2000步、batch1、seed91，最终步checkpoint；误差评价4次固定mask均值。\n- 先mask0.6，再0与0.3；raw是原有配对归一化对照，完整18配置均报告。所有AE只共享冻结B0前向，各自初始化、优化、权重及预算独立；训练不读取XML。\n- CUDA smoke覆盖18组合，各3步，原模型状态SHA前后相同，参数无梯度；CPU26项回归通过。旧缓存只发现合成CPU样例，未使用；本轮新建全六层缓存，每图SHA校验及原子提交。全9119图的原始Gaussian距离与logits有限，六层概率归零比例均为0（cache_numeric_audit.json），低Gaussian区分度不能解释为概率下溢。\n- R1准备期依赖哈希过期已审查，runtime变化仅B0验收和backend；旧清单、源码及diff均保留。另补齐前景合并指标、非有限值失败检查、进度及恢复，不改变模型/损失/分组/阈值。\n\n'
    report+='## R1-0：六层表示与Unknown候选\n\n'+depth_text+'\n\n'+coverage_text+'\n\n'+table(dt)+'\n'+table(coverage)+'\n'+'\n'.join(feature_notes)+'\n\n上述覆盖为每层全部100 Query对非difficult Unknown GT的几何覆盖；最终Unknown几何命中另列，不等同官方匹配和重复测试条目下的U-Recall。中间层框与最终层框不同，因此层间变化不能仅归因为同框特征变得更好。\n\n完整原始/selected数量：\n\n'+table(counts.loc[counts.run=='diagnostic'])+'\n背景旧规则迁移与包含风险见diagnostic/grouping_audit.json及containment_audit.json。零交集背景排除GT内部小框，但现有80类标注不穷尽所有真实物体，不能称为无噪声真实背景。\n\n'
    report+='## R1-1：完整层、掩码与方向比较\n\n下表为全部Query的full MSE；两个方向均报告，不用跨层MSE绝对大小排名。masked-only误差、selected池、样本数与分布在完整CSV中。mask0的masked MSE未定义。\n\n'+table(mt)+'\n\nselected池完整比较（L1/L3为相同规则模拟，L6为真实最终pair去重Query；AP不能脱离各池正例比例比较）：\n\n'+table(selected_main)+'\n\n18个配置预算与训练轨迹：\n\n'+table(train)+'\n\n掩码敏感性（同seed的预定设置范围）：\n\n'+table(sensitivity)+'\n\nraw−l2及mask0.6−控制的成对差值区间见analysis/configuration_pair_differences.csv；均为探索性点态图像区间。这里只描述掩码和表示敏感性；单seed91不能评价训练seed稳定性。raw/l2改变目标量纲及几何，不能把误差数值变化直接解释为Objectness增强。\n\n'
    report+='## 与B0分数的成对比较及额外信息\n\n同一层、Query、GT组及图像bootstrap上比较full error与Gaussian，差值为error−Gaussian；没有新增融合器，也没有按照Unknown标签调整方向。\n\n'+delta_table+'\n\n所有classifier、pre/post-LN范数对照在analysis/paired_error_minus_baseline.csv；Pearson/Spearman在score_correlations.csv。严格CSV相关系数限于Known/Unknown/Background三组共同有效候选；各原始summary.json的相关系数包含全部有限Query（含Ambiguous），两者范围不同，不能混用。低相关不等于额外有用信息，高相关也不能单独证明冗余。\n\nGaussian底部区间结果（切点仅用全Query分数，实际并列保留率单列）：\n\n'+table(low,['run','score','direction','positive_count','negative_count','auroc','auprc_average_precision'])+'\n\n'+'\n'.join(conditional_notes)+'\n\n只有在固定Gaussian分数区间内仍区分Unknown/Background时，才能说观察到局部剩余排序信号；这仍不是独立因果信息或已证实的融合收益。原始Gaussian在OWOBJ推理中并未直接相乘，低分不构成漏检因果证明。完整10分位层包含空层与并列分数情况，不静默移除。\n\n'
    topk=pd.read_csv(a/'topk_softlabel_audit.csv');top10=topk.loc[(topk.pool=='all')&(topk.k==10)]
    report+='## 重建误差作为Query Objectness软标签的证据边界\n\n本轮检验的是连续误差的前景排序、分布、同候选比较及预定TopK组成，没有把MSE校准为P(object)，没有用它监督新的检测器，也没有验证端到端收益。MSE尺度随层/表示/掩码而变；AUROC较高不代表每个Query的软标签数值可靠。\n\n原协议固定K=10/50/100，两个误差方向均保留。下表为每图Top10汇总；Known+Unknown比例只指IoU≥0.5的已标注前景，Ambiguous单列，不能当作真背景或真前景。完整all/selected及三个K见analysis/topk_softlabel_audit.csv。这里没有选取最优K或阈值。\n\n'+table(top10,['run','direction','effective_k','error_labeled_foreground_fraction','error_background_fraction','error_ambiguous_fraction','gaussian_background_fraction'])+'\n\n因此，即使误差与Objectness存在排序关联，也只能称为软标签候选信号；概率校准、跨类别/数据域稳定性及实际监督效果仍未验证，不能提前宣称可靠软标签已经成立。\n\n'
    report+='## 统计验证\n\n'+(a/'stats-appendix.md').read_text().split('\n\n',1)[1]+'\n\n'
    report+='## 图表逐项解释\n\n'
    for name,desc in catalog:report+=f'![{name}](analysis/figures/{name}.png)\n\n{desc}\n\n'
    report+='## 负结果、失败记录与限制\n\n- 机器在首次提取期间关机，用户确认后12:16恢复，从6509张已提交缓存逐图SHA复核继续；两张旧图重提取后六层特征/框/分数/筛选索引逐项bitwise一致（restart_parity.json）。旧attempt保留为interrupted、退出码未知。最终指标仅接受完整9119张，不能把中断记为成功。记录见pipeline/machine_restart_20261112.json。\n- 原R1依赖门禁因R0已验收代码的哈希变化而停止，审查diff后只改R1清单。缓存加SHA后的一个旧测试错误文本断言已适配；原有23项及3项新增恢复/统计测试共26/26通过。保留首次失败日志，不把失败算入有效实验。\n- 单训练seed、单基线、单数据域，重建有效不意味着摆脱Known监督；AUROC/AP仅度量已有候选排序。\n- LayerNorm后的范数可能近常数，范数诊断不能代表全部高维信息；本轮没有训练标签线性探测器。\n- 没有GT-reference Query特征可靠定义，不虚构该结果；没有无GT新候选生成或官方检测增益实验。\n- 背景相对现有标注；Ambiguous从二分类指标排除但计数完整保留。不同池AP受正例比例影响。\n- 条件分箱是探索性，不能替代独立验证集或反事实实验；不按有利分数方向包装创新。\n\n'
    report+='## 支持、不支持与待验证假设\n\n可靠事实是六层的逐项计数、覆盖、固定分数/误差区分、成对差值和本轮18配置的敏感性。支持的解释严格限定于冻结B0的Query机制诊断与无标签重建可行性。\n\n不能据此宣称通用Objectness、摆脱Known监督、提升官方U-Recall、独立于分类器的因果信号或对未见数据域泛化。可进一步验证的假设是条件区间内的剩余排序能否在另行预登记的多seed、独立验证集及真实检测评估中复现；本次不自动启动这些额外实验。\n\n'
    report+='## 工件与可复现索引\n\n- 外层阶段命令计划：pipeline/commands.json；实际PID/退出码/日志：pipeline/stages/、pipeline/logs/。并发CPU评价命令及结果另见每配置evaluation_process.json和各bank的evaluation_group_complete.json，外层随后复核并复用。\n- checkpoint原配置：B0_original_args.json；本轮全部配置：configs/。\n- 真实接口与smoke：real_interface.json、cuda_smoke/complete.json；CPU回归：cpu_regression_final.log。\n- 六层新缓存：evaluation_cache/manifest.json、index.jsonl及逐图NPZ/SHA。\n- R1-0：diagnostic/summary.json、queries.csv.gz、unknown_cases.csv.gz、grouping_audit.json。\n- R1-1：每个l2/raw_L*_mask*目录内checkpoint.pth、run.json、train.jsonl、evaluation/。\n- 严格分析：analysis/analysis-report.md、stats-appendix.md、figure-catalog.md、all_metrics.csv、unknown_cluster_intervals.csv、paired_error_minus_baseline.csv、conditional_metrics.csv、figures/。\n- 源码：source_snapshot/与source_manifest.json；每阶段pipeline/sources/记录实际模块哈希，报告源在生成阶段另存。原R1源码在original_sources/。\n- 保护核验：protected_artifact_check.json；B0/R0关键源码、checkpoint、数据集及环境文件大小/mtime未改变，B0权重SHA复核一致。两份R0进度笔记11:46的外部收尾更新经只读审查单列reviewed_external_bookkeeping.json，本轮未写R0。\n- 无git commit、无删除旧数据、无关机动作；未尝试Obsidian写回。\n\n## 后续操作决定\n\n保留本轮全部正负结果及预定对照，作为机制诊断证据。仅在另行预登记后开展多seed复现或独立检测收益实验；本轮不修改B0结构、不添加融合模块。\n'
    front='---\ntype: results-report\ndate: 2026-11-12\nexperiment_line: R1\nround: 1\npurpose: mechanism-validation\nstatus: complete\nsource_artifacts:\n  - analysis/analysis-report.md\n  - analysis/stats-appendix.md\n  - analysis/figure-catalog.md\nlinked_experiments: []\nlinked_results: []\n---\n\n'
    path=root/'2026-11-12--R1--r1--mechanism-validation.md';path.write_text(front+report)
    # Preserve late-added report/analysis source alongside stage hashes.
    repo=Path(__file__).resolve().parents[1]
    for name in ['r1/formal_report.py','r1/formal_stats.py']:
        target=root/'source_snapshot'/name;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes((repo/name).read_bytes())
    write_json(root/'report_complete.json',dict(complete=True,report=str(path),sha256=file_sha256(path),analysis_complete=json.loads((a/'complete.json').read_text()),critical_inputs_unchanged=True,baseline_sha256=sha,report_source_sha256=file_sha256(Path(__file__)),stats_source_sha256=file_sha256(repo/'r1/formal_stats.py')))
    plan=root/'task_plan.md';text=plan.read_text().replace('- [ ]','- [x]');text+='\n\n正式报告生成，所有18配置及六层诊断完成；pipeline最终退出状态由父进程记录。\n';plan.write_text(text)
    print('R1_REPORT_COMPLETE',path,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
