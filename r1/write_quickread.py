"""Evidence-only reader's note; never changes model scores or experiment metrics."""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd


def main(root):
    root=Path(root);a=root/'analysis';ci=pd.read_csv(a/'unknown_cluster_intervals.csv');allm=pd.read_csv(a/'R1_1_complete_comparison.csv');paired=pd.read_csv(a/'paired_error_minus_baseline.csv');cfg=pd.read_csv(a/'configuration_pair_differences.csv');top=pd.read_csv(a/'topk_softlabel_audit.csv');low=pd.read_csv(a/'gaussian_bottom25_comparison.csv');coverage=pd.read_csv(a/'unknown_coverage.csv');diag=pd.read_csv(a/'R1_0_layer_summary.csv')
    text='# R1结果速读：重建误差能否作为Query Objectness软标签\n\n**本轮支持它作为前景排序的候选信号，尚未证明它已是可靠的概率软标签或有效的检测监督。masking也没有显示普遍的额外收益。**\n\n'
    text+='本轮R1-0与R1-1已全部完成：冻结B0 checkpoint0040，六层Decoder，100 Query/层、256维；测试9119唯一图，训练池16551图，各AE使用seed91固定顺序的前2000唯一图、2000 optimizer steps、batch1，训练/测试ID交集0。18配置为L1/3/6×mask0.6/0/0.3×l2/raw；不读取训练XML或Unknown GT，不更新B0。主流程13:09:35退出0，报告修订不重跑模型。\n\n'
    text+='## 1. 主实验：同一方向同时区分Known与Unknown\n\n以下是l2、mask0.6、全部原始Query、高MSE方向；Query级AP是average precision，不是官方检测mAP。\n\n|层|Known AUROC|Unknown AUROC|Unknown 95%图像区间|Unknown AP|B0 Gaussian Unknown AUROC|\n|---|---:|---:|---|---:|---:|\n'
    for layer in [1,3,6]:
        run=f'l2_L{layer}_mask0.6';p=allm.loc[(allm.representation=='l2')&(allm['mask']==.6)&(allm.layer==layer)&(allm.direction=='higher')].iloc[0];v=ci.loc[(ci.run==run)&(ci.pool=='all')&(ci.direction=='higher')].iloc[0];g=diag.loc[diag.layer==layer].iloc[0]
        text+=f'|L{layer}|{p.Known_AUROC:.6f}|{p.Unknown_AUROC:.6f}|[{v.auroc_ci_low:.6f}, {v.auroc_ci_high:.6f}]|{p.Unknown_AP:.6f}|{g.gaussian_objectness_AUROC:.6f}|\n'
    text+='\n同一主实验低MSE方向的Unknown AUROC分别为0.110、0.149、0.232，完整报告同时保留两个方向；没有从测试结果选择更有利的符号。当前是高误差更偏前景，但这不是其他基线、训练seed或数据域的普遍保证。\n\n'
    text+='## 2. 不能把收益归因于masking\n\n|l2表示|L1 Unknown AUROC|L3|L6|\n|---|---:|---:|---:|\n'
    for mask in [.6,0.,.3]:
        d=allm.loc[(allm.representation=='l2')&(allm['mask']==mask)&(allm.direction=='higher')].sort_values('layer');text+=f"|mask={mask:g}|"+'|'.join(f'{v:.6f}' for v in d.Unknown_AUROC)+'|\n'
    v=cfg.loc[(cfg.left=='l2_L6_mask0.6')&(cfg.right=='l2_L6_mask0')&(cfg.pool=='all')&(cfg.direction=='higher')].iloc[0]
    text+=f'\nL6、l2中，mask0.6−mask0的Unknown AUROC差为{v.auroc_difference:.6f}，点态成对95%图像区间[{v.auroc_ci_low:.6f}, {v.auroc_ci_high:.6f}]。raw也完整运行，不能只保留有利配置。L6主实验raw约0.846、l2约0.768，说明表示/归一化影响明显；跨层、跨表示的原始MSE大小不能直接当作效果排名。只有一个训练seed，掩码敏感性不能替代多seed稳定性。\n\n'
    text+='## 3. 有额外排序线索，但依赖层和分数区间\n\n同Query、同图像抽样比较中，主实验三层误差相对Gaussian的全量AUROC差值点态区间均大于0。Gaussian最低25%区间的结果更能说明局限：\n\n|层|Gaussian Unknown AUROC|高误差Unknown AUROC|\n|---|---:|---:|\n'
    for layer in [1,3,6]:
        d=low.loc[low.run==f'l2_L{layer}_mask0.6'];g=d.loc[(d.score=='gaussian_objectness')&(d.direction=='higher')].iloc[0];e=d.loc[(d.score=='full_error')&(d.direction=='higher')].iloc[0];text+=f'|L{layer}|{g.auroc:.6f}|{e.auroc:.6f}|\n'
    text+='\nL1/L3在该低分区域仍有剩余区分，L6主实验则接近机会水平，且不优于该区间Gaussian排序。因此不能笼统说所有层、所有区间都提供额外信息。分位切点只使用全部Query的分数，不用GT定阈值；这些条件诊断不证明独立因果信息或融合收益。全量数值审计表明距离/logits有限、Gaussian概率零值占比均0，不能把较弱Gaussian指标归因于下溢。\n\n'
    text+='## 4. 为什么还不能叫“可靠软标签”\n\n原协议固定每图Top10，主实验高误差方向的组成如下；Known+Unknown均按GT IoU≥0.5定义，Ambiguous单列：\n\n|层|明确标注前景|明确背景|歧义|\n|---|---:|---:|---:|\n'
    for layer in [1,3,6]:
        row=top.loc[(top.run==f'l2_L{layer}_mask0.6')&(top.pool=='all')&(top.k==10)&(top.direction=='higher')].iloc[0];text+=f'|L{layer}|{row.error_labeled_foreground_fraction:.2%}|{row.error_background_fraction:.2%}|{row.error_ambiguous_fraction:.2%}|\n'
    text+='\n不能把歧义部分当作正确或错误标签，也不能将“明确前景比例”误称为真实Objectness准确率。高AUROC是在已定义的前景/背景对上得到的，不能直接保证全部Query上的软标签质量。MSE尚未校准成P(object)，其数值受层、表示与掩码影响；本轮也未把它用于新检测器监督、蒸馏或端到端训练。\n\n'
    row=coverage.loc[coverage.layer==6].iloc[0];text+=f"末层固定100 Query只在IoU≥0.5下覆盖{int(row.coverage_05_count)}/{int(row.unknown_gt)}个Unknown GT（{row.coverage_05_fraction:.2%}），还有{int(row.unknown_gt-row.coverage_05_count)}个没有合格框。仅在这些固定框上重排，不能为未覆盖区域创造新候选。几何覆盖不是官方U-Recall。\n\n"
    text+='B0本身使用Known分类监督；AE在其已学习好的Query上做无标签重建，不能由此声称独立于Known监督的通用Objectness。post-LN范数的高值AUROC低于0.5也不等于没有信息，反向AUROC见完整表；没有按该反向重新选择配置。\n\n'
    text+='## 5. 统计与使用边界\n\n所有区间是200次同图像聚类bootstrap的点态区间，保留图像内Query相关性，不含训练seed变化。探索性Holm校正受200次抽样的p分辨率限制，不能包装成校正后显著性结论。正式all Query、selected池分开；L1/L3 selected为同规则模拟，L6为实际Top100 pair去重Query，GT分组Unknown不等于预测为Unknown的最终检测。GT-reference Query表示没有可靠定义，本轮不虚构。\n\n操作结论：保留重建误差作为软标签候选信号，暂不宣称可靠软标签已成立；masking贡献未获普遍支持。后续需要另行预登记校准、多seed和实际监督收益验证，本轮按指令不增加融合或端到端结构。\n\n'
    text+='[完整报告](2026-11-12--R1--r1--mechanism-validation.md) · [全部指标](analysis/all_metrics.csv) · [成对差值与区间](analysis/paired_error_minus_baseline.csv) · [配置对照](analysis/configuration_pair_differences.csv) · [软标签TopK审计](analysis/topk_softlabel_audit.csv) · [实际执行命令](R1_实际执行命令.md) · [运行和恢复说明](运行说明.md)\n'
    dest=root/'R1_结果速读_20261112.md';dest.write_text(text);(root/'quickread_complete.json').write_text(json.dumps(dict(complete=True,path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2));print(dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
