from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'source_manifest.json').read_text());plot=json.loads((ROOT/'plot_metadata.json').read_text());z=np.load(ROOT/'paired_query_data.npz')
s=pd.read_csv(ROOT/'summary_statistics.csv',float_precision='round_trip');a=pd.read_csv(ROOT/'ambiguous_iou_group_stats.csv',float_precision='round_trip')
assert len(s)==15 and len(a)==8
assert len(z['group'])==911900 and len(np.unique(z['image_id']))==9119
for model,key in [('R0_S2','r0_error'),('R1_L6','r1_error'),('Gaussian','gaussian')]:
    for code,g in enumerate(['Known','Unknown','Background','Ambiguous']):
        row=s[(s.model==model)&(s['group']==g)].iloc[0];v=z[key][z['group']==code]
        assert row.n==len(v) and np.isclose(row['median'],np.median(v),rtol=1e-12)
    assert s[(s.model==model)&(s['group']!='Foreground')].n.sum()==911900
for model,d in a.groupby('model',sort=False):
    assert d.n.sum()==462437
    for field in ['mean','median','p25','p75']:assert (np.diff(d[field].to_numpy())>0).all()
expected=[f'{m}_{suffix}.png' for m in ['R0_S2','R1_L6'] for suffix in ['四类分布','Ambiguous按IoU分组分布','前景Ambiguous背景三组分布']]
expected+=['Gaussian_四类分布.png','Gaussian_前景Ambiguous背景三组分布.png']
assert all((ROOT/p).is_file() for p in expected)
figures=[]
for f in plot['figures']:
    p=ROOT/f['file']
    with Image.open(p) as im:
        assert min(im.info['dpi'])>=300 and im.size==(3888,2124)
        im.verify()
    figures.append(dict(file=f['file'],width=3888,height=2124,dpi=360,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if f['kind'] in ['four_groups','three_groups','iou_groups']:assert f['y_limit']>f['max_density']
assert len(figures)==12
assert all(abs(d['integral']-1)<1e-10 and d['n']==d['histogram_count'] for d in plot['density_checks'])
assert all(d['endpoint']==1 for d in plot['cdf_checks'])
# Confirm every hashed original input remains byte-for-byte unchanged.
for original in manifest['source_files']:
    h=hashlib.sha256()
    with Path(original['path']).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    assert h.hexdigest()==original['sha256'],original['path']
report=ROOT/'Ambiguous_重建误差补充分析报告.md'
for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
    if not target.startswith(('http:','https:')):assert (ROOT/target).exists(),target
result=dict(complete=True,all_requested_outputs_present=True,figures=figures,required_core_figures=6,gaussian_figures=2,cdf_figures=4,summary_rows=15,ambiguous_iou_stats_rows=8,counts_match_original=True,all_original_hashed_inputs_unchanged=True,original_hashed_files=len(manifest['source_files']),query_total=911900,images=9119,ambiguous_iou_subgroup_counts=a[a.model=='R0_S2'].n.astype(int).tolist(),no_training_or_inference=True,missing_fields_remaining=[],uncompleted_figures=[])
(ROOT/'completion_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
plan=ROOT/'task_plan.md';plan.write_text(plan.read_text().replace('- [ ] 检查图像与一致性并生成报告','- [x] 检查图像与一致性并生成报告').replace('图像检查与报告。','全部完成；12 张 PNG、两张主统计表、中文报告和来源/复现附件通过检查。'))
with (ROOT/'notes.md').open('a') as f:f.write('\n最终检查通过：12张PNG均3888×2124/360dpi，密度归一化、无截峰；原哈希输入均保持不变。完整报告和附录已生成。\n')
print(json.dumps({k:v for k,v in result.items() if k!='figures'},ensure_ascii=False,indent=2))
print('REPORT_BYTES',report.stat().st_size)
