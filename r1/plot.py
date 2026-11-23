"""All figures read exported scalar records, without real model/GPU access."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def run(inputs,output):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Choose empty figure directory')
    out.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list);scatter=defaultdict(list);summaries=[];owners={}
    for input_index,directory in enumerate(inputs):
        directory=Path(directory);summaries.append(json.loads((directory/'summary.json').read_text()))
        # Histogram inputs are just scalar columns. Keep at most 20k per group
        # and 2k scatter pairs per layer, deterministically, to bound plot RAM.
        with gzip.open(directory/'queries.csv.gz','rt') as handle:
            for row in csv.DictReader(handle):
                if row['group'] not in ('known','unknown','background'):continue
                layer=int(row['layer'])
                for name in ['raw_norm','pre_norm','gaussian_objectness','classifier_unknown','full_error','masked_error']:
                    key=(layer,name,row['group'])
                    if row.get(name):owners.setdefault((layer,name),input_index)
                    if row.get(name) and owners.get((layer,name))==input_index and len(groups[key])<20000:groups[key].append(float(row[name]))
                if row.get('full_error') and owners.get((layer,'full_error'))==input_index and len(scatter[layer])<2000:
                    scatter[layer].append([float(row['gaussian_objectness']),float(row['full_error'])])
    signatures={json.dumps(x['reconstruction_config'],sort_keys=True) for x in summaries if 'reconstruction_config' in x}
    if len(signatures)>1:raise RuntimeError('Do not combine different mask rates, normalization, capacity or budgets in layer comparison')
    diagnostic=any(x['scope']=='synthetic_CPU_diagnostic' for x in summaries)
    title='CPU synthetic diagnostic | ' if diagnostic else ''
    for layer,name in sorted({(layer,name) for layer,name,group in groups}):
        fig,axis=plt.subplots(figsize=(6,3.5))
        for group,color in [('known','tab:green'),('unknown','tab:orange'),('background','tab:gray')]:
            values=groups[(layer,name,group)]
            if values:axis.hist(values,bins=min(35,max(1,len(values))),density=True,histtype='step',color=color,label=f'{group} n={len(values)}')
        axis.set(title=title+f'Layer {layer}: {name}',xlabel=name,ylabel='Density')
        if axis.get_legend_handles_labels()[0]:axis.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/f'layer{layer}_{name}.png',dpi=140);plt.close(fig)
    for layer,values in scatter.items():
        values=np.asarray(values);fig,axis=plt.subplots(figsize=(5,3.5));axis.scatter(values[:,0],values[:,1],s=5,alpha=.4)
        axis.set(title=title+f'Layer {layer}',xlabel='Gaussian objectness',ylabel='Reconstruction MSE (within this layer)')
        fig.tight_layout();fig.savefig(out/f'layer{layer}_objectness_error.png',dpi=140);plt.close(fig)
    comparable={};case_counts={}
    for summary in summaries:
        for layer,pools in summary['statistics'].items():
            for score,values in pools.get('all',{}).get('scores',{}).items():
                if score not in ('gaussian_objectness','classifier_unknown','full_error'):continue
                for direction,metrics in values['directions'].items():
                    value=metrics['unknown_vs_background']['auroc']
                    if value is not None:comparable[(int(layer),score,direction)]=value
        for layer,counts in summary['unknown_cases'].items():case_counts[int(layer)]=counts
    if comparable:
        keys=sorted(comparable);fig,axis=plt.subplots(figsize=(max(6,len(keys)*.45),4))
        axis.bar(range(len(keys)),[comparable[key] for key in keys]);axis.axhline(.5,color='gray',linestyle='--')
        axis.set_xticks(range(len(keys)),[f'L{l}\n{s}\n{d}' for l,s,d in keys],rotation=60,ha='right',fontsize=7)
        axis.set(title=title+'Unknown vs Background | both error signs are post-hoc',ylabel='AUROC',ylim=(0,1))
        fig.tight_layout();fig.savefig(out/'layer_separability.png',dpi=140);plt.close(fig)
    if case_counts:
        layers=sorted(case_counts);fig,axis=plt.subplots(figsize=(6,3.5))
        axis.bar([str(x) for x in layers],[case_counts[x].get('candidate_but_final_unknown_missed',0) for x in layers])
        axis.set(title=title+'Located but absent from final unknown geometry',xlabel='Decoder layer',ylabel='Unknown GT count (not official recall)')
        fig.tight_layout();fig.savefig(out/'located_unknown_misses.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',nargs='+',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.inputs,args.output)
