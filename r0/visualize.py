from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def distributions(records, output, diagnostic=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for source in sorted(set(r['source'] for r in records)):
        for level in sorted(set(r['level'] for r in records if r['source']==source)):
            fig,axis=plt.subplots(figsize=(6,3.8))
            for group,color in [('known','tab:green'),('unknown','tab:orange'),('background','tab:gray')]:
                values=[r['score'] for r in records if r['source']==source and r['level']==level
                        and r['group']==group and r['valid'] and np.isfinite(r['score'])]
                if values:axis.hist(values,bins=min(35,max(1,len(values))),density=True,
                                    histtype='step',color=color,label=f'{group} n={len(values)}')
            axis.set(xlabel='Mean squared reconstruction error',ylabel='Density',
                     title=('CPU synthetic diagnostic | ' if diagnostic else '')+source+' / '+level)
            if axis.get_legend_handles_labels()[0]:axis.legend(fontsize=8)
            fig.tight_layout();path=output/f'{source}_{level}.png';fig.savefig(path,dpi=150);plt.close(fig)
            paths.append(str(path))
    return paths


def group_distributions(groups, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for (source,level),values in sorted(groups.items()):
        fig,axis=plt.subplots(figsize=(6,3.8))
        for group,color in [('known','tab:green'),('unknown','tab:orange'),('background','tab:gray')]:
            array=np.asarray(values[group])
            if len(array):axis.hist(array,bins=40,density=True,histtype='step',color=color,label=f'{group} n={len(array)}')
        axis.set(xlabel='Mean squared reconstruction error',ylabel='Density',title=source+' / '+level)
        if axis.get_legend_handles_labels()[0]:axis.legend(fontsize=8)
        fig.tight_layout();fig.savefig(output/f'{source}_{level}.png',dpi=150);plt.close(fig)
