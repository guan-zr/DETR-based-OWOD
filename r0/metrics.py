"""Tie-aware ROC AUC and average precision, without adding sklearn."""
import numpy as np
from array import array


def binary_metrics(labels, scores):
    labels=np.asarray(labels,dtype=np.int64);scores=np.asarray(scores,dtype=np.float64)
    if labels.shape != scores.shape or not np.isin(labels,[0,1]).all() or not np.isfinite(scores).all():
        raise ValueError('Expected finite scores and binary labels of equal shape')
    positives=int(labels.sum());negatives=int(len(labels)-positives)
    result={'positive_count':positives,'negative_count':negatives,'auroc':None,
            'auprc_average_precision':None,'positive_prevalence':positives/len(labels) if len(labels) else None}
    if not positives or not negatives:
        result['undefined_reason']='Both positive and negative examples are required'
        return result
    order=np.argsort(-scores,kind='stable'); y=labels[order]; s=scores[order]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.cumsum(y)[ends].astype(float);fp=(ends+1)-tp
    tpr=np.r_[0,tp/positives];fpr=np.r_[0,fp/negatives]
    precision=tp/(tp+fp);recall=tp/positives
    result['auroc']=float(np.trapz(tpr,fpr))
    result['auprc_average_precision']=float(np.sum(np.diff(np.r_[0,recall])*precision))
    return result


def evaluate_records(records):
    result={}
    for source in sorted(set(r['source'] for r in records)):
        result[source]={}
        levels=sorted(set(r['level'] for r in records if r['source']==source))
        for level in levels:
            rows=[r for r in records if r['source']==source and r['level']==level and r['valid']
                  and r['group'] in ('known','unknown','background') and np.isfinite(r['score'])]
            comparisons={}
            for positive in ('known','unknown'):
                selected=[r for r in rows if r['group'] in (positive,'background')]
                comparisons[positive+'_vs_background']=binary_metrics(
                    [r['group']==positive for r in selected],[r['score'] for r in selected])
            comparisons['counts']={group:sum(r['group']==group for r in rows)
                                   for group in ('known','unknown','background')}
            result[source][level]=comparisons
    return result


class ScoreAccumulator:
    """Store scalar scores only, rather than retaining per-box record dictionaries."""
    def __init__(self):
        self.groups={}
        self.excluded={}

    def add(self,record):
        key=(record['source'],record['level'])
        groups=self.groups.setdefault(key,{name:array('d') for name in ('known','unknown','background')})
        if not record['valid'] or record['group'] not in ('known','unknown','background') or record['score'] is None or not np.isfinite(record['score']):
            self.excluded[key]=self.excluded.get(key,0)+1
            return
        groups[record['group']].append(record['score'])

    def summarize(self):
        result={}
        for (source,level),groups in sorted(self.groups.items()):
            values={name:np.asarray(value) for name,value in groups.items()}
            stats={'counts':{name:len(value) for name,value in values.items()},
                   'excluded_count':self.excluded.get((source,level),0)}
            for positive in ('known','unknown'):
                pos,negative=values[positive],values['background']
                stats[positive+'_vs_background']=binary_metrics(
                    np.r_[np.ones(len(pos),dtype=int),np.zeros(len(negative),dtype=int)],np.r_[pos,negative])
            result.setdefault(source,{})[level]=stats
        return result
