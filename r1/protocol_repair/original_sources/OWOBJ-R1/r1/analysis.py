"""Shared offline scalar statistics; no reconstruction training or GT calibration."""
from array import array
from collections import Counter,defaultdict
import csv
import gzip
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import torch
from torchvision.ops import box_iou
from r1.reuse import binary_metrics,label_regions,write_json
from r1.runtime import clipped_boxes


def correlation(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return {'pearson':None,'spearman':None}
    return {'pearson':float(np.corrcoef(a,b)[0,1]),
            'spearman':float(np.corrcoef(rankdata(a),rankdata(b))[0,1])}


class Statistics:
    def __init__(self):self.values={};self.counts=defaultdict(Counter)

    def add(self,layer,pool,group,scores):
        self.counts[(layer,pool)][group]+=1
        if group not in ('known','unknown','background'):return
        for name,value in scores.items():
            if value is not None and np.isfinite(value):
                self.values.setdefault((layer,pool,name,group),array('d')).append(value)

    def summarize(self):
        result={}
        for (layer,pool),counts in sorted(self.counts.items()):
            stats={'counts':dict(counts),'scores':{}}
            names=sorted({name for l,p,name,g in self.values if (l,p)==(layer,pool)})
            for name in names:
                groups={group:np.asarray(self.values.get((layer,pool,name,group),[])) for group in ('known','unknown','background')}
                directions=[('higher',1),('lower',-1)] if name in ('full_error','masked_error') else [('higher',1)]
                score_stats={'distribution':{group:{'count':len(values),'mean':float(values.mean()) if len(values) else None,
                    'quantiles_05_50_95':np.quantile(values,[.05,.5,.95]).tolist() if len(values) else []}
                    for group,values in groups.items()},'directions':{}}
                for direction,sign in directions:
                    comparisons={}
                    for positive in ('known','unknown'):
                        pos,neg=groups[positive],groups['background']
                        comparisons[positive+'_vs_background']=binary_metrics(
                            np.r_[np.ones(len(pos),dtype=int),np.zeros(len(neg),dtype=int)],sign*np.r_[pos,neg])
                    score_stats['directions'][direction]=comparisons
                if name in ('full_error','masked_error'):
                    score_stats['direction_status']='Both directions are post-hoc offline diagnostics; no direction selected or calibrated'
                stats['scores'][name]=score_stats
            result.setdefault(str(layer),{})[pool]=stats
        return result


def rank_differences(indices,gaussian,error,groups):
    indices=np.asarray(indices,dtype=int)
    result={}
    if not len(indices):return result
    baseline=np.asarray(gaussian)[indices];values=np.asarray(error)[indices]
    for direction,sign in [('higher',1),('lower',-1)]:
        candidate=sign*values
        entry={'normalized_mean_rank_change':float(np.mean(np.abs(rankdata(baseline)-rankdata(candidate)))/max(len(indices)-1,1)),
               'topk':{}}
        for k in (10,50,100):
            count=min(k,len(indices));a=indices[np.argsort(-baseline,kind='stable')[:count]]
            b=indices[np.argsort(-candidate,kind='stable')[:count]]
            entry['topk'][str(k)]={'effective_k':count,'overlap_fraction':len(set(a)&set(b))/count,
                'gaussian_background_count':sum(groups[i]=='background' for i in a),
                'error_background_count':sum(groups[i]=='background' for i in b),
                'error_ambiguous_count':sum(groups[i] in ('ambiguous','invalid') for i in b)}
        result[direction]=entry
    return result


def unknown_cases(frame,slot,gt,threshold=.5):
    original_hw=tuple(np.asarray(frame['original_hw']).tolist())
    gt_boxes=clipped_boxes(gt['boxes'],original_hw)
    unknown=(~gt['known']&~gt['difficult']).nonzero(as_tuple=True)[0]
    if not len(unknown):return []
    boxes=clipped_boxes(frame['boxes_xyxy'][slot],original_hw)
    overlaps=box_iou(boxes,gt_boxes[unknown]).numpy()
    final=np.asarray(frame['final_boxes_xyxy'])[np.asarray(frame['final_pair_labels'])==80]
    final=clipped_boxes(final,original_hw)
    hit=box_iou(final,gt_boxes[unknown]).max(0).values.numpy()>=.5 if len(final) else np.zeros(len(unknown),dtype=bool)
    rows=[]
    for offset,index in enumerate(unknown.tolist()):
        column=overlaps[:,offset];best=int(column.argmax());covered=column>=.5
        rows.append({'image_id':str(frame['image_id']),'layer':int(np.asarray(frame['layer_ids'])[slot]),'gt_index':index,
            'max_iou':float(column.max()),'best_query_index':best,
            'coverage_03':bool((column>=.3).any()),'coverage_05':bool(covered.any()),'coverage_07':bool((column>=.7).any()),
            'candidate_but_final_unknown_missed':bool(covered.any() and not hit[offset]),
            'final_unknown_geometric_hit':bool(hit[offset]),
            'candidate_all_gaussian_below_fixed_threshold':bool(covered.any() and np.asarray(frame['gaussian_objectness'])[slot,covered].max()<threshold),
            'candidate_all_classifier_unknown_below_fixed_threshold':bool(covered.any() and np.asarray(frame['classifier_unknown'])[slot,covered].max()<threshold)})
    return rows


def evaluate_frames(frames,gt_loader,config,output,error_provider=None,synthetic=False):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refuse to overwrite analysis')
    out.mkdir(parents=True,exist_ok=True)
    stats=Statistics();query_writer=case_writer=None
    cases=defaultdict(Counter);correlations={};rank_totals=defaultdict(Counter);images=0
    with gzip.open(out/'queries.csv.gz','wt',newline='') as queries, gzip.open(out/'unknown_cases.csv.gz','wt',newline='') as case_file, (out/'ranking.jsonl').open('w') as ranking:
        import json
        for frame in frames:
            images+=1;gt=gt_loader(str(frame['image_id']));original_hw=tuple(np.asarray(frame['original_hw']).tolist())
            bounded_gt=dict(gt,boxes=clipped_boxes(gt['boxes'],original_hw))
            errors={} if error_provider is None else error_provider(frame)
            for slot,layer in enumerate(np.asarray(frame['layer_ids']).tolist()):
                if error_provider is not None and layer not in errors:continue
                boxes=clipped_boxes(frame['boxes_xyxy'][slot],original_hw)
                groups,maximum,nearest=label_regions(boxes,bounded_gt,config['positive_iou'],config['background_iou'])
                selected=np.asarray(frame['selected_unique'])[slot].astype(bool)
                raw=np.asarray(frame['raw'])[slot];pre=np.asarray(frame['pre_layernorm'])[slot]
                finite=np.isfinite(raw).all(1)&np.isfinite(pre).all(1)
                scores={'raw_norm':np.linalg.norm(raw,axis=-1),'pre_norm':np.linalg.norm(pre,axis=-1),
                    'l2_norm':np.linalg.norm(np.asarray(frame['l2'])[slot],axis=-1),
                    'gaussian_objectness':np.asarray(frame['gaussian_objectness'])[slot],
                    'classifier_unknown':np.asarray(frame['classifier_unknown'])[slot]}
                if layer in errors:scores.update(errors[layer])
                for q,group in enumerate(groups):
                    if not finite[q]:group='invalid'
                    values={name:float(value[q]) if np.isfinite(value[q]) else None for name,value in scores.items()}
                    stats.add(layer,'all',group,values)
                    if selected[q]:stats.add(layer,'selected',group,values)
                    row={'image_id':str(frame['image_id']),'layer':layer,'query_index':q,'group':group,
                        'selected':bool(selected[q]),'selection_scope':'actual_final' if layer==int(frame['decoder_layer_count']) else 'same_rule_intermediate_simulation',
                        'max_iou_gt':float(maximum[q]),'matched_gt_index':int(nearest[q]),**values}
                    if query_writer is None:query_writer=csv.DictWriter(queries,fieldnames=list(row));query_writer.writeheader()
                    query_writer.writerow(row)
                case_rows=unknown_cases(frame,slot,gt,config['low_objectness_threshold'])
                for row in case_rows:
                    if case_writer is None:case_writer=csv.DictWriter(case_file,fieldnames=list(row));case_writer.writeheader()
                    case_writer.writerow(row)
                    cases[layer]['unknown_gt']+=1
                    for key in ['coverage_03','coverage_05','coverage_07','candidate_but_final_unknown_missed',
                                'candidate_all_gaussian_below_fixed_threshold','candidate_all_classifier_unknown_below_fixed_threshold']:
                        cases[layer][key]+=int(row[key])
                if 'full_error' in scores:
                    for pool,keep in [('all',finite),('selected',finite&selected)]:
                        indices=np.flatnonzero(keep)
                        changes=rank_differences(indices,scores['gaussian_objectness'],scores['full_error'],groups)
                        ranking.write(json.dumps({'image_id':str(frame['image_id']),'layer':layer,'pool':pool,
                            'posthoc_both_error_directions':changes})+'\n')
                        for direction,change in changes.items():
                            for k,values in change['topk'].items():
                                total=rank_totals[f'L{layer}_{pool}_{direction}_top{k}']
                                total['images']+=1
                                for key in ['effective_k','gaussian_background_count','error_background_count','error_ambiguous_count']:
                                    total[key]+=values[key]
                        arrays=correlations.setdefault((layer,pool),{key:array('d') for key in ['gaussian_objectness','classifier_unknown','full_error']})
                        for key in arrays:arrays[key].extend(np.asarray(scores[key])[indices])
    summary={'scope':'synthetic_CPU_diagnostic' if synthetic else 'real_offline_analysis','images':images,
        'statistics':stats.summarize(),'unknown_cases':{str(layer):dict(counts) for layer,counts in cases.items()},
        'correlations':{str(layer)+'_'+pool:{score:correlation(values[score],values['full_error'])
            for score in ['gaussian_objectness','classifier_unknown']} for (layer,pool),values in correlations.items()},
        'fixed_topk_background_counts':{key:dict(value) for key,value in rank_totals.items()},
        'error_direction_policy':'Both signs are post-hoc diagnostics; no best direction chosen; no unknown GT training/calibration',
        'coverage_policy':'Unique-image independent GT geometric coverage, not official one-to-one U-Recall',
        'low_score_threshold':config['low_objectness_threshold'],
        'low_score_causality':'Gaussian score is not multiplied during OWOBJ inference; low-score candidates do not prove filtering cause',
        'annotation_limit':'Background is relative to all available 80-class annotations; unannotated/out-of-taxonomy objects may exist',
        'layer_error_policy':'Compare within-layer AUROC/AP, never rank layers by uncalibrated raw MSE'}
    write_json(out/'summary.json',summary)
    return summary
