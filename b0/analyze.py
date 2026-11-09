"""Full official evaluation + offline query coverage, in a single streaming pass."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from b0.common import (ARTIFACTS, build, clone_outputs, datasets, json_write,
                       load_checkpoint, model_args, seed_all, to_device)
from b0.features import FeatureCapture
from b0.metrics import iou, official_unknown_hits, query_groups
from b0.prepare_data import ALIASES
from b0.smoke import equal, subset
from datasets.open_world_eval import OWEvaluator
from util.box_ops import box_cxcywh_to_xyxy
from util.misc import collate_fn


def ground_truth(path,known_names):
    tree=ET.parse(path).getroot()
    xml_boxes=[];known=[];difficult=[];names=[]
    for obj in tree.findall('object'):
        name=obj.findtext('name');name=ALIASES.get(name,name)
        names.append(name)
        xml_boxes.append([float(obj.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']])
        known.append(name in known_names)
        difficult.append(bool(int(obj.findtext('difficult','0'))))
    xml_boxes=np.array(xml_boxes,dtype=float).reshape(-1,4)
    boxes=xml_boxes.copy();boxes[:,:2]-=1
    return {'boxes':boxes,'xml_boxes':xml_boxes,'known':np.array(known,dtype=bool),
            'difficult':np.array(difficult,dtype=bool),'names':names}


def plots(summary,records,groups,out):
    out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for group,color in [('known','tab:green'),('unknown','tab:orange'),('background','tab:gray')]:
        for axis,score in zip(axes,['latent','classifier_unknown']):
            values=np.concatenate(groups[group][score]) if groups[group][score] else np.array([])
            if len(values):axis.hist(values,bins=np.linspace(0,1,51),density=True,histtype='step',label=f'{group} (n={len(values)})',color=color)
    for axis,title in zip(axes,['Gaussian objectness exp(-T*d)','Classifier sigmoid(unknown logit)']):
        axis.set(xlabel=title,ylabel='Density');axis.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'objectness_distributions.png',dpi=180);plt.close(fig)
    values=[r['max_iou'] for r in records if r['group']=='unknown']
    fig,axis=plt.subplots(figsize=(6,3.6));axis.hist(values,bins=np.linspace(0,1,41))
    axis.set(xlabel='Maximum IoU over all 100 final-layer queries',ylabel='Unknown GT count')
    fig.tight_layout();fig.savefig(out/'unknown_max_iou.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for axis,ranking in zip(axes,['latent','classifier_unknown']):
        for group,style in [('known','--'),('unknown','-')]:
            for threshold,color in [('0.3','tab:blue'),('0.5','tab:orange'),('0.7','tab:green')]:
                ks=[10,50,100]
                ys=[summary['coverage'][group][ranking][str(k)][threshold]*100 for k in ks]
                axis.plot(ks,ys,style,marker='o',color=color,label=f'{group} IoU {threshold}')
        axis.set(title=ranking,xlabel='Top K queries',ylabel='GT coverage (%)',ylim=(0,100));axis.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'topk_coverage.png',dpi=180);plt.close(fig)
    fig,axis=plt.subplots(figsize=(7,4))
    labels=['All queries\nunique GT','Top10 latent\nunique GT','Final unknown TP\nunique GT','Official U-Recall\noriginal split rows']
    vals=[summary['coverage']['unknown']['all']['0.5']*100,
          summary['coverage']['unknown']['latent']['10']['0.5']*100,
          summary['unknown_misses']['official_hit_unique_fraction']*100,summary['official_metrics']['U_R50']]
    axis.bar(labels,vals,color=['tab:blue','tab:orange','tab:green','tab:gray'])
    axis.set(ylabel='Percentage (denominators shown)',ylim=(0,100));axis.tick_params(axis='x',labelsize=8)
    fig.tight_layout();fig.savefig(out/'coverage_vs_final_recall.png',dpi=180);plt.close(fig)


def run(checkpoint,config=None,smoke_limit=0,output=None):
    args=model_args(config);seed_all(args.seed)
    model,criterion,postprocessors,_=build(args)
    saved=load_checkpoint(model,checkpoint)
    if not smoke_limit:
        assert saved.get('epoch_complete',True) and saved['epoch']==args.epochs-1, 'Full B0 requires final epoch checkpoint'
    provenance={'checkpoint':str(Path(checkpoint).resolve()),'epoch':saved['epoch'],
                'sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                'training_git_commit':saved.get('git_commit')}
    del saved
    _,dataset=datasets(args)
    if smoke_limit:
        # Deliberately include annotated COCO unknowns, not only first VOC images.
        indices=[i for i,stem in enumerate(dataset.image_set) if len(stem)==6][:smoke_limit//2]
        indices += [i for i,stem in enumerate(dataset.image_set) if len(stem)==12][:smoke_limit-len(indices)]
        dataset=subset(dataset,indices)
    out=Path(output) if output else ARTIFACTS/('query_analysis/smoke' if smoke_limit else 'query_analysis/final')
    out.mkdir(parents=True,exist_ok=True)
    loader=DataLoader(dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=collate_fn,
                      num_workers=args.num_workers,pin_memory=True)
    evaluator=OWEvaluator(dataset,('bbox',),args=args)
    model.eval();criterion.eval()
    records=[];seen=set();gt_by_image={};group_counts=Counter()
    group_values={name:{score:[] for score in ['latent','classifier_unknown']} for name in ['known','unknown','background','ambiguous']}
    row_counts=Counter(dataset.image_set);sample_count=Counter()
    started=time.time();feature_shapes=None
    with torch.no_grad():
        for batch_index,(samples,targets) in enumerate(loader):
            samples,targets=to_device(samples,targets)
            if batch_index==0:
                before=model(samples)
                with FeatureCapture(model) as capture:
                    outputs=model(samples);equal(before,outputs);feature_shapes=capture.shapes()
                del before
            else:outputs=model(samples)
            sizes=torch.stack([t['orig_size'] for t in targets])
            # PostProcess mutates invalid logits in-place in the official release.
            # Clone inputs so analysis preserves every original query logit.
            results=postprocessors['bbox'](clone_outputs(outputs),sizes)
            res={target['image_id'].item():result for target,result in zip(targets,results)}
            evaluator.update(res)
            normalized_boxes=box_cxcywh_to_xyxy(outputs['pred_boxes'])
            masked=outputs['pred_logits'].clone();masked[:,:,postprocessors['bbox'].invalid_cls_logits]=-10e10
            selected=masked.sigmoid().flatten(1).topk(100,dim=1).indices
            for b,(target,result) in enumerate(zip(targets,results)):
                stem=dataset.convert_image_id(int(target['image_id'].item()),to_string=True)
                if stem in seen:continue # primary offline analysis counts physical objects once
                seen.add(stem)
                gt=ground_truth(Path(args.data_root)/'Annotations'/f'{stem}.xml',dataset.CLASS_NAMES[:20])
                gt_by_image[stem]=gt
                height,width=target['orig_size'].cpu().numpy()
                query_boxes=(normalized_boxes[b]*normalized_boxes.new_tensor([width,height,width,height])).cpu().numpy()
                distance=outputs['pred_obj'][b].cpu().numpy()
                latent=np.exp(-args.obj_temp/args.hidden_dim*distance)
                cls_unknown=outputs['pred_logits'][b,:,-1].sigmoid().cpu().numpy()
                overlaps=iou(query_boxes,gt['boxes'])
                groups=query_groups(query_boxes,gt['boxes'],gt['known'],gt['difficult'])
                group_counts.update(groups.tolist())
                for name in group_values:
                    group_values[name]['latent'].append(latent[groups==name])
                    group_values[name]['classifier_unknown'].append(cls_unknown[groups==name])
                ranks={'latent':np.argsort(-latent,kind='stable'),
                       'classifier_unknown':np.argsort(-cls_unknown,kind='stable')}
                for j in range(len(gt['boxes'])):
                    if gt['difficult'][j]:continue
                    maximum=float(overlaps[:,j].max())
                    nearest=int(overlaps[:,j].argmax())
                    record={'image_id':stem,'gt_index':j,'group':'known' if gt['known'][j] else 'unknown',
                        'class_name':gt['names'][j],'split_occurrences':row_counts[stem],'max_iou':maximum,
                        'best_iou_query':nearest,'best_iou_latent':float(latent[nearest]),
                        'best_iou_classifier_unknown':float(cls_unknown[nearest])}
                    for ranking,order in ranks.items():
                        for k in [10,50,100]:
                            record[f'{ranking}_top{k}_iou']=float(overlaps[order[:k],j].max())
                    records.append(record)
                sample_kind='coco' if len(stem)==12 else 'voc'
                if sample_count[sample_kind] < (12 if sample_kind=='coco' else 4):
                    query_ids=(selected[b]//81).cpu().numpy()
                    assert np.array_equal((selected[b]%81).cpu().numpy(),result['labels'].cpu().numpy())
                    np.testing.assert_allclose(query_boxes[query_ids],result['boxes'].cpu().numpy(),rtol=1e-6,atol=1e-4)
                    np.savez_compressed(out/f'sample_{stem}.npz',image_id=stem,query_index=np.arange(100),
                        boxes_xyxy=query_boxes,logits=outputs['pred_logits'][b].cpu().numpy(),
                        sigmoid_class_scores=outputs['pred_logits'][b].sigmoid().cpu().numpy(),
                        gaussian_distance=distance,latent_objectness=latent,classifier_unknown=cls_unknown,
                        final_query_indices=query_ids,final_boxes=result['boxes'].cpu().numpy(),
                        final_scores=result['scores'].cpu().numpy(),final_labels=result['labels'].cpu().numpy(),
                        gt_boxes_xyxy=gt['boxes'],gt_known=gt['known'],gt_difficult=gt['difficult'],
                        gt_names=np.array(gt['names']),orig_size=np.array([height,width]))
                    sample_count[sample_kind]+=1
            if batch_index%50==0:
                print(f'Analysis batch {batch_index}/{len(loader)}, unique_images={len(seen)}, seconds={time.time()-started:.1f}',flush=True)
    evaluator.synchronize_between_processes();evaluator.accumulate();official=evaluator.summarize()
    unknown_lines=[line for line,cls in zip(evaluator.lines,evaluator.lines_cls.tolist()) if cls==80]
    hits=official_unknown_hits(unknown_lines,gt_by_image)
    for record in records:
        record['official_unknown_hit']=bool((record['image_id'],record['gt_index']) in hits) if record['group']=='unknown' else False
        if record['group']=='unknown':
            record['miss_type']='detected' if record['official_unknown_hit'] else ('A_candidate_but_not_detected' if record['max_iou']>=0.5 else 'B_no_candidate')
        else:record['miss_type']='not_applicable'
    unknown=[r for r in records if r['group']=='unknown'];known=[r for r in records if r['group']=='known']
    assert abs(len(hits)/max(evaluator.num_unks[50][0],1)*100-official['U_R50'])<1e-6, 'Offline official hit replay differs'
    coverage={}
    for name,rows in [('known',known),('unknown',unknown)]:
        if not rows:raise RuntimeError(f'No {name} GT in analysis')
        coverage[name]={'all':{str(t):float(np.mean([r['max_iou']>=t for r in rows])) for t in [0.3,0.5,0.7]}}
        for rank in ranks:
            coverage[name][rank]={str(k):{str(t):float(np.mean([r[f'{rank}_top{k}_iou']>=t for r in rows]))
                for t in [0.3,0.5,0.7]} for k in [10,50,100]}
            for threshold in ['0.3','0.5','0.7']:
                assert coverage[name][rank]['100'][threshold]==coverage[name]['all'][threshold]
    miss=Counter(r['miss_type'] for r in unknown);missed=len(unknown)-len(hits)
    summary={'scope':'smoke_only' if smoke_limit else 'full_task1','checkpoint':provenance,
        'test_rows':len(dataset),'unique_images':len(seen),'unknown_gt_unique_non_difficult':len(unknown),
        'known_gt_unique_non_difficult':len(known),'unknown_gt_official_row_denominator':int(evaluator.num_unks[50][0]),
        'coverage':coverage,'official_metrics':official,'query_group_counts':dict(group_counts),
        'unknown_misses':{'counts':dict(miss),'missed_unique_gt':missed,
            'official_hit_unique_fraction':len(hits)/len(unknown),
            'A_fraction_all_unknown':miss['A_candidate_but_not_detected']/len(unknown),
            'B_fraction_all_unknown':miss['B_no_candidate']/len(unknown),
            'A_fraction_missed_unknown':miss['A_candidate_but_not_detected']/max(missed,1),
            'B_fraction_missed_unknown':miss['B_no_candidate']/max(missed,1)},
        'feature_shapes':feature_shapes,'feature_hooks_bitwise_equal':True,'seconds':time.time()-started,
        'coordinate_convention':'continuous XYXY in original image pixels; GT xmin/ymin minus 1, xmax/ymax unchanged (PROB loader)',
        'grouping':'nearest of all annotated GT; >=0.5 and non-difficult -> known/unknown; max<0.1 -> background; otherwise ambiguous. Ties XML order.',
        'A_definition':'missed by official unknown one-to-one matching but some raw final-layer query IoU>=0.5; includes wrong class, ranking and duplicates, not proof of low objectness',
        'B_definition':'missed by official unknown one-to-one matching and all raw queries IoU<0.5',
        'coverage_definition':'unique non-difficult GT, independent max over queries (query reuse allowed); no label/score threshold; TopK sorted by stated score',
        'official_protocol':'unmodified supplied split including duplicate IDs, original evaluator score/box rounding, IoU>0.5, VOC07 AP11; no new NMS/threshold',
        'inference_score':'official OWOBJ uses classifier sigmoid and global top100 query/class pairs; latent Gaussian objectness is NOT multiplied again'}
    distributions={}
    for name,values in group_values.items():
        distributions[name]={}
        for score,parts in values.items():
            array=np.concatenate(parts) if parts else np.array([])
            distributions[name][score]={'count':len(array),'mean':float(array.mean()) if len(array) else None,
                'quantiles_05_25_50_75_95':np.quantile(array,[.05,.25,.5,.75,.95]).tolist() if len(array) else []}
    summary['objectness_distributions']=distributions
    if not smoke_limit:
        original=ARTIFACTS/'evaluations/final/official_metrics.json'
        if original.exists():
            reference=json.loads(original.read_text())
            for key in ['K_AP50','U_R50','WI','AOSA']:
                assert abs(reference[key]-official[key])<1e-9, f'Official entrypoint mismatch: {key}'
            summary['official_entrypoint_metrics_identical']=True
    json_write(out/'summary.json',summary)
    json_write(out/'official_metrics.json',official)
    with (out/'per_gt.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    np.savez_compressed(out/'objectness_groups.npz',**{f'{group}_{score}':np.concatenate(parts)
        for group,values in group_values.items() for score,parts in values.items()})
    plots(summary,records,group_values,out/'figures')
    print('B0_ANALYSIS_COMPLETE '+json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--config')
    parser.add_argument('--smoke-limit',type=int,default=0)
    parser.add_argument('--output')
    args=parser.parse_args()
    run(args.checkpoint,args.config,args.smoke_limit,args.output)
