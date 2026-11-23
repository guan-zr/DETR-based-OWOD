"""Offline GT scoring; never imported by the image-only reconstruction trainer."""
import argparse
import csv
import gzip
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from r0.data import ImageOnlyDataset,image_collate,read_eval_gt
from r0.labels import background_candidates,label_regions,GroupingAudit
from r0.metrics import ScoreAccumulator
from r0.reconstruction import FMRBank
from r0.regions import mean_region_error
from r0.runtime import guard_b0_complete,load_detector,pack_images,read_config,seed_all,write_json,git_provenance,reconstruction_config_matches
from r0.visualize import group_distributions


def image_seed(image_id,base):
    return (base+int(hashlib.sha256(image_id.encode()).hexdigest()[:8],16))%(2**63-1)


def candidate_pools(gt,query_boxes,geometry,seed,settings,final_boxes=None,audit=None):
    def clipped(boxes):
        boxes=boxes.detach().cpu().clone()
        height,width=geometry.original_hw
        boxes[:,0::2]=boxes[:,0::2].clamp(0,width)
        boxes[:,1::2]=boxes[:,1::2].clamp(0,height)
        return boxes
    # Label the exact bounded regions that ROIAlign will score. Otherwise an
    # oversized out-of-image query can appear to be background before clipping.
    gt=dict(gt,boxes=clipped(gt['boxes']))
    query_boxes=clipped(query_boxes)
    if final_boxes is not None:final_boxes=clipped(final_boxes)
    proposals=background_candidates(geometry.original_hw,seed,settings['background_proposals'])
    rule=settings.get('background_rule','disjoint_gt')
    if audit is not None:
        audit.add('gt_reference_proposals',proposals,gt,settings['positive_iou'],settings['background_iou'])
    group,_,_=label_regions(proposals,gt,settings['positive_iou'],settings['background_iou'],rule)
    background=proposals[torch.tensor([x=='background' for x in group],dtype=torch.bool)]
    gt_labels=['ambiguous' if difficult else ('known' if known else 'unknown')
               for known,difficult in zip(gt['known'].tolist(),gt['difficult'].tolist())]
    pools={'gt_reference':{'boxes':torch.cat([gt['boxes'],background]),
                           'labels':gt_labels+['background']*len(background),
                           'origin':['annotated_gt']*len(gt_labels)+['sampled_background']*len(background),
                           'max_iou':[1.0]*len(gt_labels)+[float(x) for x in label_regions(background,gt,
                               settings['positive_iou'],settings['background_iou'],rule)[1]]}}
    prediction_pools={'query':query_boxes}
    if final_boxes is not None:prediction_pools['final_detection']=final_boxes
    for name,boxes in prediction_pools.items():
        if audit is not None:audit.add(name,boxes,gt,settings['positive_iou'],settings['background_iou'])
        labels,maximum,_=label_regions(boxes,gt,settings['positive_iou'],settings['background_iou'],rule)
        pools[name]={'boxes':boxes,'labels':labels,'origin':[name]*len(boxes),'max_iou':maximum.tolist()}
    return pools


def ensemble_errors(bank,batch,ratio,repeats,seed):
    if repeats < 1:raise ValueError('At least one evaluation repeat is required')
    sums=None
    with torch.no_grad():
        for repeat in range(repeats):
            _,levels=bank(batch,mask_ratio=ratio,seed=seed+repeat*100)
            if sums is None:
                sums=[{'error':torch.zeros_like(x['error']), 'masked_numerator':torch.zeros_like(x['error']),
                       'masked_support':torch.zeros_like(x['error']), 'valid':x['valid']} for x in levels]
            for total,item in zip(sums,levels):
                total['error']+=item['error']/repeats
                total['masked_numerator']+=item['error']*item['masked']/repeats
                total['masked_support']+=item['masked'].to(item['error'].dtype)/repeats
    for total in sums:
        total['masked_error']=total['masked_numerator']/total['masked_support'].clamp(min=1e-12)
    return sums


def region_records(batch,errors,pools,settings):
    if len(batch.geometry)!=1:raise ValueError('Offline evaluation is intentionally one image per batch')
    geometry=batch.geometry[0]
    records=[]
    for source,pool in pools.items():
        boxes=pool['boxes'].cpu()
        per_scale=[]
        for level,(maps,stride) in enumerate(zip(errors,batch.strides)):
            kwargs={key:settings[key] for key in ('output_size','sampling_ratio','min_box_size','min_valid_fraction')}
            score=mean_region_error(maps['error'][0],maps['valid'][0],boxes,geometry,stride,**kwargs)
            masked=mean_region_error(maps['masked_error'][0],maps['masked_support'][0],boxes,geometry,stride,
                                     **dict(kwargs,min_valid_fraction=0.0))
            rows=[]
            for index,(box,group) in enumerate(zip(boxes.tolist(),pool['labels'])):
                rows.append({'image_id':geometry.image_id,'source':source,'candidate_id':index,
                    'origin':pool['origin'][index],'group':group,'max_iou_gt':pool['max_iou'][index],
                    'level':f's{level}_stride{stride}','x1':box[0],'y1':box[1],'x2':box[2],'y2':box[3],
                    'score':float(score.scores[index]) if score.valid[index] else None,
                    'masked_score':float(masked.scores[index]) if masked.valid[index] else None,
                    'valid':bool(score.valid[index]),'valid_fraction':float(score.valid_fraction[index]),
                    'small_feature_roi':bool(score.small_feature_roi[index])})
            records.extend(rows);per_scale.append(rows)
        for index in range(len(boxes)):
            levels=[rows[index] for rows in per_scale]
            overall=dict(levels[0],level='overall')
            overall['valid']=all(row['valid'] for row in levels)
            overall['score']=float(np.mean([row['score'] for row in levels])) if overall['valid'] else None
            overall['masked_score']=float(np.mean([row['masked_score'] for row in levels])) if all(row['masked_score'] is not None for row in levels) else None
            overall['valid_fraction']=min(row['valid_fraction'] for row in levels)
            overall['small_feature_roi']=any(row['small_feature_roi'] for row in levels)
            records.append(overall)
    return records


def run(config_path,reconstruction_checkpoint,output=None,device='cuda'):
    config=read_config(config_path)
    if config['evaluation']['include_final_detections']:
        raise RuntimeError('final_detection is blocked until offline query-ID deduplication is implemented; official postprocessing is unchanged')
    guard_b0_complete(config);seed_all(config['seed'])
    saved=torch.load(reconstruction_checkpoint,map_location='cpu')
    if not reconstruction_config_matches(saved['config'],config) or not saved.get('complete'):
        raise RuntimeError('Use a complete R0 checkpoint matching all training/extraction/scoring settings')
    out=Path(output or str(Path(config['output'])/'evaluation'))
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refuse to overwrite existing evaluation')
    out.mkdir(parents=True,exist_ok=True)
    extractor,postprocessors,baseline=load_detector(config,torch.device(device))
    if saved['baseline']['sha256']!=baseline['sha256']:raise RuntimeError('Frozen detector checksum differs')
    bank=FMRBank(config['channels'],**config['autoencoder'],mask_ratio=config['mask_ratio']).to(device)
    bank.load_state_dict(saved['bank'],strict=True);bank.eval();training_config=saved['config'];del saved
    data=ImageOnlyDataset(config['data_root'],config['eval_split'],unique=True,**config['resize'])
    loader=DataLoader(data,batch_size=1,shuffle=False,num_workers=config['num_workers'],collate_fn=image_collate)
    accumulator=ScoreAccumulator();audit=GroupingAudit();writer=None
    settings=config['evaluation'];roi=config['roi']
    with gzip.open(out/'regions.csv.gz','wt',newline='') as handle, (out/'images.jsonl').open('w') as images:
        for index,items in enumerate(loader):
            samples,geometry=pack_images(items,torch.device(device))
            batch,outputs=extractor(samples,geometry)
            seed=image_seed(geometry[0].image_id,config['seed'])
            errors=ensemble_errors(bank,batch,settings['mask_ratio'],settings['repeats'],seed)
            from util.box_ops import box_cxcywh_to_xyxy
            height,width=geometry[0].original_hw
            query=(box_cxcywh_to_xyxy(outputs['pred_boxes'][0])*outputs['pred_boxes'].new_tensor([width,height,width,height])).cpu()
            final=None
            if settings['include_final_detections']:
                copied={key:value.clone() if torch.is_tensor(value) else value for key,value in outputs.items()}
                size=torch.tensor([[height,width]],device=device)
                final=postprocessors['bbox'](copied,size)[0]['boxes'].cpu()
            gt=read_eval_gt(config['data_root'],geometry[0].image_id)
            pools=candidate_pools(gt,query,geometry[0],seed,settings,final,audit)
            records=region_records(batch,errors,pools,roi)
            if writer is None:
                writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader()
            for row in records:
                writer.writerow(row);accumulator.add(row)
            write={'image_id':geometry[0].image_id,'original_hw':geometry[0].original_hw,
                   'resized_hw':geometry[0].resized_hw,'padded_hw':geometry[0].padded_hw,
                   'feature_shapes':[list(x.shape) for x in batch.levels],
                   'regions_per_source':{key:len(pool['boxes']) for key,pool in pools.items()}}
            import json
            images.write(json.dumps(write)+'\n')
            if index%100==0:print(f'Offline evaluation {index}/{len(data)}',flush=True)
    write_json(out/'grouping_audit.json',audit.summarize())
    summary={'grouping_audit':audit.summarize(),'background_rule':settings.get('background_rule','disjoint_gt'),'metrics':accumulator.summarize(),'config':config,'checkpoint_training_config':training_config,'baseline':baseline,'code':git_provenance(),
             'unique_images':len(data),'higher_error_is_foreground':True,
             'auprc_definition':'step-integral average precision, score ties grouped',
             'overall_definition':'equal mean over all four valid scale scores; otherwise invalid',
             'background_limit':'disjoint_gt requires zero intersection with all available GT (including difficult); iou_only is a legacy low-IoU diagnostic; neither certifies absence of unannotated objects',
             'gt_reference_definition':'annotated non-difficult objects + image-only sampled background filtered against ALL GT',
             'unknown_labels_used':'offline evaluation only, never AE training or parameter selection'}
    write_json(out/'summary.json',summary)
    np.savez_compressed(out/'scalar_distributions.npz',**{source+'__'+level+'__'+group:np.asarray(values)
        for (source,level),groups in accumulator.groups.items() for group,values in groups.items()})
    group_distributions(accumulator.groups,out/'figures')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output');parser.add_argument('--device',default='cuda')
    args=parser.parse_args();run(args.config,args.checkpoint,args.output,args.device)
