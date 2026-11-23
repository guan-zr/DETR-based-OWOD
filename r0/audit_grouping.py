"""CPU-only, score-free grouping migration audit on deterministic GT proposals.

No model/checkpoint loads, image decoding, AUROC, or threshold selection.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from r0.data import read_eval_gt
from r0.evaluate import candidate_pools,image_seed
from r0.labels import GroupingAudit,GROUPS
from r0.runtime import file_sha256,read_config,write_json
from r0.types import ImageGeometry


def run(config_path,output,ids_path=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Hide CUDA for CPU-only grouping audit')
    torch.set_num_threads(1)
    config=read_config(config_path);root=Path(config['data_root']);settings=config['evaluation']
    split=root/'ImageSets/TOWOD'/f"{config['eval_split']}.txt"
    ids=list(dict.fromkeys(split.read_text().splitlines()))
    if ids_path:
        requested=json.loads(Path(ids_path).read_text())
        if len(set(requested))!=len(requested) or not set(requested)<=set(ids):raise ValueError('Audit IDs must be unique members of eval split')
        ids=requested
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refuse to overwrite audit')
    out.mkdir(parents=True,exist_ok=True)
    audit=GroupingAudit();before=Counter();after=Counter()
    with (out/'images.jsonl').open('w') as handle:
        for stem in ids:
            xml=ET.parse(root/'Annotations'/f'{stem}.xml').getroot()
            hw=(int(xml.findtext('size/height')),int(xml.findtext('size/width')))
            gt=read_eval_gt(root,stem);geometry=ImageGeometry(stem,hw,hw,hw)
            seed=image_seed(stem,config['seed']);empty=torch.empty(0,4)
            old=candidate_pools(gt,empty,geometry,seed,dict(settings,background_rule='iou_only'))
            new=candidate_pools(gt,empty,geometry,seed,dict(settings,background_rule='disjoint_gt'),audit=audit)
            a=Counter(old['gt_reference']['labels']);b=Counter(new['gt_reference']['labels'])
            before.update(a);after.update(b)
            # Independently assert every retained background has zero intersection
            # with ALL bounded GT, including unknown and difficult.
            background=new['gt_reference']['boxes'][len(gt['boxes']):]
            bounded=new['gt_reference']['boxes'][:len(gt['boxes'])]
            if len(background) and len(bounded):
                sides=torch.minimum(background[:,None,2:],bounded[None,:,2:])-torch.maximum(background[:,None,:2],bounded[None,:,:2])
                if (sides>0).all(2).any():raise AssertionError('Background intersects GT')
            handle.write(json.dumps({'image_id':stem,'gt_reference_before':dict(a),'gt_reference_after':dict(b)})+'\n')
    result={'scope':'real_annotation_geometry_only; no detector queries or reconstruction scores',
            'images':len(ids),'split_file':str(split),'split_sha256':file_sha256(split),
            'positive_iou':settings['positive_iou'],'legacy_background_iou':settings['background_iou'],
            'candidate_seed':config['seed'],'proposals_per_image':settings['background_proposals'],
            'proposal_grouping':audit.summarize()['gt_reference_proposals'],
            'gt_reference':{'before_iou_only':{g:before[g] for g in GROUPS},
                            'after_disjoint_gt':{g:after[g] for g in GROUPS},
                            'delta':{g:after[g]-before[g] for g in GROUPS}},
            'query_counts':'pending final B0; must audit R0 queries and every R1 saved layer on actual extraction',
            'zero_intersection_assertion':True,'cuda_initialized':torch.cuda.is_initialized()}
    assert not result['cuda_initialized']
    write_json(out/'summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--ids-json');args=parser.parse_args();run(args.config,args.output,args.ids_json)
