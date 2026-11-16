"""Official entrypoint, unchanged scoring/evaluator, forward observer for raw query coverage."""
import argparse, collections, csv, json, time, xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import DataLoader
from b1.common import *
import main_open_world as official
from util.box_ops import box_cxcywh_to_xyxy
from b0.metrics import iou
from datasets.torchvision_datasets.open_world import VOC_CLASS_NAMES_COCOFIED,BASE_VOC_CLASS_NAMES

def run(stage,smoke=0):
 a=args_for('eval_'+stage);seed(training=False);assert backend()==dict(matmul_tf32=False,cudnn_tf32=True,cudnn_benchmark=False,cudnn_deterministic=False,deterministic_algorithms=False)
 if smoke:
  a.pretrain=str(B0/'checkpoints/task1/checkpoint0040.pth');a.output_dir=str(E/'evaluation'/('smoke_'+stage))
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 source_files=['main_open_world.py','engine.py','models/prob_deformable_sketch_detr.py','datasets/open_world_eval.py','datasets/torchvision_datasets/open_world.py']
 sources={f:sha(W/f) for f in source_files};write(out/'config.json',vars(a))
 aliases=dict(zip(VOC_CLASS_NAMES_COCOFIED,BASE_VOC_CLASS_NAMES));seen=set();rows=[];progress={'batches':0};beg=time.time()
 original_evaluate=official.evaluate
 def observed(model,criterion,post,loader,base,device,output_dir,args):
  checkpoint=torch.load(args.pretrain,map_location='cpu');msg=model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
  write(out/'strict_load.json',dict(missing_keys=msg.missing_keys,unexpected_keys=msg.unexpected_keys))
  if smoke:
   import copy
   ds=copy.copy(base)
   voc=[i for i,s in enumerate(base.image_set) if len(s)==6];coco=[i for i,s in enumerate(base.image_set) if len(s)==12]
   indices=voc[:min(4,smoke)]+coco[:max(0,smoke-min(4,smoke))]
   for attr in ['images','annotations','imgids','image_set']:setattr(ds,attr,[getattr(base,attr)[i] for i in indices])
   ds.imgid2annotations=dict(zip(ds.imgids,ds.annotations));base=ds
   loader=DataLoader(ds,batch_size=5,shuffle=False,num_workers=4,collate_fn=official.utils.collate_fn,pin_memory=True)
  names=set(base.CLASS_NAMES[:40])
  def hook(module,inputs,outputs):
   boxes=box_cxcywh_to_xyxy(outputs['pred_boxes']).detach().cpu().numpy()
   # Sequential loader order, duplicates retained for official evaluator and deduplicated only for diagnostic.
   start=progress['batches']*5
   for b,q in enumerate(boxes):
    stem=base.image_set[start+b]
    if stem in seen:continue
    seen.add(stem);tree=ET.parse(Path(args.data_root)/'Annotations'/f'{stem}.xml').getroot()
    width=float(tree.findtext('size/width'));height=float(tree.findtext('size/height'));q=q*np.array([width,height,width,height],dtype=np.float32)
    gt=[];meta=[]
    for j,obj in enumerate(tree.findall('object')):
     n=aliases.get(obj.findtext('name'),obj.findtext('name'))
     if n in names or int(obj.findtext('difficult','0')):continue
     box=[float(obj.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']];box[0]-=1;box[1]-=1;gt.append(box);meta.append((j,n))
    if gt:
     overlaps=iou(q,gt);maximum=overlaps.max(axis=0);best=overlaps.argmax(axis=0)
     for j,((idx,n),value,k) in enumerate(zip(meta,maximum,best)): rows.append(dict(image_id=stem,gt_index=idx,class_name=n,max_iou=float(value),best_query=int(k),covered_iou_ge_050=bool(value>=.5)))
   progress['batches']+=1
   if progress['batches']%100==0:write(out/'progress.json',dict(at=now(),batches=progress['batches'],unique_images=len(seen),unknown_gt=len(rows),seconds=time.time()-beg))
  handle=model.register_forward_hook(hook)
  try:stats,evaluator=original_evaluate(model,criterion,post,loader,base,device,output_dir,args)
  finally:handle.remove()
  write(out/'per_class_ap50.json',dict(zip(base.CLASS_NAMES,evaluator.AP[:,0].tolist())))
  stats['metrics']={k:float(v) for k,v in stats['metrics'].items()}
  return stats,evaluator
 official.evaluate=observed
 try:official.main(a)
 finally:official.evaluate=original_evaluate
 covered=sum(r['covered_iou_ge_050'] for r in rows)
 with (out/'query_unknown_gt.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['image_id','gt_index','class_name','max_iou','best_query','covered_iou_ge_050']);writer.writeheader();writer.writerows(rows)
 diagnostic=dict(definition='unique image non-difficult unknown GT; max continuous xyxy IoU over all100 original queries >=0.5; no score/class threshold; query reuse allowed; xmin/ymin XML minus1; independent of official VOC quantization and greedy matching',known_classes=40,unknown_classes=40,unknown_index=80,unknown_gt_unique=len(rows),unknown_gt_covered=covered,coverage_ratio=covered/len(rows) if rows else None,coverage_percent=100*covered/len(rows) if rows else None,unique_images=len(seen),batches=progress['batches'])
 write(out/'query_coverage.json',diagnostic)
 metrics=json.loads((out/'official_metrics.json').read_text())
 manifest=dict(at=now(),complete=True,scope='smoke' if smoke else 'full',checkpoint=a.pretrain,checkpoint_sha256=sha(a.pretrain),test_set=a.test_set,split_sha256=sha(Path(a.data_root)/'ImageSets/TOWOD'/f'{a.test_set}.txt'),evaluation_batch_size=5,backend=backend(),source_sha256=sources,official_entrypoint='main_open_world.main -> engine.evaluate -> PostProcess -> OWEvaluator',score_threshold=None,postprocessing='global top100 over query/class sigmoid; no extra NMS; no objectness multiplication',iou='official VOC IoU>0.5, inclusive pixels, VOC07 AP11',units='mAP/AP/recall/P are percentages; WI fraction at recall0.8; AOSA is A-OSE count',metrics=metrics,query_diagnostic=diagnostic,seconds=time.time()-beg)
 assert sources=={f:sha(W/f) for f in source_files};write(out/'manifest.json',manifest)
 print('EVALUATION_COMPLETE '+json.dumps(manifest),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['stage_a','stage_b'],required=True);p.add_argument('--smoke-limit',type=int,default=0);o=p.parse_args();run(o.stage,o.smoke_limit)
