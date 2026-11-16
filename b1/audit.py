import collections, json, os, platform, shutil, subprocess, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path
import torch, torchvision
from b1.common import *
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES, VOC_CLASS_NAMES_COCOFIED, BASE_VOC_CLASS_NAMES

def main():
 seed(); d=shutil.disk_usage(E); ckpath=B0/'checkpoints/task1/checkpoint0040.pth'; size=ckpath.stat().st_size
 # 2 snapshots/stage + one atomic temporary + two smoke states + logs/evaluation allowance.
 budget=7*size+2*GIB
 storage=dict(at=now(),total_bytes=d.total,used_bytes=d.used,available_bytes=d.free,b0_single_checkpoint_bytes=size,b0_single_checkpoint_mib=size/1024**2,
 stage_a_retained_checkpoint_bytes=2*size,stage_b_retained_checkpoint_bytes=2*size,atomic_write_peak_extra_bytes=size,smoke_allowance_bytes=2*size,logs_evaluation_allowance_bytes=2*GIB,
 estimated_additional_peak_bytes=budget,reserve_bytes=5*GIB,passed=d.free>budget+5*GIB,
 strategy='Each stage: rolling latest + previous, final hardlink to latest; atomic tmp; no per-epoch archive; smoke at most two states; check free>=5GiB each update and save; fail safely without deleting protected files')
 write(E/'reports/storage_budget.json',storage); assert storage['passed'],storage
 ck=torch.load(ckpath,map_location='cpu'); a=args_for('stage_a'); m,c,p,x=build(a); msg=m.load_state_dict(ck['model'],strict=True)
 maxdiff=max(float((v.cpu()-ck['model'][k]).abs().max()) for k,v in m.state_dict().items() if v.is_floating_point())
 checkpoint=dict(path=str(ckpath),sha256=sha(ckpath),epoch=ck['epoch'],epoch_complete=ck.get('epoch_complete',True),git_commit=ck.get('git_commit'),missing_keys=msg.missing_keys,unexpected_keys=msg.unexpected_keys,all_loaded_float_parameters_max_abs_difference=maxdiff,old_mapping=list(VOC_COCO_CLASS_NAMES['TOWOD'][:20]),new_mapping=list(VOC_COCO_CLASS_NAMES['TOWOD'][20:40]),unknown_index=80,head_shapes={k:list(v.shape) for k,v in ck['model'].items() if 'class_embed' in k},invalid_logits=p['bbox'].invalid_cls_logits)
 assert checkpoint['epoch']==40 and maxdiff==0 and ck['args'].CUR_INTRODUCED_CLS==20 and ck['args'].dataset=='TOWOD'
 write(E/'reports/checkpoint_audit.json',checkpoint); del ck,m,c,p,x; torch.cuda.empty_cache()
 aliases=dict(zip(VOC_CLASS_NAMES_COCOFIED,BASE_VOC_CLASS_NAMES)); names=VOC_COCO_CLASS_NAMES['TOWOD']; root=E/'data'; report={}; errors=[]
 for split in ['owod_t1_train','owod_t2_train','owod_all_task_test']:
  path=root/'ImageSets/TOWOD'/f'{split}.txt'; stems=path.read_text().splitlines(); cnt=collections.Counter(); empty=0; missing=[]
  for stem in set(stems):
   if not (root/'JPEGImages'/f'{stem}.jpg').is_file() or not (root/'Annotations'/f'{stem}.xml').is_file(): missing.append(stem); continue
   objs=ET.parse(root/'Annotations'/f'{stem}.xml').getroot().findall('object'); labels=[]
   for o in objs:
    n=aliases.get(o.findtext('name'),o.findtext('name'))
    if n not in names: errors.append([stem,n]); continue
    labels.append(names.index(n)); cnt[names.index(n)]+=1
   valid=range(20,40) if split=='owod_t2_train' else range(0,20) if split=='owod_t1_train' else range(81)
   if not any(l in valid for l in labels): empty+=1
  report[split]=dict(rows=len(stems),unique_images=len(set(stems)),duplicate_rows=len(stems)-len(set(stems)),sha256=sha(path),missing=missing,raw_annotation_counts_by_class=dict(sorted(cnt.items())),images_without_current_supervision=empty)
  if missing: errors.append([split,'missing',len(missing)])
 t1=set((root/'ImageSets/TOWOD/owod_t1_train.txt').read_text().splitlines()); t2=set((root/'ImageSets/TOWOD/owod_t2_train.txt').read_text().splitlines()); test=set((root/'ImageSets/TOWOD/owod_all_task_test.txt').read_text().splitlines())
 report.update(errors=errors,split_overlap=dict(t1_t2=len(t1&t2),t1_test=len(t1&test),t2_test=len(t2&test)),unknown_supervision='removed in train/ft; test labels collapsed to80')
 assert not errors,errors; assert not t1&test and not t2&test
 write(E/'reports/data_audit.json',report)
 import MultiScaleDeformableAttention as op
 env=dict(at=now(),python=sys.version,executable=sys.executable,torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),op_path=op.__file__,backend=backend(),env_reused=True)
 write(E/'reports/environment.json',env)
 print(json.dumps({'storage':storage,'data_splits':{k:v['rows'] for k,v in report.items() if isinstance(v,dict) and 'rows' in v},'environment':env},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
