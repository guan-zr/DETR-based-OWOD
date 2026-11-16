"""Official learned selection; separate from training, no validation/test GT."""
import argparse, collections, json, pickle, time
from pathlib import Path
from torch.utils.data import DataLoader
from b1.common import *
from main_open_world import create_ft_dataset
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import OWDetection
from util.misc import collate_fn
from engine import get_exemplar_replay

def generate(stage):
 config='rebuild_task1_exemplar' if stage=='task1' else 'stage_a'; a=args_for(config); seed()
 dst=E/'replay'/a.exemplar_replay_cur_file; metadata=dst.with_suffix('.provenance.json')
 if metadata.exists():
  old=json.loads(metadata.read_text()); assert old['checkpoint_sha256']==sha(a.pretrain if stage=='task1' else E/'checkpoints/task2/checkpoint0050.pth') and old['index_sha256']==sha(dst)
  print('Verified existing replay',dst,flush=True); return
 disk_check(); model,criterion,post,selection=build(a)
 source=a.pretrain if stage=='task1' else str(E/'checkpoints/task2/checkpoint0050.pth')
 checkpoint=strict_load(model,source); del checkpoint
 dataset=OWDetection(a,a.data_root,image_set=a.train_set,transforms=make_coco_transforms(a.train_set),dataset=a.dataset)
 # Official selection sees the train loader, with train augmentation and current-only GT.
 # A fixed seed reconstructs the absent Task1 selection RNG; drop_last=False avoids losing last singleton.
 generator=torch.Generator().manual_seed(a.seed)
 loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=True,generator=generator,drop_last=False,num_workers=a.num_workers,collate_fn=collate_fn,pin_memory=True)
 started=time.time(); scores=get_exemplar_replay(model,selection,torch.device('cuda'),loader)
 with (E/'replay'/f'{stage}_selection_scores.pkl').open('wb') as f: pickle.dump(scores,f)
 # Reset only final shuffling seed for a documented, repeatable file order.
 np.random.seed(a.seed); create_ft_dataset(a,scores)
 stems=dst.read_text().splitlines(); allowed=set(dataset.image_set)
 prev=[]
 if a.exemplar_replay_prev_file:
  prev=(E/'replay'/a.exemplar_replay_prev_file).read_text().splitlines(); allowed.update(prev)
 assert set(stems)<=allowed and len(stems)==len(set(stems)) and len(stems)<=a.exemplar_replay_max_length
 from datasets.torchvision_datasets.open_world import VOC_CLASS_NAMES_COCOFIED,BASE_VOC_CLASS_NAMES
 import xml.etree.ElementTree as ET
 aliases=dict(zip(VOC_CLASS_NAMES_COCOFIED,BASE_VOC_CLASS_NAMES)); counts=collections.Counter()
 for stem in stems:
  for obj in ET.parse(Path(a.data_root)/'Annotations'/f'{stem}.xml').getroot().findall('object'):
   n=aliases.get(obj.findtext('name'),obj.findtext('name')); idx=dataset.CLASS_NAMES.index(n)
   if idx<a.PREV_INTRODUCED_CLS+a.CUR_INTRODUCED_CLS: counts[idx]+=1
 assert all(counts[i]>0 for i in range(a.PREV_INTRODUCED_CLS+a.CUR_INTRODUCED_CLS))
 provenance=dict(at=now(),stage=stage,checkpoint=source,checkpoint_sha256=sha(source),index=str(dst),index_sha256=sha(dst),source_split=a.train_set,source_rows=len(dataset),selection_images=len(scores),seed=a.seed,selection='official ExemplarSelection + create_ft_dataset; both score tails;50 instances/class',max_images=a.exemplar_replay_max_length,images=len(stems),previous_index=a.exemplar_replay_prev_file,previous_images_retained=len(set(stems)&set(prev)),supervised_counts_by_class=dict(sorted(counts.items())),seconds=time.time()-started,
 compatibility_differences=['deterministic seed42 instead of unavailable end-of-Task1 loader RNG','drop_last=False selection-only to score every allowed train image','single GPU batch2 instead of4x5; official train augmentation retained'],test_gt_used=False,future_unknown_gt_used_for_selection=False)
 write(metadata,provenance); print('REPLAY_COMPLETE '+json.dumps(provenance),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['task1','task2'],required=True);o=p.parse_args();generate(o.stage)
