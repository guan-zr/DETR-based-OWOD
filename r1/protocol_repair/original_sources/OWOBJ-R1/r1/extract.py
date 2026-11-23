"""Feature cache preparation only, independent from both diagnosis and AE results."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from r1.cache import FrameWriter
from r1.features import image_frames
from r1.reuse import ImageOnlyDataset,image_collate,pack_images,seed_all,file_sha256
from r1.runtime import config_read,load_frozen


def run(config_path,output,split='evaluation',device='cuda',all_layers=False):
    config=config_read(config_path)
    extractor,postprocessors,baseline=load_frozen(config,torch.device(device));seed_all(config['seed'])
    split_name=config['eval_split'] if split=='evaluation' else config['train_split']
    dataset=ImageOnlyDataset(config['data_root'],split_name,unique=True,**config['resize'])
    loader=DataLoader(dataset,batch_size=config['extract_batch_size'],shuffle=False,
                      num_workers=0,collate_fn=image_collate)
    writer=FrameWriter(output,{'baseline':baseline,'split':split,'split_file':split_name,
        'split_sha256':file_sha256(Path(config['data_root'])/'ImageSets/TOWOD'/f'{split_name}.txt'),
        'normalizations':['actual_post_norm3','explicit_L2','actual_pre_norm3'],
        'selection':'OWOBJ sigmoid global Top100 query/class pairs; final layer actual, intermediate layers simulation',
        'synthetic':False},config['max_cache_gib'])
    for index,items in enumerate(loader):
        samples,geometry=pack_images(items,torch.device(device));batch,_=extractor(samples)
        layers=list(range(1,batch.raw.shape[0]+1)) if all_layers else None
        for frame in image_frames(batch,geometry,postprocessors['bbox'].invalid_cls_logits,
                                  postprocessors['bbox'].temperature,postprocessors['bbox'].pred_per_im,layers):
            writer.write(frame)
        if index%100==0:print(f'Query cache {index}/{len(loader)}',flush=True)
    writer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--split',choices=['train','evaluation'],default='evaluation');parser.add_argument('--device',default='cuda')
    parser.add_argument('--all-layers',action='store_true');args=parser.parse_args()
    run(args.config,args.output,args.split,args.device,args.all_layers)
