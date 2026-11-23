"""Post-B0 real-checkpoint hook equivalence check. Not executed during preparation."""
import argparse
import copy
from pathlib import Path

import torch

from r0.data import ImageOnlyDataset
from r0.features import FrozenSpatialExtractor
from r0.reconstruction import FMRBank
from r0.runtime import guard_b0_complete,load_detector,pack_images,read_config,write_json,seed_all


def compare(a,b):
    if torch.is_tensor(a):
        if not torch.equal(a,b):raise AssertionError('Hook changed detector prediction tensor')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('Prediction keys changed')
        for key in a:compare(a[key],b[key])
    elif isinstance(a,list):
        if len(a)!=len(b):raise AssertionError('Prediction list size changed')
        for left,right in zip(a,b):compare(left,right)


def run(config_path,output,device='cuda',images=2):
    config=read_config(config_path);guard_b0_complete(config);seed_all(config['seed'])
    extractor,postprocessors,baseline=load_detector(config,torch.device(device))
    detector=extractor.detector
    dataset=ImageOnlyDataset(config['data_root'],config['eval_split'],unique=True,**config['resize'])
    shapes={};reconstruction_checks={}
    with torch.no_grad():
        for index in range(min(images,len(dataset))):
            samples,geometry=pack_images([dataset[index]],torch.device(device))
            before=detector(samples)
            original_size=torch.tensor([geometry[0].original_hw],device=device)
            final_before=postprocessors['bbox'](copy.deepcopy(before),original_size)
            for source in ['S1','S2']:
                reader=FrozenSpatialExtractor(detector,source)
                batch,after=reader(samples,geometry);compare(before,after)
                final_after=postprocessors['bbox'](copy.deepcopy(after),original_size)
                compare(final_before,final_after)
                shapes[source]=[list(x.shape) for x in batch.levels]
                # Deferred real-device smoke, using image-only features and a
                # fresh local AE; this is never executed during preparation.
                with torch.enable_grad():
                    bank=FMRBank(config['channels'],**config['autoencoder'],mask_ratio=config['mask_ratio']).to(device)
                    optimizer=torch.optim.AdamW(bank.parameters(),**config['optimizer'])
                    loss,_=bank(batch,seed=config['seed'])
                    if not torch.isfinite(loss):raise AssertionError('Nonfinite real-device FMR loss')
                    loss.backward();optimizer.step()
                if any(p.grad is not None or p.requires_grad for p in detector.parameters()):
                    raise AssertionError('Detector gradients were enabled')
                reconstruction_checks[source]={'finite_loss':float(loss.detach()),'optimizer_update':True}
                del loss,bank,optimizer
    write_json(output,{'passed':True,'baseline':baseline,'raw_outputs_bitwise_equal':True,
                       'postprocessed_outputs_bitwise_equal':True,'shapes':shapes,'images':images,
                       'fresh_ae_real_device_checks':reconstruction_checks})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--output',default='/root/autodl-tmp/experiments/R0/reports/real_detector_validation.json')
    parser.add_argument('--device',default='cuda');parser.add_argument('--images',type=int,default=2)
    args=parser.parse_args();run(args.config,args.output,args.device,args.images)
