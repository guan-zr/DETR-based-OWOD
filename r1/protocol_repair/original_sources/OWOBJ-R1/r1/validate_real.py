"""Deferred real-checkpoint interface check; guarded until B0 is fully finished."""
import argparse
import copy
import torch
from r1.features import image_frames
from r1.reuse import ImageOnlyDataset,pack_images,write_json
from r1.runtime import config_read,load_frozen


def run(path,output,device='cuda'):
    config=config_read(path);extractor,postprocessors,baseline=load_frozen(config,torch.device(device))
    dataset=ImageOnlyDataset(config['data_root'],config['eval_split'],unique=True,**config['resize'])
    samples,geometry=pack_images([dataset[0]],torch.device(device))
    with torch.no_grad():
        before=extractor.detector(samples);batch,after=extractor(samples)
        for key in ['pred_logits','pred_boxes','pred_obj','penu_layer']:
            if not torch.equal(before[key],after[key]):raise AssertionError('Real hook changed '+key)
        frames=list(image_frames(batch,geometry,postprocessors['bbox'].invalid_cls_logits,
                                 postprocessors['bbox'].temperature,postprocessors['bbox'].pred_per_im))
        final=postprocessors['bbox'](copy.deepcopy(before),torch.tensor([geometry[0].original_hw],device=device))[0]
        import numpy as np
        np.testing.assert_array_equal(final['boxes'].cpu().numpy(),frames[0]['final_boxes_xyxy'])
        np.testing.assert_array_equal(final['scores'].cpu().numpy(),frames[0]['final_pair_scores'])
    write_json(output,{'passed':True,'baseline':baseline,'shape':list(batch.raw.shape),
                       'layers_saved':frames[0]['layer_ids'],'own_layer_features_and_boxes_verified':True,
                       'real_forward_and_final_selection_identical':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--output',default='/root/autodl-tmp/experiments/R1/reports/real_interface.json');parser.add_argument('--device',default='cuda')
    args=parser.parse_args();run(args.config,args.output,args.device)
