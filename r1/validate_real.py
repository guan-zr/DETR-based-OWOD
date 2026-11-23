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
    # Deferred measurement on actual B0 queries. Batch=2 is diagnostic only;
    # formal training/extraction remains batch=1.
    def compare(first,second):
        result={}
        for name in ('raw','l2','pre_layernorm','boxes','logits','distance'):
            a=first.representation(name) if name in ('raw','l2','pre_layernorm') else getattr(first,name)
            b=second.representation(name) if name in ('raw','l2','pre_layernorm') else getattr(second,name)
            a=a[:,0];b=b[:,0]
            result[name]=[{'layer':layer+1,'max_abs':float((x-y).abs().max()),
                          'rms_difference':float((x-y).square().mean().sqrt()),
                          'allclose_rtol1e5_atol1e6':bool(torch.allclose(x,y,rtol=1e-5,atol=1e-6))}
                         for layer,(x,y) in enumerate(zip(a,b))]
        return result
    first=dataset[0];repeat,_=extractor(samples)
    repeated=compare(batch,repeat)
    if not all(row['allclose_rtol1e5_atol1e6'] for rows in repeated.values() for row in rows):
        raise AssertionError('Real batch=1 repeat exceeds fixed tolerance')
    companion=None
    for index in range(1,min(len(dataset),32)):
        candidate=dataset[index]
        if any(b>a for a,b in zip(first['resized_hw'],candidate['resized_hw'])):
            companion=candidate;break
    if companion is None:raise RuntimeError('No padding-changing companion in first 32 images; real batch diagnostic incomplete')
    paired_samples,paired_geometry=pack_images([first,companion],torch.device(device))
    paired,_=extractor(paired_samples)
    padding_comparison={'scope':'measured real B0 queries; no GT or synthetic GroupNorm inference',
                        'first_image_id':first['image_id'],'companion_image_id':companion['image_id'],
                        'standalone_resized_hw':geometry[0].resized_hw,'standalone_padded_hw':geometry[0].padded_hw,
                        'paired_padded_hw':paired_geometry[0].padded_hw,
                        'batch1_repeat':repeated,'batch1_vs_batch2':compare(batch,paired),
                        'formal_strategy':'training batch=1 and evaluation extraction batch=1'}
    write_json(output,{'passed':True,'real_query_batch_padding_measurement':padding_comparison,'baseline':baseline,'shape':list(batch.raw.shape),
                       'layers_saved':frames[0]['layer_ids'],'own_layer_features_and_boxes_verified':True,
                       'real_forward_and_final_selection_identical':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--output',default='/root/autodl-tmp/experiments/R1/reports/real_interface.json');parser.add_argument('--device',default='cuda')
    args=parser.parse_args();run(args.config,args.output,args.device)
