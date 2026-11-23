"""Real CUDA forward, deterministic AE parity, checkpoint/error/output smoke."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from r1.runtime import config_read,load_frozen
from r1.reuse import ImageOnlyDataset,pack_images,seed_all,write_json,read_eval_gt
from r1.features import image_frames
from r1.cache import validate_frame
from r1.train import train_step
from r1.reconstruction import QueryAutoencoder
from r1.evaluate import error_provider
from r1.analysis import evaluate_frames


def digest(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(config_path,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True);c=config_read(config_path);device=torch.device('cuda');extractor,post,baseline=load_frozen(c,device);before=digest(extractor.detector)
    train=ImageOnlyDataset(c['data_root'],c['train_split'],**c['resize']);test=ImageOnlyDataset(c['data_root'],c['eval_split'],unique=True,**c['resize'])
    assert not set(train.ids)&set(test.ids)
    samples,_=pack_images([train[0]],device);q,_=extractor(samples);assert tuple(q.raw.shape)==(6,1,100,256)
    sample,geometry=pack_images([test[0]],device);batch,_=extractor(sample);frame=list(image_frames(batch,geometry,post['bbox'].invalid_cls_logits,post['bbox'].temperature,post['bbox'].pred_per_im,list(range(1,7))))[0];validate_frame(frame)
    records=[]
    for rep in ['l2','raw']:
        for mask in [.6,0.,.3]:
            for layer in [1,3,6]:
                conf=dict(c,representation=rep,mask_ratio=mask,layer=layer)
                seed_all(91);m=QueryAutoencoder(256,**c['autoencoder'],mask_ratio=mask).cuda();reference=copy.deepcopy(m);opt=torch.optim.AdamW(m.parameters(),**c['optimizer']);refopt=torch.optim.AdamW(reference.parameters(),**c['optimizer'])
                for step in range(3):
                    train_step(m,q.representation(rep)[layer-1],opt,91+step);train_step(reference,q.representation(rep)[layer-1],refopt,91+step)
                assert all(torch.equal(a,b) for a,b in zip(m.state_dict().values(),reference.state_dict().values()))
                provider=error_provider(m.eval(),layer,conf,'cuda');errors=provider(frame)[layer];assert np.isfinite(errors['full_error']).all()
                assert np.isnan(errors['masked_error']).all() if mask==0 else np.isfinite(errors['masked_error']).all()
                if rep=='l2' and mask==.6 and layer==1:
                    torch.save(dict(model=m.state_dict(),config=conf,baseline=baseline,step=3,smoke_only=True),out/'smoke_checkpoint.pth')
                    summary=evaluate_frames([frame],lambda stem:read_eval_gt(c['data_root'],stem),c['evaluation'],out/'evaluation',provider)
                    assert summary['images']==1 and 'foreground_vs_background' in summary['statistics']['1']['all']['scores']['full_error']['directions']['higher']
                records.append(dict(representation=rep,mask_ratio=mask,layer=layer,steps=3,parity=True,finite_full_error=True))
    after=digest(extractor.detector);assert before==after;assert all(not p.requires_grad and p.grad is None for p in extractor.detector.parameters())
    write_json(out/'complete.json',dict(passed=True,baseline=baseline,shape=list(q.raw.shape),train_eval_overlap=0,smoke_configs=records,frozen_detector_state_sha256=before,detector_state_unchanged=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.config,a.output)
