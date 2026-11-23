"""R1-1 evaluation is independent of any R1-0 diagnosis output or probe."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from r1.analysis import evaluate_frames
from r1.cache import FrameReader
from r1.reconstruction import QueryAutoencoder
from r1.reuse import read_eval_gt,write_json,reconstruction_config_matches
from r1.runtime import config_read


def error_provider(model,layer,config,device):
    def provide(frame):
        layers=np.asarray(frame['layer_ids']).tolist()
        if layer not in layers:raise RuntimeError('Reconstruction layer absent from this feature cache')
        slot=layers.index(layer)
        feature=np.asarray(frame[config['representation']])[slot]
        query=torch.from_numpy(feature.copy()).unsqueeze(0).to(device)
        seed=config['seed']+int(hashlib.sha256(str(frame['image_id']).encode()).hexdigest()[:8],16)
        full=[];numerator=torch.zeros(query.shape[:2],device=device);count=torch.zeros_like(numerator)
        with torch.no_grad():
            for repeat in range(config['evaluation']['repeats']):
                result=model(query,seed=(seed+repeat)%(2**63-1),mask_ratio=config['mask_ratio'])
                full.append(result['full_error'])
                numerator+=result['masked_error']*result['masked_count'];count+=result['masked_count']
        masked=(numerator/count.clamp(min=1)).squeeze(0).cpu().numpy();masked[count.squeeze(0).cpu().numpy()==0]=np.nan
        return {layer:{'full_error':torch.stack(full).mean(0).squeeze(0).cpu().numpy(),'masked_error':masked}}
    return provide


def run(path,cache,checkpoint,output,device='cpu'):
    config=config_read(path);frames=FrameReader(cache,config)
    if frames.metadata['split']!='evaluation':raise RuntimeError('Only evaluation cache can be grouped by GT')
    saved=torch.load(checkpoint,map_location=device)
    if not saved['complete'] or not reconstruction_config_matches(saved['config'],config) or saved['baseline']['sha256']!=frames.metadata['baseline']['sha256']:
        raise RuntimeError('Checkpoint/cache/config provenance mismatch')
    model=QueryAutoencoder(saved['channels'],**config['autoencoder'],mask_ratio=config['mask_ratio']).to(device)
    model.load_state_dict(saved['model'],strict=True);model.eval()
    provider=error_provider(model,saved['layer'],config,device)
    summary=evaluate_frames(frames,lambda stem:read_eval_gt(config['data_root'],stem),config['evaluation'],output,provider,
                            synthetic=frames.metadata.get('synthetic',False))
    summary['checkpoint_training_config']=saved['config']
    summary['evaluation_config']=config
    summary['cache_provenance']=frames.metadata
    summary['reconstruction_config']={key:config[key] for key in ['representation','mask_ratio','autoencoder','steps']}
    summary['baseline']=saved['baseline'];summary['trained_layer']=saved['layer']
    write_json(Path(output)/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--cache',required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();run(args.config,args.cache,args.checkpoint,args.output,args.device)
