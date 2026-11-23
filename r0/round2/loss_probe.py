"""Before/after reconstruction loss on the same fixed 64 training images, never labels."""
import json,time
import numpy as np
import pandas as pd
import torch
from r0.round2.common import *
from r0.round2.train import make_bank
from r0.round2.scoring import simple_maps
from r0.evaluate import ensemble_errors,image_seed
from r0.data import ImageOnlyDataset
from r0.runtime import load_detector,pack_images,seed_all
from r0.streaming import extract_both

def main():
    out=ROOT/'loss_probe';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    seed_all(73);c=config();extractor,_,_=load_detector(c,torch.device('cuda'));banks={s:{f'trained{k}':make_bank(k,s,True).eval() for k in SEEDS}|{f'random{k}':make_bank(k).eval() for k in SEEDS} for s in SOURCES}
    mean=np.load(ROOT/'training/training_means.npz');means={s:torch.tensor(mean[s],device='cuda') for s in SOURCES};data=ImageOnlyDataset(c['data_root'],c['train_split'],**c['resize']);lookup={s:i for i,s in enumerate(data.ids)};rows=[]
    with torch.no_grad():
        for stem in sequence()[:64]:
            samples,g=pack_images([data[lookup[stem]]],torch.device('cuda'));batches,_=extract_both(extractor,samples,g)
            for source,batch in batches.items():
                maps={name:[e['error'] for e in ensemble_errors(bank,batch,.6,4,image_seed(stem,73))] for name,bank in banks[source].items()}
                maps.update({k:v for k,v in simple_maps(batch,means[source]).items() if k in ['energy','mean_error']})
                for method,levels in maps.items():
                    values=[]
                    for li,(error,padding) in enumerate(zip(levels,batch.padding_masks)):
                        value=float(error[~padding].mean());values.append(value);rows.append(dict(image_id=stem,source=source,method=method,level=LEVELS[li],mse=value))
                    rows.append(dict(image_id=stem,source=source,method=method,level='overall',mse=float(np.mean(values))))
    pd.DataFrame(rows).to_csv(out/'same_training_images_loss.csv',index=False);write_json(out/'complete.json',dict(complete=True,images=64,annotation_labels_loaded=False,scope='same training-side image/mask loss; not objectness evidence',sha256={'same_training_images_loss.csv':file_sha256(out/'same_training_images_loss.csv')}))
if __name__=='__main__':main()
