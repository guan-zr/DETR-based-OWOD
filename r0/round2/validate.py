"""Real CUDA correctness and speed probe before additional formal training."""
import json,time
import numpy as np
import pandas as pd
import torch
from r0.round2.common import *
from r0.round2.train import ensure_initializations,make_bank
from r0.round2.scoring import score_batch,simple_maps,roi_maps
from r0.runtime import load_detector,pack_images,seed_all
from r0.data import ImageOnlyDataset
from r0.streaming import extract_both
from r0.validate_detector import compare

def main():
    ensure_initializations();seed_all(73);c=config();extractor,_,baseline=load_detector(c,torch.device('cuda'))
    train=ImageOnlyDataset(c['data_root'],c['train_split'],**c['resize']);indices={s:i for i,s in enumerate(train.ids)}
    samples,g=pack_images([train[indices[sequence()[0]]]],torch.device('cuda'));batches,outputs=extract_both(extractor,samples,g);checks={}
    for source,batch in batches.items():
        bank=make_bank(73).eval()
        with torch.no_grad():loss,_=bank(batch,seed=73)
        reference=json.loads((R0/f'R0-{source}/attempt_00/train.jsonl').read_text().splitlines()[0])['loss']
        if not np.isclose(float(loss),reference,rtol=1e-6,atol=1e-7):raise RuntimeError('Cannot reproduce original seed73 pre-training loss')
        checks[source]=dict(recreated_initial_loss=float(loss),original_initial_loss=reference)
        del bank
    banks={s:{} for s in SOURCES}
    for source in SOURCES:
        for seed in SEEDS:
            banks[source]['random'+str(seed)]=make_bank(seed).eval()
            banks[source]['trained'+str(seed)]=make_bank(73,source,True).eval() if seed==73 else make_bank(seed).eval()
    means={s:torch.zeros(4,256,device='cuda') for s in SOURCES}
    # Trained74/75 slots here are explicitly fresh diagnostic banks. Never saved as trained.
    original=dict(chunks=[dict(directory=str(ROOT/'prepared'/f'{i:03d}'),**json.loads((ROOT/'prepared'/f'{i:03d}'/'complete.json').read_text())) for i in range(2)]);data=ImageOnlyDataset(c['data_root'],c['eval_split'],unique=True,**c['resize']);lookup={s:i for i,s in enumerate(data.ids)}
    durations=[];maximum=0.;nrows=0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for chunk in original['chunks'][:2]:
            meta=pd.read_csv(Path(chunk['directory'])/'metadata.csv.gz',dtype={'image_id':str},float_precision='round_trip');old=np.load(Path(chunk['directory'])/'original_scores.npz')['scores']
            for stem in chunk['image_ids'][:2]:
                keep=np.flatnonzero((meta.image_id==stem).to_numpy());f=meta.iloc[keep];boxes=torch.tensor(f[['x1','y1','x2','y2']].to_numpy(),dtype=torch.float32)
                samples,g=pack_images([data[lookup[stem]]],torch.device('cuda'));torch.cuda.synchronize();start=time.perf_counter();batches,outputs=extract_both(extractor,samples,g)
                compare(outputs,extractor.detector(samples))
                scores,v=score_batch(batches,banks,means,boxes)
                delta=np.nanmax(np.abs(scores[:,:,METHODS.index('trained73'),:]-old[keep]));maximum=max(maximum,float(delta))
                if not np.allclose(scores[:,:,0,:],old[keep],rtol=1e-5,atol=1e-6,equal_nan=True):raise RuntimeError('Seed73 original ROI scores drifted')
                for source,batch in batches.items():
                    features=simple_maps(batch,means[source]);assert all(torch.equal(a,b) for a,b in zip(features['energy'],features['mean_error']))
                torch.cuda.synchronize();durations.append(time.perf_counter()-start);nrows+=len(keep)
                print('probe',stem,'seconds',durations[-1],'seed73 max abs',delta,flush=True)
    # One real optimizer update measures four-bank shared-feature training cost.
    start=time.perf_counter()
    for source,batch in batches.items():
        for seed in [74,75]:
            bank=make_bank(seed).train();optimizer=torch.optim.AdamW(bank.parameters(),**c['optimizer'])
            with torch.enable_grad():
                loss,_=bank(batch,seed=seed);loss.backward();torch.nn.utils.clip_grad_norm_(bank.parameters(),1.0);optimizer.step()
            assert all(p.grad is None and not p.requires_grad for p in extractor.detector.parameters())
    torch.cuda.synchronize();training_probe=time.perf_counter()-start
    write_json(ROOT/'gpu_probe.json',dict(passed=True,scope='4 real evaluation images; fresh diagnostic slots for not-yet-trained74/75; zero mean timing proxy, not scientific metrics',baseline=baseline,
        original_initialization_checks=checks,reused_seed73_max_abs_drift=maximum,seconds_per_evaluation_image=durations,projected_evaluation_seconds=float(np.mean(durations)*9119),
        four_bank_training_probe_seconds_including_initialization=training_probe,projected_training_upper_seconds=training_probe*2000,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,rows=nrows,feature_cache='none'))
if __name__=='__main__':main()
