"""Single command for bounded synthetic CPU checks and persistent demo artifacts."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import time
import unittest

import torch

from r0.evaluate import candidate_pools,ensemble_errors,region_records
from r0.metrics import ScoreAccumulator
from r0.reconstruction import FMRBank
from r0.runtime import write_json
from r0.tests import test_cpu,test_protocol_repair
from r0.types import SpatialBatch
from r0.visualize import distributions


def run(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run CPU smoke with CUDA_VISIBLE_DEVICES set to an empty string')
    torch.set_num_threads(1)
    started=time.time()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module)
                             for module in (test_cpu,test_protocol_repair))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('CPU tests failed')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    all_rows=[];shapes={}
    for source in ['S1','S2']:
        torch.manual_seed(73)
        samples,geometry,_,extractor=test_cpu.synthetic(source)
        batch,_=extractor(samples,geometry)
        batch=SpatialBatch([x[:1] for x in batch.levels],[x[:1] for x in batch.padding_masks],batch.strides,geometry[:1],source)
        shapes[source]=[list(x.shape) for x in batch.levels]
        bank=FMRBank([8]*4,hidden=4,bottleneck=2).eval()
        errors=ensemble_errors(bank,batch,.6,2,73)
        gt={'boxes':torch.tensor([[2.,2.,20.,24.],[50.,30.,80.,55.]]),
            'known':torch.tensor([True,False]),'difficult':torch.tensor([False,False])}
        queries=torch.cat([gt['boxes'],torch.tensor([[85.,0.,95.,15.]])])
        pools=candidate_pools(gt,queries,geometry[0],73,{'positive_iou':.5,'background_iou':.1,'background_proposals':16})
        rows=region_records(batch,errors,pools,{'output_size':3,'sampling_ratio':2,'min_box_size':1.,'min_valid_fraction':.5})
        for row in rows:row['source']=source+'_'+row['source']
        all_rows.extend(rows)
    accumulator=ScoreAccumulator()
    for row in all_rows:accumulator.add(row)
    with gzip.open(out/'synthetic_regions.csv.gz','wt',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    figures=distributions(all_rows,out/'figures',diagnostic=True)
    assert not torch.cuda.is_initialized(), 'CPU smoke unexpectedly initialized CUDA'
    report={'passed':True,'tests':result.testsRun,'seconds':time.time()-started,
        'scope':'CPU synthetic diagnostic only; no real detector CUDA forward or reconstruction training',
        'cuda_initialized':torch.cuda.is_initialized(),'torch_threads':torch.get_num_threads(),
        'feature_shapes_synthetic_8_channels':shapes,'real_256_channel_ae_tested':True,
        'diagnostic_metrics_not_research_results':accumulator.summarize(),'figures':figures,
        'rows':len(all_rows),'torch':torch.__version__}
    write_json(out/'summary.json',report)
    print(json.dumps({k:report[k] for k in ['passed','tests','seconds','cuda_initialized','rows']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='/root/autodl-tmp/experiments/R0/cpu_smoke/verified')
    args=parser.parse_args();run(args.output)
