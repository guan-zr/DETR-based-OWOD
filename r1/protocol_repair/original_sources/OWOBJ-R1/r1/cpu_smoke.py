"""Separate repeatable CPU suites. Exclusive-stage failures do not block siblings."""
import argparse
import importlib
import os
from pathlib import Path
import time
import unittest
import torch
from r1.reuse import write_json


def run(stage,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Explicitly hide GPUs for CPU smoke')
    torch.set_num_threads(1);torch.manual_seed(91)
    mapping={'common':'r1.tests.test_common','r1-0':'r1.tests.test_diagnostic','r1-1':'r1.tests.test_reconstruction'}
    stages=list(mapping) if stage=='all' else [stage];reports={}
    for name in stages:
        start=time.time();module=importlib.import_module(mapping[name])
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
        reports[name]={'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),
                       'errors':len(result.errors),'seconds':time.time()-start}
        if name=='common' and not result.wasSuccessful():break
    assert not torch.cuda.is_initialized()
    report={'scope':'CPU synthetic only','stages':reports,'cuda_initialized':False,'torch_threads':1}
    write_json(Path(output)/f'{stage}.json',report)
    if any(not x['passed'] for x in reports.values()):raise RuntimeError('One or more CPU stages failed; see separate results')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['common','r1-0','r1-1','all'],default='all')
    parser.add_argument('--output',default='/root/autodl-tmp/experiments/R1/cpu_smoke/results');args=parser.parse_args()
    run(args.stage,args.output)
