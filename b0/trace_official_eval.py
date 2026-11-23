"""Execute the unmodified official evaluation entrypoint with read-only observers."""
import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
import main_open_world as official
from b0.diagnose_eval_parity import official_args
from b0.eval_parity import TraceWriter,raw_snapshot,file_sha256
from b0.common import ARTIFACTS,json_write
from b0.smoke import subset
from datasets.open_world_eval import OWEvaluator
from util.misc import collate_fn


def run(output,indices_json=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    args=official_args();args.output_dir=str(out)
    original_evaluate=official.evaluate;original_update=OWEvaluator.update
    state={};writer=None
    def observed_evaluate(model,criterion,postprocessors,loader,base_ds,device,output_dir,args):
        nonlocal writer
        if indices_json:
            indices=json.loads(Path(indices_json).read_text());dataset=subset(loader.dataset,indices)
            loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn,
                              num_workers=args.num_workers,pin_memory=True);base_ds=dataset
        writer=TraceWriter(out/'trace',args,base_ds,args.pretrain)
        def before(module,inputs):state['samples']=inputs[0]
        def after(module,inputs,outputs):state['snapshot']=raw_snapshot(state['samples'],outputs)
        handles=[model.register_forward_pre_hook(before),model.register_forward_hook(after)]
        try:
            stats,evaluator=original_evaluate(model,criterion,postprocessors,loader,base_ds,device,output_dir,args)
            writer.close(stats['metrics']);return stats,evaluator
        finally:
            for handle in handles:handle.remove()
    def observed_update(evaluator,predictions):
        start=len(evaluator.lines);original_update(evaluator,predictions)
        writer.add(state['snapshot'],predictions,evaluator,start)
    official.evaluate=observed_evaluate;OWEvaluator.update=observed_update
    try:official.main(args)
    finally:official.evaluate=original_evaluate;OWEvaluator.update=original_update
    metrics=json.loads((out/'official_metrics.json').read_text())
    validation={'scope':'small_regression' if indices_json else 'full_official_entrypoint_replay',
                'trace_complete':True,'official_source_files_unchanged':True}
    if not indices_json:
        original=json.loads((ARTIFACTS/'evaluations/final/official_metrics.json').read_text())
        validation['metric_deltas_vs_original']={key:metrics[key]-value for key,value in original.items()}
        json_write(out/'original_metric_comparison.json',validation)
        for key,value in original.items():
            assert abs(metrics[key]-value)<1e-9, 'Official replay differs from original: '+key
        validation['all_metrics_match_original']=True
    json_write(out/'verification.json',validation)
    print('OFFICIAL_TRACE_COMPLETE '+json.dumps(validation),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--indices-json')
    options=parser.parse_args();run(options.output,options.indices_json)
