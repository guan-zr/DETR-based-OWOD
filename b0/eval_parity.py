"""Read-only evaluation traces. No scoring, postprocessing, or tolerance changes."""
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from b0.common import json_write

FIELDS=('pred_logits','pred_boxes','pred_obj','penu_layer')


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def tensor_sha256(tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    h=hashlib.sha256(str((array.dtype.str,array.shape)).encode());h.update(array.tobytes())
    return h.hexdigest()


def backend_profile():
    return {'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}


def seed_official_evaluation(seed):
    # Exactly main_open_world.main's RNG seeding. In particular, do not call
    # b0.common.seed_all: it additionally disables cuDNN TF32 for training.
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)


def evaluation_signature(args,batch_size):
    ignored={'accumulation_steps','batch_size','eval_batch_size','epochs','eval','output_dir',
             'pretrain','resume','start_epoch','distributed','world_size','rank','gpu','dist_url','dist_backend'}
    values={k:v for k,v in vars(args).items() if k not in ignored}
    values['evaluation_batch_size']=batch_size
    return values


def raw_snapshot(samples,outputs):
    return {'input_sha256':tensor_sha256(samples.tensors),'mask_sha256':tensor_sha256(samples.mask),
            'input_shape':list(samples.tensors.shape),
            'raw_sha256':[tensor_sha256(outputs[key]) for key in FIELDS]}


def lines_sha256(lines,classes):
    return hashlib.sha256(json.dumps([list(lines),list(classes)],separators=(',',':')).encode()).hexdigest()


class TraceWriter:
    def __init__(self,path,args,dataset,checkpoint):
        self.root=Path(path);self.root.mkdir(parents=True,exist_ok=False)
        self.ids=[];self.index=[]
        split=Path(args.data_root)/'ImageSets'/args.dataset/(args.test_set+'.txt')
        self.metadata={'schema':'b0-official-eval-trace-v1','complete':False,
            'checkpoint':{'path':str(Path(checkpoint).resolve()),'sha256':file_sha256(checkpoint)},
            'split_file':str(split),'split_sha256':file_sha256(split),
            'image_set':list(dataset.image_set),'class_names':list(dataset.CLASS_NAMES),
            'signature':evaluation_signature(args,args.batch_size),'backend':backend_profile(),
            'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'source':'unmodified main_open_world.main -> engine.evaluate -> official PostProcess -> OWEvaluator'}
        json_write(self.root/'manifest.json',self.metadata)

    def add(self,snapshot,predictions,evaluator,lines_start):
        ids=list(predictions);batch=len(self.index)
        arrays={'image_ids':np.array(ids,dtype=np.int64),
                **{k:np.stack([v[k].detach().cpu().numpy() for v in predictions.values()]) for k in ('boxes','scores','labels')}}
        path=self.root/f'batch_{batch:05}.npz';np.savez_compressed(path,**arrays)
        entry={'batch_index':batch,**snapshot,'image_ids':ids,'file':path.name,'sha256':file_sha256(path),
               'eval_lines_sha256':lines_sha256(evaluator.lines[lines_start:],evaluator.lines_cls[lines_start:])}
        self.index.append(entry);self.ids.extend(ids)
        with (self.root/'index.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')

    def close(self,metrics):
        self.metadata.update(complete=True,batches=len(self.index),evaluator_image_occurrences=len(self.ids),
                             input_rows=sum(x['input_shape'][0] for x in self.index),
                             metrics={k:(float(v) if np.isfinite(v) else None) for k,v in metrics.items()},
                             index_sha256=file_sha256(self.root/'index.jsonl'))
        json_write(self.root/'manifest.json',self.metadata)


class TraceComparator:
    def __init__(self,path,args,dataset,checkpoint,output):
        self.root=Path(path);self.metadata=json.loads((self.root/'manifest.json').read_text())
        if self.metadata.get('schema')!='b0-official-eval-trace-v1' or not self.metadata.get('complete'):
            raise RuntimeError('Official detection trace incomplete/incompatible')
        if file_sha256(self.root/'index.jsonl')!=self.metadata['index_sha256']:raise RuntimeError('Trace index checksum mismatch')
        self.index=[json.loads(line) for line in (self.root/'index.jsonl').read_text().splitlines()]
        expected={'checkpoint':{'path':str(Path(checkpoint).resolve()),'sha256':file_sha256(checkpoint)},
                  'signature':evaluation_signature(args,args.eval_batch_size),'image_set':list(dataset.image_set),
                  'class_names':list(dataset.CLASS_NAMES),'backend':backend_profile(),
                  'split_sha256':file_sha256(Path(args.data_root)/'ImageSets'/args.dataset/(args.test_set+'.txt'))}
        for key,value in expected.items():
            if value!=self.metadata.get(key):raise RuntimeError('Official trace provenance mismatch: '+key)
        self.output=Path(output);self.count=0;self.images=0;self.log=self.output/'per_image_parity.jsonl'
        if self.log.exists():raise RuntimeError('Refuse to overwrite per-image parity evidence')

    def compare(self,samples,outputs,predictions,evaluator,lines_start):
        if self.count>=len(self.index):raise RuntimeError('More analysis batches than official trace')
        expected=self.index[self.count];snapshot=raw_snapshot(samples,outputs)
        for key in ('input_sha256','mask_sha256','input_shape','raw_sha256'):
            if snapshot[key]!=expected[key]:raise AssertionError(f'First differing stage: batch {self.count}, {key}')
        if list(predictions)!=expected['image_ids']:raise AssertionError('Official image/batch ordering mismatch')
        path=self.root/expected['file']
        if file_sha256(path)!=expected['sha256']:raise RuntimeError('Official trace payload checksum mismatch')
        with np.load(path,allow_pickle=False) as saved, self.log.open('a') as log:
            for i,(image_id,result) in enumerate(predictions.items()):
                for key in ('boxes','scores','labels'):
                    if not np.array_equal(result[key].detach().cpu().numpy(),saved[key][i]):
                        raise AssertionError(f'Final detection mismatch: batch {self.count}, image {image_id}, {key}')
                log.write(json.dumps({'batch_index':self.count,'image_id':image_id,
                                     'boxes_bitwise_equal':True,'scores_bitwise_equal':True,'labels_identical':True})+'\n')
        line_hash=lines_sha256(evaluator.lines[lines_start:],evaluator.lines_cls[lines_start:])
        if line_hash!=expected['eval_lines_sha256']:raise AssertionError('Official evaluator serialized input mismatch')
        self.count+=1;self.images+=len(predictions)

    def close(self,metrics):
        if self.count!=len(self.index):raise AssertionError('Analysis did not consume complete official trace')
        for key,value in self.metadata['metrics'].items():
            if value is None:
                if np.isfinite(metrics[key]):raise AssertionError('Official undefined metric mismatch: '+key)
            elif not np.isfinite(metrics[key]) or abs(metrics[key]-value)>=1e-9:
                raise AssertionError('Official trace metric mismatch: '+key)
        return {'passed':True,'trace':str(self.root),'trace_manifest_sha256':file_sha256(self.root/'manifest.json'),
                'batches':self.count,'evaluator_image_occurrences':self.images,'input_rows':self.metadata['input_rows'],
                'preprocessing_bitwise_identical':True,'raw_prediction_hashes_identical':True,
                'final_detections_bitwise_identical':True,'evaluator_serialized_inputs_identical':True,
                'metrics_match_original_tolerance_1e-9':True,'backend':backend_profile()}
