"""CPU regressions for backend preservation and strict official-trace rejection."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from b0.eval_parity import (TraceWriter,TraceComparator,backend_profile,seed_official_evaluation,
                            tensor_sha256,raw_snapshot)


class EvaluationParityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        split=self.root/'ImageSets/TOWOD';split.mkdir(parents=True);(split/'test.txt').write_text('one\ntwo\n')
        self.checkpoint=self.root/'fixture.pth';self.checkpoint.write_bytes(b'CPU-only provenance fixture')
        self.args=SimpleNamespace(data_root=str(self.root),dataset='TOWOD',test_set='test',batch_size=2,eval_batch_size=2,seed=42)
        self.dataset=SimpleNamespace(image_set=['one','two'],CLASS_NAMES=['known','unknown'])
        self.samples=SimpleNamespace(tensors=torch.arange(24,dtype=torch.float32).reshape(2,3,2,2),mask=torch.zeros(2,2,2,dtype=torch.bool))
        self.outputs={key:torch.arange(12,dtype=torch.float32).reshape(2,2,3) for key in ('pred_logits','pred_boxes','pred_obj','penu_layer')}
        self.predictions={i:{'boxes':torch.tensor([[0.,0,1,1]]),'scores':torch.tensor([.5]),'labels':torch.tensor([1])} for i in (1,2)}
        self.evaluator=SimpleNamespace(lines=['one 0.500 1.0 1.0 1.0 1.0','two 0.500 1.0 1.0 1.0 1.0'],lines_cls=[1,1])
        writer=TraceWriter(self.root/'trace',self.args,self.dataset,self.checkpoint)
        writer.add(raw_snapshot(self.samples,self.outputs),self.predictions,self.evaluator,0)
        self.metrics={'K_AP50':12.5,'U_R50':25.};writer.close(self.metrics)

    def reader(self,name='out',args=None):
        out=self.root/name;out.mkdir(exist_ok=True)
        return TraceComparator(self.root/'trace',args or self.args,self.dataset,self.checkpoint,out)

    def test_official_seed_preserves_backend_and_reproduces_rng(self):
        before=backend_profile();seed_official_evaluation(42);first=torch.rand(4);a=np.random.rand(4)
        seed_official_evaluation(42)
        self.assertEqual(before,backend_profile());self.assertTrue(torch.equal(first,torch.rand(4)))
        np.testing.assert_array_equal(a,np.random.rand(4));self.assertFalse(torch.cuda.is_initialized())

    def test_identical_trace_and_strict_metric_gate(self):
        reader=self.reader();reader.compare(self.samples,self.outputs,self.predictions,self.evaluator,0)
        self.assertTrue(reader.close(self.metrics)['final_detections_bitwise_identical'])
        with self.assertRaisesRegex(AssertionError,'metric mismatch'):reader.close(dict(self.metrics,K_AP50=12.500001))
        with self.assertRaisesRegex(AssertionError,'metric mismatch'):reader.close(dict(self.metrics,K_AP50=float('nan')))

    def test_input_raw_and_final_differences_are_rejected(self):
        samples=copy.deepcopy(self.samples);samples.tensors[0,0,0,0]+=1
        with self.assertRaisesRegex(AssertionError,'input_sha256'):self.reader('input').compare(samples,self.outputs,self.predictions,self.evaluator,0)
        outputs=copy.deepcopy(self.outputs);outputs['pred_boxes'][0,0,0]+=1
        with self.assertRaisesRegex(AssertionError,'raw_sha256'):self.reader('raw').compare(self.samples,outputs,self.predictions,self.evaluator,0)
        predictions=copy.deepcopy(self.predictions);predictions[1]['scores'][0]+=.001
        with self.assertRaisesRegex(AssertionError,'Final detection'):self.reader('final').compare(self.samples,self.outputs,predictions,self.evaluator,0)

    def test_wrong_split_backend_and_batch_are_rejected(self):
        args=copy.deepcopy(self.args);args.eval_batch_size=1
        with self.assertRaisesRegex(RuntimeError,'signature'):self.reader('batch',args)
        path=self.root/'trace/manifest.json';original=path.read_text();manifest=json.loads(original)
        manifest['backend']['cudnn_tf32']=not manifest['backend']['cudnn_tf32'];path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError,'backend'):self.reader('backend')
        path.write_text(original);(self.root/'ImageSets/TOWOD/test.txt').write_text('two\none\n')
        with self.assertRaisesRegex(RuntimeError,'split_sha256'):self.reader('split')

    def test_missing_batches_corruption_and_evaluator_lines_are_rejected(self):
        with self.assertRaisesRegex(AssertionError,'complete official trace'):self.reader('missing').close(self.metrics)
        evaluator=copy.deepcopy(self.evaluator);evaluator.lines[0]+=' changed'
        with self.assertRaisesRegex(AssertionError,'serialized input'):self.reader('lines').compare(self.samples,self.outputs,self.predictions,evaluator,0)
        payload=self.root/'trace/batch_00000.npz';payload.write_bytes(payload.read_bytes()+b'changed')
        with self.assertRaisesRegex(RuntimeError,'payload checksum'):self.reader('corrupt').compare(self.samples,self.outputs,self.predictions,self.evaluator,0)


if __name__=='__main__':unittest.main()
