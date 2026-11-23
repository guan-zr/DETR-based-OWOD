import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from r1.analysis import evaluate_frames
from r1.evaluate import error_provider
from r1.reconstruction import QueryAutoencoder
from r1.tests.fixtures import synthetic
from r1.train import train_step


class ReconstructionTests(unittest.TestCase):
    def test_masks_zeros_mse_and_reproducibility(self):
        torch.manual_seed(91);x=torch.zeros(2,7,8);model=QueryAutoencoder(8,hidden=8,bottleneck=3)
        empty=model(x,mask_ratio=0,seed=3);full=model(x,mask_ratio=1,seed=3)
        self.assertTrue((empty['masked_input'][...,8:]==0).all());self.assertTrue((full['masked_input'][...,8:]==1).all())
        self.assertIsNone(empty['masked_mse']);self.assertTrue(torch.allclose(full['full_error'],full['masked_error']))
        first=model(x,seed=8);second=model(x,seed=8)
        self.assertTrue(torch.equal(first['mask'],second['mask']))
        actual=(first['prediction']-x).square()
        self.assertTrue(torch.allclose(first['full_error'],actual.mean(-1)))
        self.assertTrue(torch.allclose(first['masked_error'],(actual*first['mask']).sum(-1)/first['mask'].sum(-1).clamp(min=1)))

    def test_gradients_only_update_shared_ae(self):
        detector,extractor,samples,geometry,batch,outputs,frames,gt=synthetic()
        snapshot={name:value.clone() for name,value in detector.state_dict().items()}
        model=QueryAutoencoder(8,hidden=8,bottleneck=3)
        before=[p.clone() for p in model.parameters()];optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        target=batch.raw[2].clone().requires_grad_(True)
        train_step(model,target,optimizer,91)
        self.assertIsNone(target.grad);self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in detector.parameters()))
        self.assertTrue(all(torch.equal(value,snapshot[name]) for name,value in detector.state_dict().items()))

    def test_real_config_capacity_batch_query_shapes_and_layer_fairness(self):
        model=QueryAutoencoder(256,128,32);query=torch.randn(2,100,256,requires_grad=True)
        result=model(query,seed=91);result['loss'].backward()
        self.assertEqual(result['prediction'].shape,query.shape);self.assertIsNone(query.grad)
        base=Path(__file__).resolve().parents[1]/'configs';configs=[]
        for layer in [1,3,6]:
            config=json.loads((base/f'L{layer}_mask0.6.json').read_text())
            for key in ['name','layer','output']:config.pop(key)
            configs.append(config)
        self.assertEqual(configs[0],configs[1]);self.assertEqual(configs[1],configs[2])

    def test_reconstruction_evaluation_needs_no_diagnostic_results(self):
        *_,frames,gt=synthetic()
        model=QueryAutoencoder(8,hidden=8,bottleneck=3).eval()
        config={'representation':'l2','seed':91,'mask_ratio':.6,'evaluation':{'repeats':2}}
        provider=error_provider(model,3,config,'cpu')
        with tempfile.TemporaryDirectory() as path:
            result=evaluate_frames(frames,gt.__getitem__,{'positive_iou':.5,'background_iou':.1,'low_objectness_threshold':.5},Path(path)/'reconstruction',provider,True)
            self.assertEqual(set(result['statistics']),{'3'})
            values=result['statistics']['3']['all']['scores']['full_error']['directions']
            self.assertEqual(set(values),{'higher','lower'})
            self.assertIn('3_all',result['correlations'])
            # Checkpoint I/O is an AE-only payload and does not serialize a detector.
            checkpoint=Path(path)/'ae.pth';torch.save(model.state_dict(),checkpoint)
            copy=QueryAutoencoder(8,hidden=8,bottleneck=3);copy.load_state_dict(torch.load(checkpoint,map_location='cpu'),strict=True)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.parameters(),copy.parameters())))


if __name__=='__main__':unittest.main()
