import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from r0.data import ImageOnlyDataset
from r0.evaluate import candidate_pools,ensemble_errors,region_records
from r0.features import FrozenSpatialExtractor,unflatten_memory
from r0.labels import label_regions
from r0.metrics import ScoreAccumulator,binary_metrics
from r0.reconstruction import FMRBank,spatial_mask
from r0.regions import mean_region_error
from r0.runtime import StepBatches,guard_b0_complete
from r0.types import ImageGeometry,SpatialBatch
from r0.visualize import distributions,group_distributions


class FakeBackbone(nn.Module):
    strides=[2,4,8]


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(8,8);self.dropout=nn.Dropout(0.5);self.bn=nn.BatchNorm1d(8)
    def forward(self,x):return self.dropout(self.bn(self.linear(x).transpose(1,2)).transpose(1,2))


class FakeTransformer(nn.Module):
    def __init__(self):super().__init__();self.encoder=FakeEncoder()
    def forward(self,srcs,masks,pos,queries):
        return self.encoder(torch.cat([x.flatten(2).transpose(1,2) for x in srcs],1))


class FakeDetector(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=FakeBackbone();self.transformer=FakeTransformer()
        self.proj=nn.Conv2d(3,8,1);self.head=nn.Linear(8,12)
    def forward(self,samples):
        srcs=[F.avg_pool2d(self.proj(samples.tensors),stride) for stride in [2,4,8,16]]
        masks=[F.interpolate(samples.mask[:,None].float(),size=x.shape[-2:],mode='nearest')[:,0].bool() for x in srcs]
        memory=self.transformer(srcs,masks,[torch.zeros_like(x) for x in srcs],None)
        values=self.head(memory.mean(1)).reshape(-1,3,4)
        return {'pred_boxes':values.sigmoid(),'pred_logits':values}


def synthetic(source='S1'):
    from types import SimpleNamespace
    tensor=torch.randn(2,3,32,48)
    mask=torch.zeros(2,32,48,dtype=torch.bool);mask[1,24:]=True;mask[1,:,32:]=True
    samples=SimpleNamespace(tensors=tensor,mask=mask)
    geometry=[ImageGeometry('one',(64,96),(32,48),(32,48)),ImageGeometry('two',(24,32),(24,32),(32,48))]
    detector=FakeDetector().eval()
    extractor=FrozenSpatialExtractor(detector,source)
    return samples,geometry,detector,extractor


class CPUSmoke(unittest.TestCase):
    def setUp(self):torch.manual_seed(73);torch.set_num_threads(1)

    def test_unflatten_exact_token_order_and_shape_validation(self):
        a=torch.arange(24).reshape(1,2,3,4).float();b=100+torch.arange(4).reshape(1,2,1,2).float()
        memory=torch.cat([a.flatten(2).transpose(1,2),b.flatten(2).transpose(1,2)],1)
        output=unflatten_memory(memory,[(3,4),(1,2)])
        self.assertTrue(torch.equal(a,output[0]));self.assertTrue(torch.equal(b,output[1]))
        with self.assertRaises(ValueError):unflatten_memory(memory,[(2,4)])

    def test_hook_predictions_freezing_and_backward_both_sources(self):
        for source in ['S1','S2']:
            samples,geometry,detector,extractor=synthetic(source)
            with torch.no_grad():before=detector(samples)
            state={key:value.clone() for key,value in detector.state_dict().items()}
            extractor.train(True);self.assertFalse(detector.training)
            batch,after=extractor(samples,geometry)
            for key in before:self.assertTrue(torch.equal(before[key],after[key]))
            self.assertEqual([list(x.shape) for x in batch.levels],[[2,8,16,24],[2,8,8,12],[2,8,4,6],[2,8,2,3]])
            self.assertEqual(batch.strides,[2,4,8,16])
            self.assertTrue(all(not x.requires_grad for x in batch.levels))
            bank=FMRBank([8]*4,hidden=4,bottleneck=2,depth=1)
            optimizer=torch.optim.AdamW(bank.parameters(),lr=0.001)
            original=[p.clone() for p in bank.parameters()]
            loss,result=bank(batch,seed=73);loss.backward();optimizer.step()
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(original,bank.parameters())))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in detector.parameters()))
            self.assertTrue(all(torch.equal(state[key],value) for key,value in detector.state_dict().items()))
            self.assertEqual([r['error'].shape for r in result],[x.shape[:1]+x.shape[2:] for x in batch.levels])
            for row,padding in zip(result,batch.padding_masks):
                self.assertTrue((row['error'][padding]==0).all())

    def test_mask_reproducibility_ratio_padding_and_no_mask(self):
        padding=torch.zeros(2,100,100,dtype=torch.bool);padding[:,80:]=True
        a=spatial_mask(padding,.6,torch.Generator().manual_seed(1))
        b=spatial_mask(padding,.6,torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(a,b));self.assertFalse(a[padding].any())
        self.assertAlmostEqual(float(a.sum()/(~padding).sum()),.6,delta=.02)
        self.assertFalse(spatial_mask(padding,0).any());self.assertTrue(torch.equal(spatial_mask(padding,1),~padding))
        samples,geometry,_,extractor=synthetic();batch,_=extractor(samples,geometry)
        _,rows=FMRBank([8]*4,hidden=4,bottleneck=2)(batch,mask_ratio=0)
        self.assertTrue(all(r['masked_count']==0 and r['masked_mse'] is None for r in rows))

    def test_targets_detached_even_for_gradient_enabled_inputs(self):
        feature=torch.randn(1,8,5,7,requires_grad=True)
        batch=SpatialBatch([feature],[torch.zeros(1,5,7,dtype=torch.bool)],[8],
                           [ImageGeometry('x',(40,56),(40,56),(40,56))],'S1')
        bank=FMRBank([8],hidden=4,bottleneck=2)
        loss,_=bank(batch,seed=4);loss.backward()
        self.assertIsNone(feature.grad)

    def test_roi_constant_ramp_resize_stride_padding_invalid_and_tiny(self):
        geometry=ImageGeometry('roi',(16,16),(8,8),(12,12))
        valid=torch.zeros(6,6,dtype=torch.bool);valid[:4,:4]=True
        error=torch.full((6,6),10000.);error[valid]=3.
        boxes=torch.tensor([[0.,0.,16.,16.],[-5.,-5.,16.,16.],[4.,4.,4.,8.],
                            [0.,0.,.1,.1],[float('nan'),0,5,5]])
        score=mean_region_error(error,valid,boxes,geometry,2)
        self.assertTrue(torch.allclose(score.scores[:2],torch.tensor([3.,3.])))
        self.assertEqual(score.valid.tolist(),[True,True,False,False,False])
        self.assertTrue(score.small_feature_roi[3])
        # x-coordinate ramp: resized ROI [2,2,6,6] / stride2 -> [1,1,3,3].
        ramp=torch.arange(4).float().repeat(4,1)
        plain=ImageGeometry('ramp',(8,8),(8,8),(8,8))
        result=mean_region_error(ramp,torch.ones(4,4,dtype=torch.bool),[[2,2,6,6]],plain,2,output_size=2)
        self.assertAlmostEqual(float(result.scores[0]),1.5,places=6)
        none=mean_region_error(error,torch.zeros_like(valid),[[0,0,16,16]],geometry,2)
        self.assertFalse(none.valid.any())

    def test_mutually_exclusive_groups_include_difficult_unknown(self):
        boxes=torch.tensor([[0.,0,10,10],[20.,0,30,10],[40.,0,50,10],[60.,0,70,10],[0.,0,3,10]])
        gt={'boxes':boxes[:3],'known':torch.tensor([True,False,False]),'difficult':torch.tensor([False,False,True])}
        labels,_,_=label_regions(boxes,gt)
        self.assertEqual(labels,['known','unknown','ambiguous','background','ambiguous'])

    def test_metrics_ties_perfect_inverse_and_absent_class(self):
        self.assertEqual(binary_metrics([1,0],[1,0])['auroc'],1.)
        self.assertEqual(binary_metrics([1,0],[1,0])['auprc_average_precision'],1.)
        self.assertEqual(binary_metrics([1,0],[0,1])['auroc'],0.)
        tied=binary_metrics([1,0,0],[1,1,1])
        self.assertEqual(tied['auroc'],.5);self.assertAlmostEqual(tied['auprc_average_precision'],1/3)
        self.assertIsNone(binary_metrics([1],[.2])['auroc'])

    def test_offline_multi_scale_pipeline_and_plots(self):
        samples,geometry,_,extractor=synthetic('S2');batch,_=extractor(samples,geometry)
        batch=SpatialBatch([x[:1] for x in batch.levels],[x[:1] for x in batch.padding_masks],
                           batch.strides,geometry[:1],'S2')
        bank=FMRBank([8]*4,hidden=4,bottleneck=2).eval()
        error=ensemble_errors(bank,batch,.6,2,73)
        gt={'boxes':torch.tensor([[2.,2.,20.,24.],[50.,30.,80.,55.]]),
            'known':torch.tensor([True,False]),'difficult':torch.tensor([False,False])}
        queries=torch.cat([gt['boxes'],torch.tensor([[85.,0.,95.,15.]])])
        settings={'positive_iou':.5,'background_iou':.1,'background_proposals':16}
        pools=candidate_pools(gt,queries,geometry[0],73,settings)
        rows=region_records(batch,error,pools,{'output_size':3,'sampling_ratio':2,'min_box_size':1.,'min_valid_fraction':.5})
        accumulator=ScoreAccumulator()
        for row in rows:accumulator.add(row)
        summary=accumulator.summarize()
        self.assertIn('overall',summary['query'])
        self.assertEqual(summary['query']['overall']['counts'],{'known':1,'unknown':1,'background':1})
        with tempfile.TemporaryDirectory() as path:
            paths=distributions(rows,path,diagnostic=True)
            group_distributions(accumulator.groups,Path(path)/'streaming')
            self.assertTrue(paths and all(Path(x).stat().st_size>1000 for x in paths))

    def test_image_only_training_dataset_never_reads_annotations(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);(root/'ImageSets/TOWOD').mkdir(parents=True);(root/'JPEGImages').mkdir()
            (root/'ImageSets/TOWOD/owod_t1_train.txt').write_text('im\n')
            from PIL import Image
            Image.new('RGB',(16,12)).save(root/'JPEGImages/im.jpg')
            with patch('xml.etree.ElementTree.parse',side_effect=AssertionError('XML forbidden during training')):
                item=ImageOnlyDataset(root,short_side=8,max_size=16)[0]
            self.assertEqual(set(item),{'image','image_id','original_hw','resized_hw'})

    def test_sampler_resume_and_b0_guard(self):
        all_steps=list(StepBatches(11,2,0,10,73))
        self.assertEqual(all_steps[6:],list(StepBatches(11,2,6,10,73)))
        with tempfile.TemporaryDirectory() as path:
            state=Path(path)/'state.json';state.write_text(json.dumps({'state':'training','epoch_index':1}))
            with self.assertRaisesRegex(RuntimeError,'B0尚未完成'):
                guard_b0_complete({'b0_status':str(state),'baseline_checkpoint':'missing'})

    def test_s1_s2_configs_have_identical_budget_capacity_and_protocol(self):
        base=Path(__file__).resolve().parents[1]/'configs'
        first=json.loads((base/'R0-S1.json').read_text());second=json.loads((base/'R0-S2.json').read_text())
        for key in ['source','name','output']:first.pop(key);second.pop(key)
        self.assertEqual(first,second)

    def test_actual_configured_256_channel_capacity(self):
        features=[torch.randn(1,256,h,w) for h,w in [(8,12),(4,6),(2,3),(1,2)]]
        masks=[torch.zeros(1,*x.shape[-2:],dtype=torch.bool) for x in features]
        batch=SpatialBatch(features,masks,[8,16,32,64],[ImageGeometry('configured',(64,96),(64,96),(64,96))],'S1')
        bank=FMRBank([256]*4,hidden=64,bottleneck=32,depth=1)
        loss,rows=bank(batch,seed=73);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual([row['reconstruction'].shape for row in rows],[x.shape for x in features])

    def test_roi_odd_padding_uses_nominal_stride(self):
        geometry=ImageGeometry('odd',(8,8),(8,8),(9,9))
        ramp=torch.arange(5).float().repeat(5,1)
        result=mean_region_error(ramp,torch.ones(5,5,dtype=torch.bool),[[2,2,6,6]],geometry,2,output_size=2)
        self.assertAlmostEqual(float(result.scores[0]),1.5,places=6)

    def test_official_resize_rounding_is_preserved_without_annotations(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);(root/'ImageSets/TOWOD').mkdir(parents=True);(root/'JPEGImages').mkdir()
            (root/'ImageSets/TOWOD/owod_t1_train.txt').write_text('im\n')
            from PIL import Image
            Image.new('RGB',(17,31)).save(root/'JPEGImages/im.jpg')
            item=ImageOnlyDataset(root,short_side=8,max_size=12)[0]
            self.assertEqual(item['resized_hw'],(12,7))

    def test_all_invalid_regions_report_empty_metrics(self):
        accumulator=ScoreAccumulator()
        accumulator.add({'source':'query','level':'overall','valid':False,'group':'unknown','score':None})
        summary=accumulator.summarize()['query']['overall']
        self.assertEqual(summary['counts'],{'known':0,'unknown':0,'background':0})
        self.assertIsNone(summary['unknown_vs_background']['auroc'])

    def test_candidate_labels_match_clipped_roi_without_mutating_predictions(self):
        geometry=ImageGeometry('bounded',(16,16),(16,16),(16,16))
        gt={'boxes':torch.tensor([[0.,0.,16.,16.]]),'known':torch.tensor([True]),'difficult':torch.tensor([False])}
        boxes=torch.tensor([[-100.,-100.,16.,16.]])
        before=boxes.clone()
        pools=candidate_pools(gt,boxes,geometry,73,{'positive_iou':.5,'background_iou':.1,'background_proposals':4})
        self.assertEqual(pools['query']['labels'],['known'])
        self.assertTrue(torch.equal(pools['query']['boxes'],gt['boxes']))
        self.assertTrue(torch.equal(boxes,before))


if __name__=='__main__':unittest.main()
