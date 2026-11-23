import copy
import tempfile
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from r1.cache import FrameWriter,FrameReader,validate_frame
from r1.features import preferred_layers,selection
from r1.tests.fixtures import synthetic


class CommonTests(unittest.TestCase):
    def test_exact_layers_features_boxes_and_pre_layernorm(self):
        model,extractor,samples,geometry,batch,outputs,frames,gt=synthetic()
        self.assertEqual(batch.raw.shape,(6,2,7,8))
        for i in range(6):
            layer=outputs['aux_outputs'][i] if i<5 else outputs
            self.assertTrue(torch.equal(batch.raw[i],layer['penu_layer']))
            self.assertTrue(torch.equal(batch.boxes[i],layer['pred_boxes']))
            expected=F.layer_norm(batch.pre_layernorm[i],(8,),model.transformer.decoder.layers[i].norm3.weight,
                                  model.transformer.decoder.layers[i].norm3.bias)
            self.assertTrue(torch.equal(batch.raw[i],expected))
        for image_index,frame in enumerate(frames):
            self.assertEqual(frame['layer_ids'],[1,3,6])
            for slot,index in enumerate([0,2,5]):
                np.testing.assert_array_equal(frame['raw'][slot],batch.raw[index,image_index].numpy())
                self.assertAlmostEqual(float(frame['boxes_cxcywh'][slot,0,1]),.12+index*.05,places=6)
            self.assertFalse(np.array_equal(frame['boxes_xyxy'][0],frame['boxes_xyxy'][-1]))
        self.assertEqual(preferred_layers(4),[1,3,4]);self.assertEqual(preferred_layers(2),[1,2])

    def test_frozen_outputs_and_buffers_unchanged(self):
        model,extractor,samples,geometry,batch,outputs,frames,gt=synthetic()
        snapshot={name:value.clone() for name,value in model.state_dict().items()}
        extractor.train(True);after,_=extractor(samples)
        self.assertFalse(model.training);self.assertTrue(torch.equal(batch.raw,after.raw))
        self.assertTrue(all(torch.equal(snapshot[name],value) for name,value in model.state_dict().items()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

    def test_actual_official_postprocess_mapping_on_cpu(self):
        model,extractor,samples,geometry,batch,outputs,frames,gt=synthetic()
        from models.prob_deformable_sketch_detr import PostProcess
        sizes=torch.tensor([item.original_hw for item in geometry])
        original=PostProcess(list(range(20,80)),temperature=1.3/8)(copy.deepcopy(outputs),sizes)
        for image,result in zip(frames,original):
            np.testing.assert_array_equal(result['boxes'].numpy(),image['final_boxes_xyxy'])
            np.testing.assert_array_equal(result['labels'].numpy(),image['final_pair_labels'])
            np.testing.assert_array_equal(result['scores'].numpy(),image['final_pair_scores'])

    def test_streamed_save_load_and_corrupt_query_id_rejected(self):
        *_,frames,gt=synthetic()
        with tempfile.TemporaryDirectory() as path:
            writer=FrameWriter(path,{'split':'evaluation','synthetic':True},max_gib=.01)
            for frame in frames:writer.write(frame)
            writer.close();loaded=list(FrameReader(path))
            self.assertEqual(len(loaded),2)
            for before,after in zip(frames,loaded):
                np.testing.assert_array_equal(before['raw'],after['raw'])
                np.testing.assert_array_equal(before['boxes_xyxy'],after['boxes_xyxy'])
            bad=copy.deepcopy(frames[0]);bad['selected_pair_query_ids'][0,0]=99
            with self.assertRaises(ValueError):validate_frame(bad)

    def test_missing_own_layer_outputs_raise(self):
        model,extractor,samples,*_=synthetic();model.aux=False
        with self.assertRaisesRegex(RuntimeError,'Missing own-layer'):extractor(samples)
        self.assertEqual(len(model.transformer.decoder._forward_hooks),0)


if __name__=='__main__':unittest.main()
