"""Scientific invariants and failure gates for controls, CPU only."""
import tempfile,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from r0.types import SpatialBatch,ImageGeometry
from r0.round2.scoring import simple_maps,roi_maps
from r0.round2.prepare import match_geometry
from r0.round2.bootstrap import clustered_samples
from r0.metrics import binary_metrics

class Round2Tests(unittest.TestCase):
    def test_channel_statistics_and_zero_prediction_are_exact(self):
        feature=torch.tensor([[[[1.,3.],[2.,4.]],[[3.,1.],[4.,2.]]]])
        batch=SpatialBatch([feature]*4,[torch.zeros(1,2,2,dtype=torch.bool)]*4,[1]*4,[ImageGeometry('t',(2,2),(2,2),(2,2))],'S1')
        maps=simple_maps(batch,torch.zeros(4,2))
        self.assertTrue(torch.equal(maps['energy'][0],maps['mean_error'][0]));self.assertTrue(torch.equal(maps['channel_variance'][0],torch.ones(1,2,2)))
        self.assertTrue(torch.equal(maps['l2'][0],feature.square().sum(1).sqrt()))
        score,v,_=roi_maps(batch,maps['energy'],torch.tensor([[0.,0,2,2]]));self.assertTrue(v.all());self.assertAlmostEqual(score[0,4],float(maps['energy'][0].mean()),places=6)

    def test_pointwise_norm_is_not_norm_after_pooling(self):
        f=torch.tensor([[1.,0.],[0.,1.]])
        self.assertNotEqual(float(f.square().sum(1).sqrt().mean()),float(f.mean(0).square().sum().sqrt()))

    def test_matching_is_score_independent_caliper_and_no_replacement(self):
        frame=pd.DataFrame(dict(image_id=['a','b','c','d','e'],source=['gt_reference']*5,candidate_id=range(5),x1=[0.]*5,y1=[0.]*5,x2=[10.,10.,100.,10.,10.5],y2=[10.,10.,100.,10.,10.5],original_h=[100]*5,original_w=[100]*5,score=[0.,9.,1.,3.,4.]))
        pairs,missing=match_geometry(frame,[0,1,2],[3,4]);self.assertEqual(len(pairs),2);self.assertEqual(len(set(pairs[:,1])),2);self.assertEqual(missing.tolist(),[2])
        frame['score']=-frame['score'];same,miss=match_geometry(frame,[0,1,2],[3,4]);np.testing.assert_array_equal(pairs,same);np.testing.assert_array_equal(missing,miss)

    def test_tie_aware_bootstrap_and_paired_transform(self):
        y=np.array([0,1,1],bool);s=np.array([0.,1.,1.]);im=np.zeros(3,dtype=int);w=np.ones((20,1),dtype=np.uint16)
        a=clustered_samples(y,s,im,w);b=clustered_samples(y,-s,im,w)
        self.assertTrue((a[:,0]==1).all());self.assertTrue((b[:,0]==0).all());self.assertTrue(((a-b)[:,0]==1).all())
        self.assertEqual(binary_metrics(y,s)['auprc_average_precision'],a[0,1])

    def test_partial_evaluation_is_rejected(self):
        from r0.round2 import analyze
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'evaluation').mkdir();(root/'prepared').mkdir();(root/'evaluation/complete.json').write_text(json.dumps(dict(scope='limited_smoke',unique_images=4,complete=True)));(root/'prepared/complete.json').write_text('{}')
            with patch.object(analyze,'ROOT',root):
                with self.assertRaisesRegex(RuntimeError,'9119'):analyze.main()
if __name__=='__main__':unittest.main(verbosity=2)
