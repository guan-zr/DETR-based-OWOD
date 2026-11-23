from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from r1.analysis import evaluate_frames
from r1.plot import run as plot
from r1.reuse import label_regions,binary_metrics
from r1.tests.fixtures import synthetic


class DiagnosticTests(unittest.TestCase):
    def test_grouping_excludes_unknown_and_difficult_from_background(self):
        boxes=torch.tensor([[0.,0,10,10],[20.,0,30,10],[40.,0,50,10],[60.,0,70,10],[0.,0,3,10]])
        gt={'boxes':boxes[:3],'known':torch.tensor([True,False,False]),'difficult':torch.tensor([False,False,True])}
        groups,_,_=label_regions(boxes,gt)
        self.assertEqual(groups,['known','unknown','ambiguous','background','ambiguous'])

    def test_metrics_diagnostic_and_case_export_and_plot(self):
        *_,frames,gt=synthetic()
        self.assertEqual(binary_metrics([1,0],[1,0])['auroc'],1.)
        self.assertEqual(binary_metrics([1,0],[1,0])['auprc_average_precision'],1.)
        with tempfile.TemporaryDirectory() as path:
            out=Path(path)/'diagnosis'
            result=evaluate_frames(frames,gt.__getitem__,{'positive_iou':.5,'background_iou':.1,'low_objectness_threshold':.5},out,synthetic=True)
            self.assertEqual(set(result['statistics']),{'1','3','6'})
            for layer in ['1','3','6']:
                counts=result['statistics'][layer]['all']['counts']
                self.assertEqual(counts['known'],2);self.assertEqual(counts['unknown'],2)
                self.assertEqual(result['unknown_cases'][layer]['unknown_gt'],6)
                self.assertEqual(result['unknown_cases'][layer]['coverage_05'],2)
            plot([out],Path(path)/'figures')
            self.assertTrue((Path(path)/'figures/layer_separability.png').is_file())


if __name__=='__main__':unittest.main()
