"""CPU regression tests for conservative offline grouping; no detector loads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from r0.labels import label_regions,grouping_audit,GroupingAudit
from r0.evaluate import candidate_pools,run
from r0.types import ImageGeometry
from r0.runtime import reconstruction_config_matches


class ProtocolRepairTests(unittest.TestCase):
    def test_low_iou_containment_all_gt_and_boundary_contact(self):
        gt={'boxes':torch.tensor([[0.,0,100,100],[110.,0,210,100],[220.,0,320,100]]),
            'known':torch.tensor([True,False,False]),'difficult':torch.tensor([False,False,True])}
        boxes=torch.tensor([[1.,1,3,3],[111.,1,113,3],[221.,1,223,3],
                            [319.,99,330,110],[320.,0,330,10],[0.,0,100,100],
                            [110.,0,210,100],[220.,0,320,100],[3.,3,3,4],[float('nan'),0,2,2]])
        old,iou,_=label_regions(boxes,gt,background_rule='iou_only')
        new,_,_=label_regions(boxes,gt)
        self.assertTrue((iou[:4]<.1).all())
        self.assertEqual(old[:5],['background']*5)
        self.assertEqual(new,['ambiguous']*4+['background','known','unknown','ambiguous','invalid','invalid'])
        audit=grouping_audit(boxes,gt)
        self.assertEqual(audit['delta']['background'],-4)
        self.assertEqual(audit['transitions']['background->ambiguous'],4)
        self.assertEqual(sum(audit['before_iou_only'].values()),len(boxes))
        self.assertEqual(sum(audit['delta'].values()),0)

    def test_empty_gt_empty_candidates_and_malformed_gt(self):
        gt={'boxes':torch.empty(0,4),'known':torch.empty(0,dtype=torch.bool),'difficult':torch.empty(0,dtype=torch.bool)}
        self.assertEqual(label_regions([[0,0,1,1],[0,0,0,1]],gt)[0],['background','invalid'])
        gt={'boxes':torch.tensor([[0.,0,10,10]]),'known':torch.tensor([True]),'difficult':torch.tensor([False])}
        self.assertEqual(label_regions(torch.empty(0,4),gt)[0],[])
        gt['boxes'][0,2]=0
        with self.assertRaisesRegex(ValueError,'Invalid GT'):label_regions([[20,0,30,10]],gt)
        with self.assertRaisesRegex(ValueError,'Unknown background'):label_regions([],gt,background_rule='typo')

    def test_gt_reference_and_query_share_background_definition_and_audit(self):
        gt={'boxes':torch.tensor([[0.,0,100,100]]),'known':torch.tensor([True]),'difficult':torch.tensor([False])}
        proposals=torch.tensor([[1.,1,3,3],[100.,0,110,10]])
        geometry=ImageGeometry('cpu',(120,120),(120,120),(120,120))
        settings={'positive_iou':.5,'background_iou':.1,'background_proposals':2}
        audit=GroupingAudit()
        with patch('r0.evaluate.background_candidates',return_value=proposals):
            pools=candidate_pools(gt,proposals,geometry,73,settings,audit=audit)
            legacy=candidate_pools(gt,proposals,geometry,73,dict(settings,background_rule='iou_only'))
        self.assertEqual(pools['query']['labels'],['ambiguous','background'])
        self.assertEqual(pools['gt_reference']['labels'],['known','background'])
        self.assertEqual(legacy['gt_reference']['labels'],['known','background','background'])
        self.assertEqual(audit.summarize()['gt_reference_proposals'],audit.summarize()['query'])

    def test_iou_control_reuses_same_weights_but_cannot_change_training_or_scores(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        main=json.loads((root/'R0-S1.json').read_text())
        control=json.loads((root/'iou_only/R0-S1.json').read_text())
        self.assertTrue(reconstruction_config_matches(main,control))
        for key in ('mask_ratio','batch_size','steps'):
            changed=dict(control);changed[key]=999
            self.assertFalse(reconstruction_config_matches(main,changed))
        control['evaluation']['repeats']=999
        self.assertFalse(reconstruction_config_matches(main,control))

    def test_final_detection_blocked_before_b0_or_detector_access(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/R0-S1.json').read_text())
        config['evaluation']['include_final_detections']=True
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json';path.write_text(json.dumps(config))
            with patch('r0.evaluate.guard_b0_complete',side_effect=AssertionError('must fail first')):
                with self.assertRaisesRegex(RuntimeError,'query-ID deduplication'):run(path,'missing')


if __name__=='__main__':unittest.main()
