import unittest
import numpy as np
from b0.metrics import iou, official_unknown_hits, query_groups


class MetricsTest(unittest.TestCase):
    def test_geometry_and_thresholds(self):
        boxes=[[0,0,10,10],[0,0,5,10],[10,0,20,10]]
        overlap=iou(boxes,[[0,0,10,10]])[:,0]
        np.testing.assert_allclose(overlap,[1,0.5,0])
        self.assertEqual(int((overlap>=0.5).sum()),2)
        self.assertEqual(iou([],[[0,0,10,10]]).shape,(0,1))

    def test_exclusive_groups_include_unknown_and_difficult(self):
        gt=[[0,0,10,10],[20,0,30,10],[40,0,50,10]]
        queries=gt+[[60,0,70,10],[0,0,3,10]]
        actual=query_groups(queries,gt,[True,False,False],[False,False,True])
        self.assertEqual(actual.tolist(),['known','unknown','ambiguous','background','ambiguous'])

    def test_official_matching_duplicates_and_difficult(self):
        gt={'im':{'known':np.array([False,False]),'difficult':np.array([False,True]),
                  'xml_boxes':np.array([[1,1,10,10],[21,1,30,10]])}}
        lines=['im 0.900 1.0 1.0 10.0 10.0','im 0.800 1.0 1.0 10.0 10.0',
               'im 0.700 21.0 1.0 30.0 10.0']
        self.assertEqual(official_unknown_hits(lines,gt),{('im',0)})


if __name__=='__main__':
    unittest.main()
