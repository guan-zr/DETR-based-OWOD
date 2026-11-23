"""Recovery and clustered weighting checks; synthetic data never enter formal outputs."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from r1.tests.fixtures import synthetic
from r1.cache import FrameWriter,FrameReader
from r1.formal_stats import clustered_samples
from r1.reuse import binary_metrics

class FormalExecutionTests(unittest.TestCase):
    def test_interrupted_cache_resumes_only_committed_matching_frames(self):
        *_,frames,gt=synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cache';meta={'image_ids':['one','two'],'split':'evaluation'}
            writer=FrameWriter(root,meta);writer.write(frames[0])
            resumed=FrameWriter(root,meta,resume=True)
            self.assertEqual(resumed.ids,{'one'})
            resumed.write(frames[1]);resumed.close()
            self.assertEqual(len(list(FrameReader(root))),2)
            with self.assertRaisesRegex(RuntimeError,'Already complete'):FrameWriter(root,meta,resume=True)

    def test_resume_rejects_changed_protocol_and_corrupt_commit(self):
        *_,frames,gt=synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cache';meta={'image_ids':['one','two'],'split':'evaluation'}
            writer=FrameWriter(root,meta);writer.write(frames[0])
            with self.assertRaisesRegex(RuntimeError,'provenance'):FrameWriter(root,dict(meta,split='train'),resume=True)
            (root/'one.npz').write_bytes(b'broken commit')
            with self.assertRaisesRegex(RuntimeError,'Invalid resumed'):FrameWriter(root,meta,resume=True)

    def test_tied_cluster_bootstrap_matches_literal_image_replication(self):
        labels=np.array([1,0,1,0],bool);scores=np.array([.5,.5,.8,.2]);images=np.array([0,0,1,1]);weights=np.array([[1,1],[2,0],[0,2],[1,3]],np.uint16)
        actual=clustered_samples(labels,scores,images,weights)
        for i,row in enumerate(weights):
            indices=np.repeat(np.arange(4),row[images]);expected=binary_metrics(labels[indices].astype(int),scores[indices]);np.testing.assert_allclose(actual[i],[expected['auroc'],expected['auprc_average_precision']],rtol=0,atol=1e-14)

if __name__=='__main__':unittest.main()
