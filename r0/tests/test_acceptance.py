"""Reject corrupted or incomplete independent acceptance evidence before CUDA."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from r0.acceptance import verify_artifacts,EXPECTED_SHA,BACKEND

class AcceptanceTests(unittest.TestCase):
    def fixture(self,root):
        def put(name,data):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data));return p
        ck=root/'checkpoint0040.pth';ck.write_text('fixture')
        (root/'epochs.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,complete=True)) for e in range(41)))
        metrics={str(i):float(i) for i in range(11)}
        summary=put('summary.json',dict(checkpoint={'sha256':EXPECTED_SHA},official_metrics=metrics,split_provenance={'sha256':EXPECTED_SHA},evaluation_backend=BACKEND))
        trace=root/'trace';trace.mkdir();(trace/'index.jsonl').write_text('fixture')
        put('trace/manifest.json',dict(complete=True,checkpoint={'sha256':EXPECTED_SHA},index_sha256=EXPECTED_SHA,split_file=str(ck),split_sha256=EXPECTED_SHA,backend=BACKEND,metrics=metrics))
        parity=dict(passed=True,preprocessing_bitwise_identical=True,raw_prediction_hashes_identical=True,final_detections_bitwise_identical=True,evaluator_serialized_inputs_identical=True,**{'metrics_match_original_tolerance_1e-9':True},input_rows=10246,evaluator_image_occurrences=10246,batches=2050,trace=str(trace),trace_manifest_sha256=EXPECTED_SHA,backend=BACKEND)
        rec=dict(passed=True,all_11_metrics_exactly_equal=True,summary=str(summary),parity=parity,metrics={k:dict(difference=0,original_official=v,official_replay=v,query_analysis=v) for k,v in metrics.items()})
        put('query_analysis_recovery.json',rec)
        return dict(baseline_checkpoint=str(ck),b0_status=str(root/'state.json'),b0_analysis_summary=str(summary)),rec

    def test_recovery_success_without_pipeline_marker(self):
        with tempfile.TemporaryDirectory() as d,patch('r0.runtime.file_sha256',return_value=EXPECTED_SHA):
            cfg,_=self.fixture(Path(d));self.assertEqual(verify_artifacts(cfg)['acceptance'],'independent_recovery')

    def test_sha_incomplete_epochs_raw_parity_metrics_backend_rejected(self):
        for issue in ['sha','epochs','raw','metric','backend']:
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as d:
                root=Path(d);cfg,rec=self.fixture(root)
                if issue=='epochs':(root/'epochs.jsonl').write_text('{"epoch":40,"complete":true}')
                if issue=='raw':rec['parity']['raw_prediction_hashes_identical']=False
                if issue=='metric':rec['metrics']['0']['official_replay']=99
                if issue=='backend':rec['parity']['backend']=dict(BACKEND,cudnn_tf32=False)
                (root/'query_analysis_recovery.json').write_text(json.dumps(rec))
                with patch('r0.runtime.file_sha256',return_value='bad' if issue=='sha' else EXPECTED_SHA):
                    with self.assertRaises(RuntimeError):verify_artifacts(cfg)
