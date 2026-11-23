"""Exercise real process failure/lock behavior using disposable CPU-only outputs."""
import fcntl,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from r0 import pipeline
from r0.formal_evaluate import run as evaluate
from r0.formal_analysis import analyze

class PipelineTests(unittest.TestCase):
    def test_stage_failure_stops_control_flow_and_preserves_failed_state(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir();(root/'stages').mkdir();lock=(root/'lock').open('w')
            def missing():raise FileNotFoundError('not committed')
            original_sleep=time.sleep
            with patch.object(pipeline,'CONTROL',root),patch.object(pipeline.time,'sleep',side_effect=lambda _:original_sleep(.01)):
                with self.assertRaisesRegex(RuntimeError,'exit 7'):
                    pipeline.stage_run('bad',[sys.executable,'-c','raise SystemExit(7)'],missing,root,lock.fileno(),{})
                state=json.loads((root/'stages/bad.json').read_text());self.assertEqual(state['state'],'failed');self.assertEqual(state['returncode'],7)
                self.assertFalse((root/'complete.json').exists())

    def test_committed_artifact_recovered_without_running_command(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'stages').mkdir()
            with patch.object(pipeline,'CONTROL',root),patch('r0.pipeline.subprocess.Popen',side_effect=AssertionError('must not run')):
                state=pipeline.stage_run('done',['never'],lambda:{'valid':True},root,0,{})
                self.assertEqual(state['state'],'completed');self.assertTrue(state['recovered_committed_artifact'])

    def test_duplicate_launch_does_not_start_a_child_or_change_status(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            (root/'status.json').write_text('{"state":"running"}')
            with patch.object(pipeline,'CONTROL',root),patch('r0.pipeline.subprocess.Popen',side_effect=AssertionError('duplicate child')):
                pipeline.launch()
            self.assertEqual((root/'status.json').read_text(),'{"state":"running"}')

    def test_fresh_probe_cannot_run_as_full_evaluation(self):
        with self.assertRaisesRegex(RuntimeError,'cannot be full'):evaluate(diagnostic_fresh=True)

    def test_cluster_bootstrap_retains_ties_and_paired_effect(self):
        import pandas as pd
        from r0.formal_analysis import bootstrap_pair,metric_frame
        frame=pd.DataFrame(dict(image_id=['one']*3,group=['background','unknown','unknown'],score_S1=[0.,1.,1.],score_S2=[1.,0.,0.]))
        result=bootstrap_pair(frame,['one'],repeats=10)['estimates']
        self.assertEqual(result['S1']['auroc']['ci95'],[1.,1.]);self.assertEqual(result['S2']['auroc']['ci95'],[0.,0.])
        self.assertEqual(result['S2_minus_S1']['auroc']['ci95'],[-1.,-1.])
        f=pd.DataFrame(dict(group=['known','unknown','background'],score=[1.,2.,0.]))
        self.assertEqual(metric_frame(f,positive='foreground')['auroc'],1.)

    def test_partial_evaluation_cannot_generate_complete_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'complete.json').write_text(json.dumps(dict(complete=True,scope='limited_smoke',unique_images=4)))
            with self.assertRaisesRegex(RuntimeError,'complete 9119'):analyze(root,root/'analysis')

if __name__=='__main__':unittest.main()
