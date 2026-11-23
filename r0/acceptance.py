"""Read-only acceptance of independently recovered B0 final artifacts."""
import json
from pathlib import Path

EXPECTED_SHA = '8262699520d30162069e46ed89b363313b20842dc89d6365b44443148460edb1'
BACKEND = dict(matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False,
               cudnn_deterministic=False, deterministic_algorithms=False)

def verify_artifacts(config):
    from r0.runtime import file_sha256
    def require(ok, reason):
        if not ok: raise RuntimeError('B0 acceptance: '+reason)
    checkpoint=Path(config['baseline_checkpoint'])
    require(file_sha256(checkpoint)==EXPECTED_SHA,'final weight SHA256 mismatch')
    epochs=[json.loads(line) for line in (checkpoint.parent/'epochs.jsonl').read_text().splitlines() if line.strip()]
    require([row['epoch'] for row in epochs]==list(range(41)) and all(row['complete'] for row in epochs), 'epoch records must be exactly 0..40')
    summary_path=Path(config['b0_analysis_summary']); summary=json.loads(summary_path.read_text())
    require(summary['checkpoint']['sha256']==EXPECTED_SHA,'analysis weight mismatch')
    root=Path(config['b0_status']).parent
    recovery_path=root/'query_analysis_recovery.json'
    if recovery_path.is_file():
        recovery=json.loads(recovery_path.read_text())
        require(recovery.get('passed') and recovery.get('all_11_metrics_exactly_equal'),'recovery did not pass')
        require(Path(recovery['summary']).resolve()==summary_path.resolve(),'recovery summary mismatch')
        metrics=recovery['metrics']
        require(len(metrics)==11 and set(metrics)==set(summary['official_metrics']),'11 metric keys mismatch')
        for key, values in metrics.items():
            require(values['difference']==0 and values['original_official']==values['official_replay']==values['query_analysis']==summary['official_metrics'][key], 'metric mismatch: '+key)
        parity=recovery['parity']
        for key in ('passed','preprocessing_bitwise_identical','raw_prediction_hashes_identical',
                    'final_detections_bitwise_identical','evaluator_serialized_inputs_identical','metrics_match_original_tolerance_1e-9'):
            require(parity.get(key) is True,'parity evidence missing: '+key)
        require(parity['input_rows']==parity['evaluator_image_occurrences']==10246 and parity['batches']==2050,'full replay scope mismatch')
        trace=Path(parity['trace']); manifest_path=trace/'manifest.json'
        require(file_sha256(manifest_path)==parity['trace_manifest_sha256'],'replay manifest digest mismatch')
        manifest=json.loads(manifest_path.read_text())
        require(manifest['complete'] and manifest['checkpoint']['sha256']==EXPECTED_SHA,'replay incomplete or wrong weight')
        require(file_sha256(trace/'index.jsonl')==manifest['index_sha256'],'replay index digest mismatch')
        require(file_sha256(manifest['split_file'])==manifest['split_sha256']==summary['split_provenance']['sha256'],'split digest mismatch')
        require(manifest['backend']==parity['backend']==summary['evaluation_backend']==BACKEND,'backend evidence mismatch')
        require(manifest['metrics']==summary['official_metrics'],'manifest metrics mismatch')
        return {'acceptance':'independent_recovery','recovery_path':str(recovery_path),
                'recovery_sha256':file_sha256(recovery_path),'weight_sha256':EXPECTED_SHA,'epochs':len(epochs),
                'trace_manifest_sha256':file_sha256(manifest_path),'backend':BACKEND}
    require('B0_PIPELINE_COMPLETE' in Path(config['b0_pipeline_log']).read_text(),'neither complete pipeline nor verified recovery')
    return {'acceptance':'pipeline_complete','weight_sha256':EXPECTED_SHA,'epochs':len(epochs)}

def backend_state():
    import torch
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())

def configure_backend():
    import torch
    before=backend_state()
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=False
    torch.use_deterministic_algorithms(False)
    assert backend_state()==BACKEND
    return before
