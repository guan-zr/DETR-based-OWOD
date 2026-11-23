"""Read-only reuse of the already verified R0 generic tools."""
import hashlib
import json
from pathlib import Path
import sys

manifest=json.loads((Path(__file__).parent/'reuse_manifest.json').read_text())
for name,expected in manifest['files'].items():
    if hashlib.sha256((Path(manifest['root'])/name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('R0 dependency changed; review before use: '+name)
sys.path.append(manifest['root'])
from r0.data import ImageOnlyDataset,image_collate,read_eval_gt
from r0.labels import label_regions
from r0.metrics import binary_metrics
from r0.runtime import (guard_b0_complete,load_detector,pack_images,StepBatches,
                        seed_all,write_json,file_sha256,reconstruction_config_matches)
