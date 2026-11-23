"""CPU rejection paths: actual split bytes/IDs and extraction config must match."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from r1.cache import FrameWriter,FrameReader,cache_metadata
from r1.runtime import config_read
from r1.diagnose import run as diagnose
from r1.evaluate import run as evaluate
from r1.analysis import evaluate_frames
from r1.tests.fixtures import synthetic


class ProtocolRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        *_,self.frames,self.gt=synthetic()
        self.config=json.loads((Path(__file__).resolve().parents[1]/'configs/diagnostic.json').read_text())
        self.config.update(data_root=str(self.root/'data'),synthetic=True)
        split=Path(self.config['data_root'])/'ImageSets/TOWOD';split.mkdir(parents=True)
        self.split=split/(self.config['eval_split']+'.txt');self.split.write_text('one\ntwo\none\n')
        self.path=self.root/'config.json';self.path.write_text(json.dumps(self.config))
        self.cache=self.root/'cache'
        self.metadata=cache_metadata(self.config,'evaluation',{'sha256':'synthetic-not-B0'})
        writer=FrameWriter(self.cache,self.metadata)
        for frame in self.frames:writer.write(frame)
        writer.close()

    def mutate_manifest(self,change):
        path=self.cache/'manifest.json';manifest=json.loads(path.read_text());change(manifest)
        path.write_text(json.dumps(manifest))

    def test_valid_cache_unique_split_ids_and_raw_l2_reuse(self):
        self.assertEqual(len(list(FrameReader(self.cache,self.config))),2)
        raw=copy.deepcopy(self.config);raw['representation']='raw';raw['mask_ratio']=.3;raw['layer']=3
        self.assertEqual(len(list(FrameReader(self.cache,raw))),2)
        with patch('r1.diagnose.read_eval_gt',side_effect=lambda root,stem:self.gt[stem]):
            result=diagnose(self.path,self.cache,self.root/'diagnosis')
        self.assertEqual(result['cache_provenance']['image_ids'],['one','two'])
        self.assertEqual(result['background_rule'],'disjoint_gt')

    def test_wrong_split_hash_and_same_set_changed_bytes_rejected(self):
        self.split.write_text('two\none\n')
        with self.assertRaisesRegex(RuntimeError,'split_sha256'):FrameReader(self.cache,self.config)

    def test_wrong_split_name_and_synthetic_scope_rejected(self):
        self.mutate_manifest(lambda m:m.update(split_file='wrong'))
        with self.assertRaisesRegex(RuntimeError,'split_file'):FrameReader(self.cache,self.config)
        self.mutate_manifest(lambda m:m.update(split_file=self.config['eval_split'],synthetic=False))
        with self.assertRaisesRegex(RuntimeError,'synthetic'):FrameReader(self.cache,self.config)

    def test_wrong_batch_resize_and_source_hash_rejected(self):
        for key,value in [('batch_size',2),('resize',{'short_side':400,'max_size':1333}),('source_sha256',{})]:
            original=(self.cache/'manifest.json').read_text()
            self.mutate_manifest(lambda m:m['extraction_config'].update({key:value}))
            with self.assertRaisesRegex(RuntimeError,'extraction_config'):FrameReader(self.cache,self.config)
            (self.cache/'manifest.json').write_text(original)

    def test_missing_legacy_provenance_rejected(self):
        for key in ['split_sha256','image_ids','extraction_config','decoder_layer_count']:
            original=(self.cache/'manifest.json').read_text()
            self.mutate_manifest(lambda m:m.pop(key))
            with self.assertRaises(RuntimeError):FrameReader(self.cache,self.config)
            (self.cache/'manifest.json').write_text(original)

    def test_wrong_actual_id_set_even_with_consistent_count_rejected(self):
        self.split.write_text('one\nother\n')
        from r1.reuse import file_sha256
        self.mutate_manifest(lambda m:m.update(split_sha256=file_sha256(self.split)))
        with self.assertRaisesRegex(RuntimeError,'image_ids'):FrameReader(self.cache,self.config)

    def test_duplicate_index_missing_file_and_manifest_id_set_rejected(self):
        index=self.cache/'index.jsonl';original=index.read_text()
        index.write_text(original+original.splitlines()[0]+'\n')
        with self.assertRaisesRegex(RuntimeError,'Repeated'):FrameReader(self.cache,self.config)
        index.write_text(original)
        self.mutate_manifest(lambda m:m.update(image_ids=['one','wrong']))
        with self.assertRaisesRegex(RuntimeError,'ID set'):FrameReader(self.cache,self.config)
        self.mutate_manifest(lambda m:m.update(image_ids=['one','two']))
        (self.cache/'one.npz').unlink()
        with self.assertRaisesRegex(RuntimeError,'Missing indexed'):FrameReader(self.cache,self.config)

    def test_frame_image_id_layer_and_real_padding_mismatch_rejected(self):
        target=self.cache/'one.npz'
        original=copy.deepcopy(self.frames[0]);wrong=copy.deepcopy(original);wrong['image_id']='wrong'
        np.savez_compressed(target,**wrong)
        with self.assertRaisesRegex(RuntimeError,'Image/cache index'):list(FrameReader(self.cache,self.config))
        np.savez_compressed(target,**original)
        self.mutate_manifest(lambda m:m.update(layer_ids=[1,2,6]))
        with self.assertRaisesRegex(RuntimeError,'layer_ids'):FrameReader(self.cache,self.config)
        self.mutate_manifest(lambda m:m.update(layer_ids=[1,3,6]))
        reader=FrameReader(self.cache,self.config);reader.metadata['synthetic']=False
        with self.assertRaisesRegex(RuntimeError,'padding'):list(reader)

    def test_incomplete_writer_and_entrypoints_fail_before_evaluation_or_checkpoint(self):
        writer=FrameWriter(self.root/'partial',self.metadata);writer.write(self.frames[0])
        with self.assertRaisesRegex(RuntimeError,'ID set'):writer.close()
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):FrameReader(self.root/'partial',self.config)
        self.mutate_manifest(lambda m:m.update(split_sha256='bad'))
        with patch('r1.diagnose.evaluate_frames',side_effect=AssertionError('must reject first')):
            with self.assertRaisesRegex(RuntimeError,'split_sha256'):diagnose(self.path,self.cache,self.root/'out')
        with patch('r1.evaluate.torch.load',side_effect=AssertionError('must reject first')):
            with self.assertRaisesRegex(RuntimeError,'split_sha256'):evaluate(self.path,self.cache,'absent',self.root/'out')
        self.assertFalse((self.root/'out').exists())

    def test_formal_baseline_checksum_mismatch_rejected(self):
        config=copy.deepcopy(self.config);config['synthetic']=False
        checkpoint=self.root/'fake_baseline';checkpoint.write_bytes(b'CPU provenance fixture, not a model')
        config['baseline_checkpoint']=str(checkpoint)
        self.mutate_manifest(lambda m:m.update(synthetic=False,baseline={'path':str(checkpoint),'sha256':'wrong'}))
        with self.assertRaisesRegex(RuntimeError,'baseline checkpoint'):FrameReader(self.cache,config)

    def test_all_current_configs_batch_one_and_raw_pairs(self):
        base=Path(__file__).resolve().parents[1]/'configs'
        for path in base.glob('*.json'):
            config=config_read(path)
            self.assertEqual((config['batch_size'],config['extract_batch_size']),(1,1))
            if path.stem.startswith('L') and not path.stem.endswith('_raw'):
                raw=json.loads(path.with_name(path.stem+'_raw.json').read_text())
                for key in ('representation','name','output'):raw.pop(key);config.pop(key)
                self.assertEqual(raw,config)
        old=json.loads((base/'legacy_original/diagnostic.json').read_text())
        old_path=self.root/'old.json';old_path.write_text(json.dumps(old))
        with self.assertRaisesRegex(ValueError,'batch=1'):config_read(old_path)

    def test_low_iou_containment_is_ambiguous_at_every_saved_layer(self):
        frames=copy.deepcopy(self.frames)
        for frame in frames:
            for slot in range(3):frame['boxes_xyxy'][slot,0]=[1,1,2,2]
        def gt_loader(stem):
            return {'boxes':torch.tensor([[0.,0,120,120]]),'known':torch.tensor([False]),'difficult':torch.tensor([False])}
        result=evaluate_frames(frames,gt_loader,self.config['evaluation'],self.root/'groups',synthetic=True)
        for layer in (1,3,6):
            audit=result['grouping_audit'][f'L{layer}_all']
            self.assertGreaterEqual(audit['transitions']['background->ambiguous'],2)
            self.assertEqual(audit['after_disjoint_gt']['background'],0)


if __name__=='__main__':unittest.main()
