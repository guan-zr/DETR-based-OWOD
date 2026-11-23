"""One compressed, pickle-free frame per image, never all feature tensors in RAM."""
import json
from pathlib import Path
import numpy as np
from r1.reuse import write_json


def validate_frame(frame):
    layers=np.asarray(frame['layer_ids']).tolist();raw=np.asarray(frame['raw'])
    if raw.ndim!=3 or len(layers)!=raw.shape[0] or len(set(layers))!=len(layers):raise ValueError('Invalid layer-feature schema')
    if any(not 1<=layer<=int(frame['decoder_layer_count']) for layer in layers):raise ValueError('Layer ID outside decoder')
    for key in ['l2','pre_layernorm']:
        if np.asarray(frame[key]).shape!=raw.shape:raise ValueError('Feature representation shape mismatch')
    for key in ['boxes_xyxy','boxes_cxcywh']:
        if np.asarray(frame[key]).shape!=raw.shape[:2]+(4,):raise ValueError('Own-layer box shape mismatch')
    for key in ['gaussian_distance','gaussian_objectness','classifier_unknown','selected_unique']:
        if np.asarray(frame[key]).shape!=raw.shape[:2]:raise ValueError('Query score/index shape mismatch')
    if np.asarray(frame['logits']).shape[:2]!=raw.shape[:2]:raise ValueError('Class index mismatch')
    for slot,layer in enumerate(layers):
        ids=np.asarray(frame['selected_pair_query_ids'])[slot]
        if ids.dtype.kind not in 'iu' or (ids<0).any() or (ids>=raw.shape[1]).any():raise ValueError('Invalid selected query ID')
        expected=np.zeros(raw.shape[1],dtype=bool);expected[ids]=True
        if not np.array_equal(expected,np.asarray(frame['selected_unique'])[slot]):raise ValueError('Selected mask/query ID mismatch')
        if layer==int(frame['decoder_layer_count']):
            final_ids=np.asarray(frame['final_pair_query_ids'])
            if not np.array_equal(final_ids,ids):raise ValueError('Final selection ID mismatch')
            if not np.array_equal(np.asarray(frame['final_boxes_xyxy']),np.asarray(frame['boxes_xyxy'])[slot,final_ids],equal_nan=True):
                raise ValueError('Final query/box index mismatch')


class FrameWriter:
    def __init__(self,root,metadata,max_gib=12):
        self.root=Path(root)
        if self.root.exists() and any(self.root.iterdir()):raise RuntimeError('Refuse to overwrite feature cache')
        self.root.mkdir(parents=True,exist_ok=True)
        self.metadata=dict(metadata,schema='r1-query-v1',complete=False,images=0)
        self.limit=max_gib*1024**3;self.bytes=0;self.ids=set()
        write_json(self.root/'manifest.json',self.metadata)

    def write(self,frame):
        validate_frame(frame);stem=str(frame['image_id'])
        if stem in self.ids or '/' in stem:raise ValueError('Repeated/unsafe image ID')
        self.ids.add(stem)
        path=self.root/f'{stem}.npz';np.savez_compressed(path,**frame)
        self.bytes+=path.stat().st_size
        with (self.root/'index.jsonl').open('a') as handle:
            handle.write(json.dumps({'image_id':stem,'file':path.name})+'\n')
        if self.bytes>self.limit:raise RuntimeError('Cache disk budget exceeded; incomplete manifest retained')

    def close(self):
        self.metadata.update(complete=True,images=len(self.ids),bytes=self.bytes)
        write_json(self.root/'manifest.json',self.metadata)


class FrameReader:
    def __init__(self,root):
        self.root=Path(root);self.metadata=json.loads((self.root/'manifest.json').read_text())
        if self.metadata.get('schema')!='r1-query-v1' or not self.metadata.get('complete'):
            raise RuntimeError('Incomplete or incompatible query cache')

    def __iter__(self):
        count=0
        with (self.root/'index.jsonl').open() as handle:
            for line in handle:
                index=json.loads(line)
                with np.load(self.root/index['file'],allow_pickle=False) as saved:
                    frame={key:saved[key] for key in saved.files}
                frame['image_id']=str(frame['image_id'].item())
                validate_frame(frame)
                if frame['image_id']!=index['image_id']:raise RuntimeError('Image/cache index mismatch')
                count+=1
                yield frame
        if count!=self.metadata['images']:raise RuntimeError('Cache image count differs from manifest')
