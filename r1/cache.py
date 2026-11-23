"""One compressed, pickle-free frame per image, never all feature tensors in RAM."""
import json
from pathlib import Path
import numpy as np
from r1.reuse import write_json,file_sha256
from r1.features import preferred_layers


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
    def __init__(self,root,metadata,max_gib=12,resume=False):
        self.root=Path(root)
        if self.root.exists() and any(self.root.iterdir()) and not resume:raise RuntimeError('Refuse to overwrite feature cache')
        self.root.mkdir(parents=True,exist_ok=True)
        self.metadata=dict(metadata,schema='r1-query-v1',complete=False,images=0)
        self.limit=max_gib*1024**3;self.bytes=0;self.ids=set()
        if resume and (self.root/'manifest.json').exists():
            old=json.loads((self.root/'manifest.json').read_text())
            if old.get('complete'):raise RuntimeError('Already complete; do not repeat extraction')
            for key,value in self.metadata.items():
                if key not in ('images','complete') and old.get(key)!=value:raise RuntimeError('Resume cache provenance mismatch: '+key)
            index=self.root/'index.jsonl'
            if index.exists():
                for line in index.read_text().splitlines():
                    item=json.loads(line);path=self.root/item['file']
                    if item['image_id'] in self.ids or file_sha256(path)!=item.get('sha256'):raise RuntimeError('Invalid resumed cache commit')
                    with np.load(path,allow_pickle=False) as saved:validate_frame({key:saved[key] for key in saved.files})
                    self.ids.add(item['image_id']);self.bytes+=path.stat().st_size
        write_json(self.root/'manifest.json',self.metadata)

    def write(self,frame):
        validate_frame(frame);stem=str(frame['image_id'])
        if stem in self.ids or not stem or stem in ('.','..') or '/' in stem or '\\' in stem:raise ValueError('Repeated/unsafe image ID')
        self.ids.add(stem)
        path=self.root/f'{stem}.npz';temporary=self.root/f'{stem}.tmp.npz';np.savez_compressed(temporary,**frame);temporary.replace(path)
        self.bytes+=path.stat().st_size
        with (self.root/'index.jsonl').open('a') as handle:
            handle.write(json.dumps({'image_id':stem,'file':path.name,'sha256':file_sha256(path)})+'\n')
        if self.bytes>self.limit:raise RuntimeError('Cache disk budget exceeded; incomplete manifest retained')

    def close(self):
        if 'image_ids' in self.metadata and sorted(self.ids)!=self.metadata['image_ids']:
            raise RuntimeError('Extracted image ID set differs from requested split; cache remains incomplete')
        self.metadata.update(complete=True,images=len(self.ids),image_ids=sorted(self.ids),bytes=self.bytes)
        write_json(self.root/'manifest.json',self.metadata)


class FrameReader:
    def __init__(self,root,config=None,split='evaluation'):
        self.root=Path(root);self.metadata=json.loads((self.root/'manifest.json').read_text())
        if self.metadata.get('schema')!='r1-query-v1' or not self.metadata.get('complete'):
            raise RuntimeError('Incomplete or incompatible query cache')
        self.index=[json.loads(line) for line in (self.root/'index.jsonl').read_text().splitlines()]
        ids=[item['image_id'] for item in self.index]
        files=[item['file'] for item in self.index]
        if len(ids)!=len(set(ids)) or len(files)!=len(set(files)):
            raise RuntimeError('Repeated image ID/file in cache index')
        if len(ids)!=self.metadata['images'] or sorted(ids)!=self.metadata.get('image_ids'):
            raise RuntimeError('Cache image ID set/count differs from manifest')
        for item in self.index:
            stem=item['image_id']
            if not isinstance(stem,str) or not stem or stem in ('.','..') or '/' in stem or '\\' in stem or item['file']!=stem+'.npz':
                raise RuntimeError('Unsafe image/cache index path')
            if not (self.root/item['file']).is_file():raise RuntimeError('Missing indexed cache frame: '+stem)
        if config is not None:self.validate_provenance(config,split)

    def validate_provenance(self,config,split='evaluation'):
        expected=cache_metadata(config,split,self.metadata.get('baseline',{}),
                                self.metadata.get('all_layers',False),
                                self.metadata.get('decoder_layer_count',0))
        for key in ('split','split_file','split_sha256','image_ids','extraction_config','synthetic','layer_ids','decoder_layer_count'):
            if self.metadata.get(key)!=expected[key]:
                raise RuntimeError('Cache provenance mismatch: '+key+'; re-extract with the requested config')
        baseline=self.metadata.get('baseline',{})
        if not config.get('synthetic',False):
            checkpoint=Path(config['baseline_checkpoint'])
            if not checkpoint.is_file() or baseline.get('path')!=str(checkpoint) or baseline.get('sha256')!=file_sha256(checkpoint):
                raise RuntimeError('Cache provenance mismatch: baseline checkpoint path/SHA256')
        self.validated_config=config
        return self

    def __iter__(self):
        for index in self.index:
            if 'sha256' in index and file_sha256(self.root/index['file'])!=index['sha256']:raise RuntimeError('Image/cache index or frame digest mismatch')
            with np.load(self.root/index['file'],allow_pickle=False) as saved:
                frame={key:saved[key] for key in saved.files}
            frame['image_id']=str(frame['image_id'].item())
            validate_frame(frame)
            if frame['image_id']!=index['image_id']:raise RuntimeError('Image/cache index mismatch')
            if hasattr(self,'validated_config'):
                if np.asarray(frame['layer_ids']).tolist()!=self.metadata['layer_ids'] or int(frame['decoder_layer_count'])!=self.metadata['decoder_layer_count']:
                    raise RuntimeError('Frame extraction layer configuration differs from manifest')
                # Synthetic fixtures may model padded multi-image batches. Real
                # batch=1 extraction must have exactly the resized image canvas.
                if not self.metadata['synthetic'] and not np.array_equal(frame['resized_hw'],frame['padded_hw']):
                    raise RuntimeError('Frame padding differs from batch=1 extraction protocol')
            yield frame


def extraction_config(config,all_layers=False):
    root=Path(__file__).resolve().parents[1]
    from r1.reuse import manifest
    reused=Path(manifest['root'])
    sources={'r1/features.py':root/'r1/features.py','r1/cache.py':root/'r1/cache.py','r1/extract.py':root/'r1/extract.py','r0/acceptance.py':reused/'r0/acceptance.py',
             'models/prob_deformable_sketch_detr.py':root/'models/prob_deformable_sketch_detr.py',
             'models/deformable_transformer.py':root/'models/deformable_transformer.py',
             'models/backbone.py':root/'models/backbone.py',
             'r0/data.py':reused/'r0/data.py','r0/runtime.py':reused/'r0/runtime.py',
             'datasets/transforms.py':root/'datasets/transforms.py',
             'util/misc.py':root/'util/misc.py'}
    return {'data_root':str(Path(config['data_root']).resolve()),'resize':config['resize'],
            'batch_size':config['extract_batch_size'],'train_batch_size':config['batch_size'],
            'seed':config['seed'],'all_layers':bool(all_layers),
            'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},
            'padding':'nested_tensor_from_tensor_list; zeros; max resized H/W in batch',
            'source_sha256':{name:file_sha256(path) for name,path in sources.items()}}


def cache_metadata(config,split,baseline,all_layers=False,decoder_layer_count=6):
    """Shared writer/reader expectation, derived from actual split bytes."""
    if split not in ('train','evaluation'):raise ValueError('Invalid cache split')
    if type(all_layers) is not bool or type(decoder_layer_count) is not int or decoder_layer_count<1:
        raise RuntimeError('Missing/invalid cache extraction layer metadata')
    split_name=config['eval_split'] if split=='evaluation' else config['train_split']
    path=Path(config['data_root'])/'ImageSets/TOWOD'/f'{split_name}.txt'
    ids=path.read_text().splitlines()
    if not ids or any(not stem or stem.strip()!=stem or '/' in stem or '\\' in stem for stem in ids):
        raise RuntimeError('Empty or unsafe image IDs in split')
    return {'baseline':baseline,'split':split,'split_file':split_name,'split_sha256':file_sha256(path),
            'image_ids':sorted(set(ids)),'synthetic':bool(config.get('synthetic',False)),
            'extraction_config':extraction_config(config,all_layers),
            'all_layers':all_layers,'decoder_layer_count':decoder_layer_count,
            'layer_ids':list(range(1,decoder_layer_count+1)) if all_layers else preferred_layers(decoder_layer_count)}
