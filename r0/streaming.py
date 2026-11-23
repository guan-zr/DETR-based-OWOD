"""Memory-bounded paired extraction/scoring. The AE/FMR and ROI definitions are reused verbatim."""
import numpy as np
import torch
from r0.types import SpatialBatch
from r0.regions import mean_region_error

@torch.no_grad()
def extract_both(s2_extractor,samples,geometry):
    captured={}
    def before(module,inputs):
        captured['srcs']=[x.detach() for x in inputs[0]]
        captured['masks']=[m.detach() for m in inputs[1]]
    handle=s2_extractor.detector.transformer.register_forward_pre_hook(before)
    try:s2,outputs=s2_extractor(samples,geometry)
    finally:handle.remove()
    s1=SpatialBatch([x.masked_fill(m[:,None],0).detach() for x,m in zip(captured['srcs'],captured['masks'])],
        captured['masks'],s2.strides,geometry,'S1').validate()
    return {'S1':s1,'S2':s2},outputs


def region_records(batch,errors,pools,settings):
    """Same arithmetic as r0.evaluate.region_records; transfer vectors once per scale."""
    if len(batch.geometry)!=1:raise ValueError('Offline evaluation requires batch1')
    geometry=batch.geometry[0];records=[]
    kwargs={k:settings[k] for k in ('output_size','sampling_ratio','min_box_size','min_valid_fraction')}
    for source,pool in pools.items():
        boxes=pool['boxes'].cpu();per_scale=[]
        for level,(maps,stride) in enumerate(zip(errors,batch.strides)):
            scored=mean_region_error(maps['error'][0],maps['valid'][0],boxes,geometry,stride,**kwargs)
            masked=mean_region_error(maps['masked_error'][0],maps['masked_support'][0],boxes,geometry,stride,**dict(kwargs,min_valid_fraction=0.0))
            score=scored.scores.cpu().tolist();valid=scored.valid.cpu().tolist()
            frac=scored.valid_fraction.cpu().tolist();small=scored.small_feature_roi.cpu().tolist()
            mscore=masked.scores.cpu().tolist();mvalid=masked.valid.cpu().tolist();rows=[]
            for index,(box,group) in enumerate(zip(boxes.tolist(),pool['labels'])):
                rows.append(dict(image_id=geometry.image_id,source=source,candidate_id=index,
                    origin=pool['origin'][index],group=group,max_iou_gt=pool['max_iou'][index],level=f's{level}_stride{stride}',
                    x1=box[0],y1=box[1],x2=box[2],y2=box[3],score=score[index] if valid[index] else None,
                    masked_score=mscore[index] if mvalid[index] else None,valid=valid[index],valid_fraction=frac[index],small_feature_roi=small[index]))
            records.extend(rows);per_scale.append(rows)
        for index in range(len(boxes)):
            levels=[rows[index] for rows in per_scale];overall=dict(levels[0],level='overall')
            overall['valid']=all(r['valid'] for r in levels)
            overall['score']=float(np.mean([r['score'] for r in levels])) if overall['valid'] else None
            overall['masked_score']=float(np.mean([r['masked_score'] for r in levels])) if all(r['masked_score'] is not None for r in levels) else None
            overall['valid_fraction']=min(r['valid_fraction'] for r in levels)
            overall['small_feature_roi']=any(r['small_feature_roi'] for r in levels)
            records.append(overall)
    return records
