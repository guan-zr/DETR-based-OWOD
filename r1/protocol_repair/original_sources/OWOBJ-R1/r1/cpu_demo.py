"""Small persistent synthetic cache -> real CLI functions -> figures, CPU only."""
import argparse
import copy
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from r1.cache import FrameWriter
from r1.diagnose import run as diagnose
from r1.evaluate import run as evaluate
from r1.plot import run as plot
from r1.reconstruction import QueryAutoencoder
from r1.reuse import write_json
from r1.tests.fixtures import synthetic
from r1.train import train_step


def run(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU fixture must hide CUDA')
    torch.set_num_threads(1);out=Path(output)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Choose new CPU demo output')
    out.mkdir(parents=True,exist_ok=True)
    model,extractor,samples,geometry,batch,outputs,frames,truth=synthetic()
    xml_root=out/'synthetic_dataset';(xml_root/'Annotations').mkdir(parents=True)
    for stem,gt in truth.items():
        tree=ET.Element('annotation')
        for box,known,difficult in zip(gt['boxes'].tolist(),gt['known'].tolist(),gt['difficult'].tolist()):
            obj=ET.SubElement(tree,'object');ET.SubElement(obj,'name').text='aeroplane' if known else 'truck'
            ET.SubElement(obj,'difficult').text=str(int(difficult));node=ET.SubElement(obj,'bndbox')
            box=[box[0]+1,box[1]+1,box[2],box[3]]
            for key,value in zip(['xmin','ymin','xmax','ymax'],box):ET.SubElement(node,key).text=str(value)
        ET.ElementTree(tree).write(xml_root/'Annotations'/f'{stem}.xml')
    cache=out/'cache';writer=FrameWriter(cache,{'split':'evaluation','synthetic':True,'baseline':{'sha256':'synthetic-not-B0'}},.01)
    for frame in frames:writer.write(frame)
    writer.close()
    config=json.loads((Path(__file__).parent/'configs/diagnostic.json').read_text())
    config.update(data_root=str(xml_root),steps=1,autoencoder={'hidden':8,'bottleneck':3},synthetic=True)
    path=out/'synthetic_config.json';write_json(path,config)
    diag=out/'diagnostic';diagnose(path,cache,diag)
    folders=[diag]
    for layer in [1,3,6]:
        current=copy.deepcopy(config);current['layer']=layer
        cfg=out/f'config_L{layer}.json';write_json(cfg,current)
        ae=QueryAutoencoder(8,**current['autoencoder'],mask_ratio=.6)
        optimizer=torch.optim.AdamW(ae.parameters(),lr=.001)
        train_step(ae,batch.representation('l2')[layer-1],optimizer,91)
        checkpoint=out/f'synthetic_L{layer}.pth'
        torch.save({'model':ae.state_dict(),'config':current,'complete':True,'step':1,'channels':8,'layer':layer,
                    'baseline':{'sha256':'synthetic-not-B0'}},checkpoint)
        result=out/f'reconstruction_L{layer}'
        summary=evaluate(cfg,cache,checkpoint,result)
        assert summary['scope']=='synthetic_CPU_diagnostic' and set(summary['statistics'])=={str(layer)}
        folders.append(result)
    plot(folders,out/'figures')
    assert not torch.cuda.is_initialized()
    write_json(out/'verification.json',{'passed':True,'scope':'Synthetic CPU software verification only',
        'cuda_initialized':False,'images':2,'layers':[1,3,6],
        'frame_cache_read_write':True,'diagnostic_entrypoint':True,'independent_ae_entrypoints':True,
        'figures':len(list((out/'figures').glob('*.png')))})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='/root/autodl-tmp/experiments/R1/cpu_smoke/demo')
    args=parser.parse_args();run(args.output)
