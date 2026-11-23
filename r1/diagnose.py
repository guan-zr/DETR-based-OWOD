import argparse
from r1.analysis import evaluate_frames
from r1.cache import FrameReader
from pathlib import Path
from r1.reuse import read_eval_gt,write_json
from r1.runtime import config_read


def run(config_path,cache,output):
    config=config_read(config_path);frames=FrameReader(cache,config)
    if frames.metadata['split']!='evaluation':raise RuntimeError('GT diagnosis accepts evaluation cache only')
    result=evaluate_frames(frames,lambda stem:read_eval_gt(config['data_root'],stem),config['evaluation'],output,
                           synthetic=frames.metadata.get('synthetic',False))
    result['cache_provenance']=frames.metadata;result['baseline']=frames.metadata['baseline'];write_json(Path(output)/'summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();run(args.config,args.cache,args.output)
