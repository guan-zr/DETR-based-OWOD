"""Read configuration and readiness only; no detector, dataset or GPU execution."""
import argparse
import json
from r0.runtime import guard_b0_complete,read_config


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=read_config(args.config)
    try:guard_b0_complete(config);ready=True;reason='B0完整流水线已结束；可显式启动R0。'
    except (RuntimeError,FileNotFoundError,KeyError) as error:ready=False;reason=str(error)
    print(json.dumps({'ready':ready,'reason':reason,'config':config},indent=2,ensure_ascii=False))
