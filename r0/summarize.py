"""CPU-only metrics/replot from a streamed region CSV, no detector required."""
import argparse
import csv
import gzip
from pathlib import Path

from r0.metrics import ScoreAccumulator
from r0.runtime import write_json
from r0.visualize import group_distributions


def run(path,output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise RuntimeError('Choose an empty summary output')
    output.mkdir(parents=True,exist_ok=True)
    accumulator=ScoreAccumulator()
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='') as handle:
        for row in csv.DictReader(handle):
            row['valid']=row['valid'].lower() in ('true','1')
            row['score']=float(row['score']) if row['score'] else None
            accumulator.add(row)
    write_json(output/'summary.json',accumulator.summarize())
    group_distributions(accumulator.groups,output/'figures')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--records',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.records,args.output)
