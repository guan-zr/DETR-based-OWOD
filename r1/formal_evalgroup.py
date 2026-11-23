"""Independent CPU evaluations with bounded concurrency and per-output locks."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from r1.runtime import config_read
from r1.reuse import write_json,file_sha256


def run_one(path):
    config=config_read(path);root=Path(config['output']);out=root/'evaluation';lock=(root/'evaluation.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if (out/'summary.json').exists():
        old=json.loads((out/'summary.json').read_text())
        if old.get('images')!=9119 or old.get('evaluation_config')!=config:raise RuntimeError('Existing evaluation mismatch')
        return dict(config=str(path),verified_reuse=True,summary=str(out/'summary.json'))
    command=[os.sys.executable,'-u','-m','r1.evaluate','--config',str(path),'--cache',str(root.parent/'evaluation_cache'),'--checkpoint',str(root/'checkpoint.pth'),'--output',str(out),'--device','cpu']
    record=dict(command=command,config=str(path),checkpoint_sha256=file_sha256(root/'checkpoint.pth'),started_unix=time.time(),log=str(root/'evaluation_run.log'))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (root/'evaluation_run.log').open('a') as f:
        child=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);record['pid']=child.pid;write_json(root/'evaluation_process.json',record);code=child.wait()
    record.update(exit_code=code,finished_unix=time.time());write_json(root/'evaluation_process.json',record)
    if code!=0:raise RuntimeError('CPU evaluation failed: '+str(root))
    summary=json.loads((out/'summary.json').read_text())
    if summary['images']!=9119 or summary['evaluation_config']!=config:raise RuntimeError('Incomplete evaluation')
    record['summary_sha256']=file_sha256(out/'summary.json');return record


def run(paths,out):
    out=Path(out);results=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(run_one,p) for p in paths]):
            results.append(f.result());write_json(out/'evaluation_group_progress.json',dict(completed=len(results),total=len(paths)))
    write_json(out/'evaluation_group_complete.json',dict(complete=True,evaluations=results,max_parallel_cpu_evaluations=6))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.configs,a.output)
