"""Detached locked round-two controls, durable commits, explicit exit status, never shutdown."""
import argparse,fcntl,os,signal,subprocess,time,json,shutil
from pathlib import Path
from r0.round2.common import *
from r0.runtime import guard_b0_complete
CONTROL=ROOT/'pipeline'

def environment():
    e=os.environ.copy();e.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',MPLCONFIGDIR=str(ROOT/'mpl_cache'),CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1');return e

def guards():
    guard_b0_complete(config());old=json.loads((R0/'pipeline/source_manifest.json').read_text())
    for name,h in old.items():
        if file_sha256(REPO/name)!=h:raise RuntimeError('Round-one model/method source changed: '+name)
    if not json.loads((ROOT/'gpu_probe.json').read_text()).get('passed'):raise RuntimeError('GPU probe failed')
    if not (ROOT/'cpu_tests.log').read_text().strip().endswith('OK'):raise RuntimeError('Controls CPU tests failed')
    if not (ROOT/'original_cpu_tests.log').read_text().strip().endswith('OK'):raise RuntimeError('Original CPU regression failed')
    prep=json.loads((ROOT/'prepared/complete.json').read_text())
    if prep['unique_images']!=9119 or file_sha256(ROOT/'prepared/metadata.csv.gz')!=prep['metadata_sha256'] or file_sha256(ROOT/'prepared/geometry_matches.npz')!=prep['matches_sha256']:raise RuntimeError('Prepared candidate source mismatch')
    if shutil.disk_usage(ROOT).free<8*2**30:raise RuntimeError('Need at least 8GiB disk')

def verify_stage(name):
    if name=='training':
        record=manifest_check(ROOT/'training/complete.json')
        if record['steps_per_bank']!=2000 or record['banks']!=4:raise RuntimeError('Wrong training budget')
        for source in SOURCES:
            for seed in [74,75]:
                rows=[json.loads(l) for l in (ROOT/'training'/f'{source}_seed{seed}'/'train.jsonl').read_text().splitlines()];canonical={r['step']:r for r in rows}
                if set(canonical)!=set(range(1,2001)) or [canonical[k]['image_id'] for k in range(1,2001)]!=sequence():raise RuntimeError('Training image sequence mismatch')
        return {'commit':str(ROOT/'training/complete.json')}
    if name=='loss_probe':manifest_check(ROOT/'loss_probe/complete.json');return {'commit':str(ROOT/'loss_probe/complete.json')}
    if name=='evaluation':
        record=json.loads((ROOT/'evaluation/complete.json').read_text());seen=[]
        if record['scope']!='full_9119_unique' or record['unique_images']!=9119:raise RuntimeError('Incomplete evaluation scope')
        for chunk in record['chunks']:manifest_check(Path(chunk['directory'])/'complete.json');seen+=chunk['image_ids']
        if seen!=record['image_ids'] or len(set(seen))!=9119:raise RuntimeError('Evaluation duplicates/missing')
        return {'commit':str(ROOT/'evaluation/complete.json')}
    if name=='analysis':manifest_check(ROOT/'analysis/complete.json');return {'commit':str(ROOT/'analysis/complete.json')}
    if name=='report':
        record=json.loads((ROOT/'report_complete.json').read_text())
        if not record['complete'] or file_sha256(record['report'])!=record['sha256']:raise RuntimeError('Report digest mismatch')
        return {'commit':str(ROOT/'report_complete.json'),'report':record['report']}
    raise RuntimeError('Unknown stage')

def snapshot():
    hashes=source_hashes();p=ROOT/'source_manifest.json'
    if p.exists() and json.loads(p.read_text())!=hashes:raise RuntimeError('Source changed after launch; review before continuation')
    if not p.exists():
        write_json(p,hashes)
        for name in hashes:
            dest=ROOT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)

def worker(lock_fd):
    signal.signal(signal.SIGHUP,signal.SIG_IGN);os.fstat(lock_fd)
    state=dict(state='running',pid=os.getpid(),sid=os.getsid(0),pgid=os.getpgrp(),started_at=stamp(),stage='preflight',automatic_shutdown=False)
    write_json(CONTROL/'process.json',dict(**state,ppid=os.getppid(),stdin=os.readlink('/proc/self/fd/0'),tty_nr=Path('/proc/self/stat').read_text().split()[6],sighup='ignored'))
    try:
        guards();snapshot();commands=[]
        for name,module in [('training','train'),('loss_probe','loss_probe'),('evaluation','evaluate'),('analysis','analyze'),('report','report')]:
            path=CONTROL/'stages'/f'{name}.json';previous=json.loads(path.read_text()) if path.exists() else {};command=[PYTHON,'-u','-m','r0.round2.'+module];commands.append(dict(stage=name,command=command));write_json(CONTROL/'commands.json',commands)
            try:result=verify_stage(name)
            except (FileNotFoundError,KeyError,RuntimeError):result=None
            if result is not None:
                write_json(path,dict(previous,state='completed',result=result,verified_at=stamp()));print(stamp(),'REUSE',name,flush=True);continue
            attempts=previous.get('attempts',[]);log=CONTROL/'logs'/f'{name}_{len(attempts):02d}.log';entry=dict(command=command,started_at=stamp(),log=str(log));attempts.append(entry)
            phase=dict(state='running',stage=name,attempts=attempts);write_json(path,phase);state.update(stage=name,state='running',stage_log=str(log),last_update=stamp());write_json(CONTROL/'status.json',state)
            print(stamp(),'START',name,flush=True)
            with log.open('x') as handle:
                child=subprocess.Popen(command,cwd=REPO,env=environment(),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,pass_fds=(lock_fd,));entry['child_pid']=child.pid;write_json(path,phase)
                while child.poll() is None:state.update(child_pid=child.pid,last_heartbeat=stamp());write_json(CONTROL/'status.json',state);time.sleep(5)
            entry.update(exit_code=child.returncode,finished_at=stamp());phase['exit_code']=child.returncode
            try:
                if child.returncode!=0:raise RuntimeError(name+' failed; see '+str(log))
                result=verify_stage(name);phase.update(state='completed',result=result,finished_at=stamp());write_json(path,phase)
            except Exception as e:phase.update(state='failed',error=str(e));write_json(path,phase);raise
            print(stamp(),'DONE',name,flush=True)
        state.update(state='completed',stage='complete',finished_at=stamp(),exit_code=0,automatic_shutdown=False);write_json(CONTROL/'status.json',state);write_json(CONTROL/'complete.json',dict(complete=True,finished_at=stamp(),exit_code=0,automatic_shutdown=False,report=str(ROOT/'R0_第二轮有效性验证报告.md')));print(stamp(),'R0_ROUND2_COMPLETE exit_code=0',flush=True)
    except BaseException as e:
        state.update(state='failed',error=str(e),finished_at=stamp(),exit_code=1);write_json(CONTROL/'status.json',state);print(stamp(),'R0_ROUND2_FAILED',repr(e),flush=True);raise

def launch():
    CONTROL.mkdir(exist_ok=True);(CONTROL/'stages').mkdir(exist_ok=True);(CONTROL/'logs').mkdir(exist_ok=True);lock=(CONTROL/'lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print(json.dumps(dict(launched=False,reason='already_running',pid=(CONTROL/'pid').read_text().strip() if (CONTROL/'pid').exists() else None)));return
    if (CONTROL/'complete.json').exists():print(json.dumps(dict(launched=False,reason='already_completed')));return
    guards();snapshot();log=CONTROL/'logs'/f'pipeline_{len(list((CONTROL/"logs").glob("pipeline_*.log"))):02d}.log'
    with log.open('x') as handle:child=subprocess.Popen([PYTHON,'-u','-m','r0.round2.pipeline','--worker','--lock-fd',str(lock.fileno())],cwd=REPO,env=environment(),start_new_session=True,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
    (CONTROL/'pid').write_text(str(child.pid)+'\n');write_json(CONTROL/'launch.json',dict(pid=child.pid,launched_at=stamp(),log=str(log),new_session=True,automatic_shutdown=False));print(json.dumps(dict(launched=True,pid=child.pid,log=str(log)),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--lock-fd',type=int);a=p.parse_args()
    if a.worker:worker(a.lock_fd)
    elif a.launch:launch()
    else:p.error('choose --launch')
