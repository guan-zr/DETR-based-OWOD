"""Detached, locked formal R1; failures stop, immutable commands and stage commits."""
import argparse,fcntl,json,os,signal,subprocess,time,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta

ROOT=Path('/root/autodl-tmp/experiments/R1/formal_20261112')
REPO=Path('/root/autodl-tmp/projects/OWOBJ-R1')
PYTHON='/root/autodl-tmp/experiments/B0_PROB_OWOBJ/env/bin/python'

def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def write(p,value):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(p)
def env():
    e=os.environ.copy();e.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',MPLCONFIGDIR=str(ROOT/'mpl_cache'),WANDB_MODE='disabled',PYTHONUNBUFFERED='1');return e

def commands():
    c=ROOT/'configs';cache=ROOT/'evaluation_cache'
    stages=[('extract',[PYTHON,'-u','-m','r1.extract','--config',str(c/'diagnostic.json'),'--output',str(cache),'--all-layers','--resume']),('diagnose',[PYTHON,'-u','-m','r1.diagnose','--config',str(c/'diagnostic.json'),'--cache',str(cache),'--output',str(ROOT/'diagnostic')])]
    for mask in ['0.6','0','0.3']:
        paths=[str(c/f'{rep}_L{layer}_mask{mask}.json') for rep in ['l2','raw'] for layer in [1,3,6]]
        stages.append(('train_mask'+mask,[PYTHON,'-u','-m','r1.formal_train','--configs',*paths,'--output',str(ROOT/('bank_mask'+mask))]))
        for rep in ['l2','raw']:
            for layer in [1,3,6]:
                name=f'{rep}_L{layer}_mask{mask}';stages.append(('evaluate_'+name,[PYTHON,'-u','-m','r1.evaluate','--config',str(c/(name+'.json')),'--cache',str(cache),'--checkpoint',str(ROOT/name/'checkpoint.pth'),'--output',str(ROOT/name/'evaluation'),'--device','cpu']))
    stages.append(('report',[PYTHON,'-u','-m','r1.formal_report','--root',str(ROOT)]));return stages

def verified(name):
    if name=='extract':
        from r1.cache import FrameReader
        from r1.runtime import config_read
        reader=FrameReader(ROOT/'evaluation_cache',config_read(ROOT/'configs/diagnostic.json'))
        return reader.metadata['images']==9119 and reader.metadata['layer_ids']==list(range(1,7))
    if name=='diagnose':return json.loads((ROOT/'diagnostic/summary.json').read_text())['images']==9119
    if name.startswith('train_mask'):return json.loads((ROOT/('bank_mask'+name[len('train_mask'):])/'complete.json').read_text())['complete']
    if name.startswith('evaluate_'):return json.loads((ROOT/name[len('evaluate_'):]/'evaluation/summary.json').read_text())['images']==9119
    if name=='report':return json.loads((ROOT/'report_complete.json').read_text())['complete']
    return False

def worker(fd):
    signal.signal(signal.SIGHUP,signal.SIG_IGN);os.fstat(fd);control=ROOT/'pipeline';state=dict(state='running',pid=os.getpid(),started_at=stamp(),automatic_shutdown=False)
    write(control/'process.json',dict(**state,sid=os.getsid(0),ppid=os.getppid(),stdin=os.readlink('/proc/self/fd/0'),tty_nr=Path('/proc/self/stat').read_text().split()[6],sighup='ignored'))
    try:
        for p in [ROOT/'real_interface.json',ROOT/'cuda_smoke/complete.json']:
            if not json.loads(p.read_text())['passed']:raise RuntimeError('CUDA gate failed')
        from r1.reuse import file_sha256
        frozen=json.loads((ROOT/'source_manifest.json').read_text())
        for name,h in frozen.items():
            if file_sha256(REPO/name)!=h:raise RuntimeError('R1 core source changed: '+name)
        todo=commands();write(control/'commands.json',[dict(stage=n,command=c) for n,c in todo])
        for name,command in todo:
            path=control/'stages'/(name+'.json')
            try:done=verified(name)
            except (OSError,KeyError,RuntimeError):done=False
            if done:print(stamp(),'VERIFIED REUSE',name,flush=True);continue
            if shutil.disk_usage(ROOT).free<5*2**30:raise RuntimeError('Less than 5GiB disk available')
            old=json.loads(path.read_text()) if path.exists() else {};attempts=old.get('attempts',[]);log=control/'logs'/f'{name}_{len(attempts):02d}.log';entry=dict(command=command,started_at=stamp(),log=str(log));attempts.append(entry)
            # Freeze every actual entry and analysis/report source when its stage begins.
            sources={str(p.relative_to(REPO)):file_sha256(p) for p in (REPO/'r1').glob('*.py')}
            write(control/'sources'/(name+'.json'),sources)
            state.update(stage=name,last_update=stamp(),stage_log=str(log));write(control/'status.json',state);print(stamp(),'START',name,flush=True)
            with log.open('x') as f:
                child=subprocess.Popen(command,cwd=REPO,env=env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,pass_fds=(fd,))
                entry['pid']=child.pid;write(path,dict(state='running',attempts=attempts))
                while child.poll() is None:
                    state.update(child_pid=child.pid,last_heartbeat=stamp());write(control/'status.json',state);time.sleep(5)
            entry.update(exit_code=child.returncode,finished_at=stamp())
            if child.returncode!=0:
                write(path,dict(state='failed',attempts=attempts));raise RuntimeError(name+' failed: '+str(log))
            if not verified(name):raise RuntimeError(name+' missing full success commit')
            write(path,dict(state='completed',attempts=attempts,exit_code=0,finished_at=stamp()));print(stamp(),'DONE',name,flush=True)
        state.update(state='completed',stage='complete',finished_at=stamp(),exit_code=0);write(control/'status.json',state);write(control/'complete.json',state);print(stamp(),'R1_FORMAL_COMPLETE exit_code=0',flush=True)
    except BaseException as e:
        state.update(state='failed',error=str(e),finished_at=stamp(),exit_code=1);write(control/'status.json',state);print(stamp(),'R1_FORMAL_FAILED',repr(e),flush=True);raise

def launch():
    control=ROOT/'pipeline';(control/'logs').mkdir(parents=True,exist_ok=True);(control/'stages').mkdir(exist_ok=True);lock=(control/'lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print(json.dumps(dict(launched=False,reason='already_running')));return
    if (control/'complete.json').exists():print(json.dumps(dict(launched=False,reason='already_completed')));return
    log=control/'logs'/f'pipeline_{len(list((control/"logs").glob("pipeline_*.log"))):02d}.log'
    with log.open('x') as f:child=subprocess.Popen([PYTHON,'-u','-m','r1.formal_pipeline','--worker','--lock-fd',str(lock.fileno())],cwd=REPO,env=env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
    write(control/'launch.json',dict(pid=child.pid,log=str(log),launched_at=stamp(),automatic_shutdown=False));(control/'pid').write_text(str(child.pid)+'\n');print(json.dumps(dict(launched=True,pid=child.pid,log=str(log))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--lock-fd',type=int);a=p.parse_args();worker(a.lock_fd) if a.worker else launch()
