"""Detached, locked, fail-fast server pipeline for the fixed R0 experiment."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from r0.runtime import read_config,guard_b0_complete,write_json,file_sha256
ROOT=Path('/root/autodl-tmp/experiments/R0');CONTROL=ROOT/'pipeline';EVIDENCE=ROOT/'formal_20261111'
REPO=Path(__file__).resolve().parents[1]
PYTHON='/root/autodl-tmp/experiments/B0_PROB_OWOBJ/env/bin/python'

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')

def assert_preflight():
    guard_b0_complete(read_config(REPO/'r0/configs/R0-S1.json'))
    for name in ['gpu_validation.json','resource_probe.json']:
        if json.loads((EVIDENCE/name).read_text()).get('passed') is not True:raise RuntimeError('Preflight failed: '+name)
    if 'Ran 29 tests' not in (EVIDENCE/'cpu_final.log').read_text() or not (EVIDENCE/'cpu_final.log').read_text().strip().endswith('OK'):
        raise RuntimeError('CPU regression not accepted')
    if not (EVIDENCE/'pipeline_tests_v2.log').read_text().strip().endswith('OK'):raise RuntimeError('Pipeline tests not accepted')
    if not json.loads((EVIDENCE/'official_source_check.json').read_text()).get('passed'):raise RuntimeError('Detector/preprocessing source mismatch')
    if not json.loads((EVIDENCE/'streaming_resume_check.json').read_text()).get('passed'):raise RuntimeError('Streaming resume not verified')
    if shutil.disk_usage(ROOT).free<10*2**30:raise RuntimeError('Need >=10GiB disk before launch')

def environment():
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
        MPLCONFIGDIR=str(EVIDENCE/'mpl_cache'),WANDB_MODE='disabled',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
    return env

def snapshot():
    from r0.formal_evaluate import source_manifest
    manifest=source_manifest();p=CONTROL/'source_manifest.json'
    if p.exists():
        if json.loads(p.read_text())!=manifest:raise RuntimeError('Pipeline code changed since first launch; explicit review/new output needed')
    else:
        write_json(p,manifest)
        for name in manifest:
            dest=CONTROL/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
    return manifest

def verify_training(directory,source):
    import torch
    config=read_config(REPO/f'r0/configs/R0-{source}.json');p=directory/'checkpoint.pth'
    if not p.is_file() or not (directory/'training_complete.json').is_file():raise RuntimeError('Training artifacts incomplete')
    saved=torch.load(p,map_location='cpu');end=json.loads((directory/'training_complete.json').read_text())
    from r0.acceptance import EXPECTED_SHA
    if not saved.get('complete') or saved['step']!=2000 or saved['config']!=config or saved['baseline']['sha256']!=EXPECTED_SHA or not end['complete'] or end['steps']!=2000:raise RuntimeError('Training checkpoint/config/baseline mismatch')
    logs=[json.loads(l) for l in (directory/'train.jsonl').read_text().splitlines() if l.strip()];steps={r['step']:r for r in logs}
    expected=(EVIDENCE/'train_image_sequence.txt').read_text().splitlines()
    if set(steps)!=set(range(1,2001)) or [steps[i]['image_ids'][0] for i in range(1,2001)]!=expected:raise RuntimeError('Training sequence/budget mismatch')
    return dict(checkpoint=str(p),checkpoint_sha256=file_sha256(p),steps=2000,train_sequence_sha256=json.loads((EVIDENCE/'preflight.json').read_text())['train_sequence_sha256'])

def stage_run(name,command,verify,artifact_directory,lock_fd,progress):
    state_path=CONTROL/'stages'/f'{name}.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get('state')=='completed':
        verify();print(now(),'SKIP verified completed',name,flush=True);return state
    # An interruption may happen after the artifact commits but before state commits.
    try:result=verify()
    except (FileNotFoundError,RuntimeError,KeyError):result=None
    if result is not None:
        state.update(state='completed',verified_at=now(),result=result,artifact_directory=str(artifact_directory),recovered_committed_artifact=True)
        write_json(state_path,state);return state
    attempt=len(state.get('attempts',[]));log=CONTROL/'logs'/f'{name}_{attempt:02d}.log'
    history=state.get('attempts',[]);entry=dict(attempt=attempt,started_at=now(),command=command,log=str(log))
    history.append(entry);state=dict(state='running',stage=name,artifact_directory=str(artifact_directory),attempts=history,started_at=now())
    write_json(state_path,state);progress.update(state='running',stage=name,stage_log=str(log),last_update=now());write_json(CONTROL/'status.json',progress)
    print(now(),'START',name,command,flush=True)
    with log.open('x') as handle:
        child=subprocess.Popen(command,cwd=REPO,env=environment(),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,pass_fds=(lock_fd,))
        entry['child_pid']=child.pid;write_json(state_path,state)
        while child.poll() is None:
            progress.update(last_heartbeat=now(),child_pid=child.pid);write_json(CONTROL/'status.json',progress);time.sleep(5)
        code=child.returncode
    entry.update(returncode=code,finished_at=now());state['finished_at']=now();state['returncode']=code
    try:
        if code!=0:raise RuntimeError(f'{name} failed (exit {code}); see {log}')
        state['result']=verify();state['state']='completed';write_json(state_path,state)
    except Exception as error:
        state.update(state='failed',error=str(error));write_json(state_path,state);raise
    print(now(),'DONE',name,flush=True);return state

def training_directory(source):
    state_path=CONTROL/'stages'/f'{source}_train.json'
    if state_path.exists():
        old=json.loads(state_path.read_text());d=Path(old['artifact_directory'])
        if (d/'checkpoint.pth').is_file():return d
        # A partial attempt with no recoverable checkpoint is preserved, never overwritten.
    parent=ROOT/f'R0-{source}';parent.mkdir(exist_ok=True)
    return parent/f'attempt_{len(list(parent.glob("attempt_*"))):02d}'

def verify_commit(directory,marker,name):
    record=json.loads((directory/marker).read_text())
    if not record.get('complete'):raise RuntimeError('Stage commit incomplete')
    if name=='full_evaluation':
        if record.get('unique_images')!=9119 or record.get('scope')!='full_9119_unique':raise RuntimeError('Full evaluation scope mismatch')
        seen=[]
        for chunk in record['chunks']:
            seen+=chunk['image_ids']
            for filename,digest in chunk['sha256'].items():
                if file_sha256(Path(chunk['directory'])/filename)!=digest:raise RuntimeError('Damaged complete evaluation chunk')
        if seen!=record['image_ids'] or len(set(seen))!=9119:raise RuntimeError('Missing or duplicate images')
    if name=='analysis':
        for filename,digest in record['artifacts'].items():
            if file_sha256(directory/filename)!=digest:raise RuntimeError('Analysis artifact changed')
    if name=='report' and file_sha256(record['report'])!=record['sha256']:raise RuntimeError('Report digest mismatch')
    return dict(commit=str(directory/marker),sha256=file_sha256(directory/marker))

def worker(lock_fd):
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    # Inherited file description holds one exclusive lock through parent and active child.
    os.fstat(lock_fd)
    assert_preflight();snapshot()
    (CONTROL/'pipeline.pid').write_text(str(os.getpid())+'\n')
    progress=dict(state='running',pid=os.getpid(),sid=os.getsid(0),pgid=os.getpgrp(),started_at=now(),stage='initializing')
    write_json(CONTROL/'process.json',dict(**progress,stdin=os.readlink('/proc/self/fd/0'),tty_nr=Path('/proc/self/stat').read_text().split()[6],sighup='ignored',cwd=str(REPO)))
    write_json(CONTROL/'status.json',progress);commands=[];paths={}
    try:
        for source in ['S1','S2']:
            d=training_directory(source);cmd=[PYTHON,'-u','-m','r0.train','--config',f'r0/configs/R0-{source}.json','--output',str(d)]
            if (d/'checkpoint.pth').exists():cmd+=['--resume',str(d/'checkpoint.pth')]
            commands.append(dict(stage=source+'_train',command=cmd));write_json(CONTROL/'commands.json',commands)
            s=stage_run(source+'_train',cmd,lambda d=d,s=source:verify_training(d,s),d,lock_fd,progress);paths[source]=s['result']['checkpoint']
        stages=[
            ('full_evaluation',[PYTHON,'-u','-m','r0.formal_evaluate','--s1-checkpoint',paths['S1'],'--s2-checkpoint',paths['S2']],ROOT/'paired_evaluation','complete.json'),
            ('analysis',[PYTHON,'-u','-m','r0.formal_analysis'],ROOT/'analysis','complete.json'),
            ('report',[PYTHON,'-u','-m','r0.formal_report'],ROOT/'reports','report_complete.json')]
        for name,cmd,directory,marker in stages:
            def verify(d=directory,m=marker,n=name):return verify_commit(d,m,n)
            commands.append(dict(stage=name,command=cmd));write_json(CONTROL/'commands.json',commands)
            stage_run(name,cmd,verify,directory,lock_fd,progress)
        progress.update(state='completed',stage='complete',finished_at=now());write_json(CONTROL/'status.json',progress)
        write_json(CONTROL/'complete.json',dict(complete=True,finished_at=now(),report=str(ROOT/'reports/R0_S1_S2_正式验证报告.md')))
        print(now(),'R0_PIPELINE_COMPLETE',flush=True)
    except BaseException as error:
        progress.update(state='failed',error=str(error),failed_at=now());write_json(CONTROL/'status.json',progress)
        print(now(),'R0_PIPELINE_FAILED',repr(error),flush=True);raise

def launch():
    CONTROL.mkdir(exist_ok=True);(CONTROL/'logs').mkdir(exist_ok=True);(CONTROL/'stages').mkdir(exist_ok=True)
    lock=(CONTROL/'pipeline.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps(dict(launched=False,reason='already_running_lock_held',status=str(CONTROL/'status.json'),pid=(CONTROL/'pipeline.pid').read_text().strip() if (CONTROL/'pipeline.pid').exists() else None)));return
    if (CONTROL/'complete.json').exists():
        print(json.dumps(dict(launched=False,reason='already_completed',result=str(CONTROL/'complete.json'))));return
    assert_preflight();snapshot()
    log=CONTROL/'logs'/f'pipeline_launch_{len(list((CONTROL/"logs").glob("pipeline_launch_*.log"))):02d}.log'
    with log.open('x') as handle:
        child=subprocess.Popen([PYTHON,'-u','-m','r0.pipeline','--worker','--lock-fd',str(lock.fileno())],cwd=REPO,env=environment(),
            start_new_session=True,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
    (CONTROL/'pipeline.pid').write_text(str(child.pid)+'\n')
    write_json(CONTROL/'launch.json',dict(pid=child.pid,launched_at=now(),log=str(log),new_session=True,lock_inherited=True))
    print(json.dumps(dict(launched=True,pid=child.pid,log=str(log),status=str(CONTROL/'status.json')),ensure_ascii=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--lock-fd',type=int);a=p.parse_args()
    if a.worker:
        try:worker(a.lock_fd)
        except BaseException as error:
            state_path=CONTROL/'status.json'
            state=json.loads(state_path.read_text()) if state_path.exists() else dict(stage='initialization',pid=os.getpid())
            state.update(state='failed',error=str(error),failed_at=now());write_json(state_path,state)
            raise
    elif a.launch:launch()
    else:p.error('Choose --launch (safe to repeat)')
if __name__=='__main__':main()
