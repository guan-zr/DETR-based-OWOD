"""One detached session, sequential GPU ownership, fail closed, truthful step exit codes."""
import argparse, fcntl, json, os, signal, subprocess, sys, threading, time, traceback
from pathlib import Path
from b1.common import *

def run_step(name,args):
 log=E/'logs'/f'{name}.log';begin=now()
 process=None
 with log.open('a') as stream:
  process=subprocess.Popen([sys.executable,'-B','-u','-m',*args],cwd=W,stdout=stream,stderr=subprocess.STDOUT)
  write(E/'run_state'/f'{name}.pid.json',dict(pid=process.pid,parent=os.getpid(),started=begin,command=[sys.executable,'-B','-u','-m',*args],log=str(log)))
  write(E/'run_state/pipeline.json',dict(state='running',step=name,pid=os.getpid(),child_pid=process.pid,at=now(),log=str(log)))
  print(f'START {name} PID={process.pid} log={log}',flush=True)
  def relay(signum,frame):
   if process.poll() is None:process.send_signal(signum)
  for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,relay)
  rc=process.wait()
 record=dict(name=name,started=begin,ended=now(),exit_code=rc,pid=process.pid,log=str(log))
 with (E/'run_state/steps.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 print('STEP_EXIT '+json.dumps(record),flush=True)
 if rc:raise RuntimeError(f'{name} exited {rc}; pipeline stops, no subsequent GPU work')

def gpu_idle():
 result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
 if result:raise RuntimeError('GPU occupied before B1 launch: '+result)

def resources():
 while True:
  try:
   d=shutil.disk_usage(E);gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader'],text=True).strip()
   with (E/'run_state/resources.jsonl').open('a') as f:f.write(json.dumps(dict(at=now(),free_bytes=d.free,gpu=gpu))+'\n')
  except Exception as e:print('RESOURCE_MONITOR '+str(e),flush=True)
  time.sleep(30)

def ensure_train(stage):
 a=args_for(stage);out=Path(a.output_dir);done=E/'run_state'/f'{stage}_complete.json'
 if done.exists():
  assert Path(json.loads(done.read_text())['checkpoint']).is_file();print('Verified complete',stage,flush=True);return
 latest=out/'checkpoint.pth';backup=out/'checkpoint.previous.pth';resume=None
 for path in [latest,backup]:
  if not path.exists():continue
  try:
   ck=torch.load(path,map_location='cpu');assert ck.get('stage')==stage
   resume=str(path);del ck;break
  except Exception as exc:print(f'Recovery candidate {path} unreadable: {exc}',flush=True)
 command=['b1.train','--stage',stage]+(['--resume',resume] if resume else [])
 run_step(stage,command)

def main():
 lock=(E/'run_state/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(E/'run_state/pipeline.pid.json',dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0),at=now()))
 # Existing Task1 selection was launched independently; wait on that PID without GPU contention.
 replay_meta=E/'replay/learned_owod_t1_ft.provenance.json';pidfile=E/'run_state/rebuild_task1_exemplar.pid'
 if not replay_meta.exists() and pidfile.exists():
  pid=int(pidfile.read_text())
  while Path(f'/proc/{pid}').exists():
   state=Path(f'/proc/{pid}/stat').read_text().split()[2]
   if state=='Z':break
   print(f'Waiting for Task1 replay selection pid={pid}',flush=True);time.sleep(30)
 gpu_idle();threading.Thread(target=resources,daemon=True).start()
 run_step('verify_task1_replay',['b1.replay','--stage','task1'])
 if not (E/'reports/smoke_stage_a.json').exists():run_step('smoke_stage_a',['b1.train','--stage','stage_a','--smoke-updates','2'])
 if not (E/'evaluation/smoke_stage_a/manifest.json').exists():run_step('smoke_evaluation',['b1.evaluate','--stage','stage_a','--smoke-limit','10'])
 run_step('report_pretrain',['b1.report'])
 # Recheck measured storage and GPU after successful smoke before any formal training.
 disk_check(10)
 ensure_train('stage_a')
 if not (E/'replay/learned_owod_t2_ft.provenance.json').exists():run_step('task2_exemplar',['b1.replay','--stage','task2'])
 if not (E/'reports/smoke_stage_b.json').exists():run_step('smoke_stage_b',['b1.train','--stage','stage_b','--smoke-updates','2'])
 ensure_train('stage_b')
 for stage in ['stage_a','stage_b']:
  if not (E/f'evaluation/{stage}/manifest.json').exists():run_step('evaluate_'+stage,['b1.evaluate','--stage',stage])
 # Exact protected-content integrity before declaring success.
 protected=json.loads((E/'run_state/protected_before_sha256.json').read_text());changed={p:{'before':h,'after':sha(p)} for p,h in protected.items() if sha(p)!=h}
 write(E/'run_state/protected_after_verification.json',dict(at=now(),checked=len(protected),changed=changed,passed=not changed));assert not changed,changed
 write(E/'run_state/pipeline.json',dict(state='complete',exit_code=0,pid=os.getpid(),at=now()))
 from b1.report import generate
 generate(final=True)
 (E/'task_plan.md').write_text('# B1 Task2 执行完成\n\n- [x] 审计与独立配置\n- [x] exemplar补建和smoke\n- [x] Stage A epoch41–50\n- [x] Stage B epoch51–110\n- [x] 两阶段官方评价与query诊断\n- [x] 中文报告及退出状态\n\n全部完成，详见run_state/pipeline.json和reports/。\n')
 return 0
if __name__=='__main__':
 code=0
 try:code=main()
 except BaseException as exc:
  traceback.print_exc();code=1
  write(E/'run_state/pipeline_failure.json',dict(at=now(),error=repr(exc),traceback=traceback.format_exc()))
  write(E/'run_state/pipeline.json',dict(state='failed',exit_code=1,at=now(),error=repr(exc),pid=os.getpid()))
  try:
   from b1.report import generate
   generate()
  except Exception:traceback.print_exc()
 finally:
  (E/'run_state/pipeline.exit_code').write_text(str(code)+'\n')
  with (E/'run_state/final_resources.txt').open('w') as f:
   f.write(now()+'\n');subprocess.run(['nvidia-smi'],stdout=f,stderr=f);subprocess.run(['df','-h',str(E)],stdout=f,stderr=f)
 raise SystemExit(code)
