import json, random, hashlib, os, shutil, datetime
from pathlib import Path
import numpy as np
import torch
E=Path('/root/autodl-tmp/experiments/B1_PROB_OWOBJ')
B0=Path('/root/autodl-tmp/experiments/B0_PROB_OWOBJ')
W=Path(__file__).resolve().parents[1]
GIB=1024**3

def now(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)); tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
 return h.hexdigest()
def args_for(name):
 from main_open_world import get_args_parser
 a=get_args_parser().parse_args([]); a.__dict__.update(json.loads((E/f'configs/{name}.json').read_text()))
 a.distributed=False; a.rank=0; a.world_size=1
 return a
def seed(seed=42,training=True):
 torch.set_num_threads(8); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
 torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=not training
 torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=False

def backend(): return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
def build(a):
 from models import build_model
 m,c,p,x=build_model(a,mode=a.model_type)
 return m.cuda(),c.cuda(),p,x.cuda()
def strict_load(m,path):
 ck=torch.load(path,map_location='cpu'); msg=m.load_state_dict(ck['model'],strict=True)
 assert not msg.missing_keys and not msg.unexpected_keys
 print(f'STRICT_LOAD {path} epoch={ck["epoch"]}; {msg}',flush=True)
 return ck

def disk_check(required_gib=5):
 d=shutil.disk_usage(E); state=dict(at=now(),free_bytes=d.free,free_gib=d.free/GIB,required_gib=required_gib,pid=os.getpid())
 write(E/'run_state/disk.json',state)
 if d.free<required_gib*GIB: raise RuntimeError(f'B1 space guard: {state}; protected files will not be deleted')
 return state

def save(path,state):
 disk_check(); path=Path(path); tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f: torch.save(state,f); f.flush(); os.fsync(f.fileno())
 # Retain one previous complete save for recovery; never remove unrelated files.
 backup=path.with_name('checkpoint.previous.pth')
 if path.name=='checkpoint.pth' and path.exists(): os.replace(path,backup)
 os.replace(tmp,path)
def final_link(latest,final):
 latest,final=Path(latest),Path(final)
 if not final.exists(): os.link(latest,final)

def rng_state(): return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(r):
 torch.set_rng_state(r['torch']); torch.cuda.set_rng_state_all(r['cuda']); np.random.set_state(r['numpy']); random.setstate(r['python'])
def move(samples,targets): return samples.to(torch.device('cuda')),[{k:v.cuda() for k,v in t.items()} for t in targets]
