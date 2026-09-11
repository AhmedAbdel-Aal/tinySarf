import hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
DATA=ROOT/'packages/training/data'
GENERATED=DATA/'generated/teacher-sampled-v2'
RUNS=ROOT/'packages/training/runs'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def write_json(path,value,exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb' if exclusive else 'wb') as f: f.write(canonical(value))
def read_json(path): return json.loads(Path(path).read_text())
def provenance(command=None):
    def git(*args):
        p=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True)
        return p.stdout.strip() if p.returncode==0 else None
    status=git('status','--porcelain')
    return {'git':{'commit':git('rev-parse','HEAD'),'dirty':None if status is None else bool(status)},
      'createdAt':datetime.now(timezone.utc).isoformat(),'commands':[command or ' '.join(sys.argv)],
      'environment':{'os':platform.platform(),'python':platform.python_version(),'browser':None,'webgpuAdapter':None}}
