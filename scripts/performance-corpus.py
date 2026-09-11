import hashlib,json,random
from pathlib import Path
root=Path(__file__).resolve().parents[1]
split=json.loads((root/'packages/training/data/split-manifest.json').read_text())
rows=json.loads((root/'packages/training/data'/split['directory']/'verification.json').read_text())['records']
words=sorted(r['word'] for r in rows);random.Random(42).shuffle(words)
base=words[:256]
workloads=[{'id':'single-short','words':['كتب']},{'id':'single-median','words':['المدرسة']},{'id':'single-clitic','words':['وبكتابهم']}]
for size in [8,32,128,512,2048]:workloads.append({'id':f'batch-{size}','words':[base[i%len(base)] for i in range(size)]})
uniform=[word for word in words if len(word)==5]
workloads.append({'id':'uniform-128','words':[uniform[i%len(uniform)] for i in range(128)]})
workloads.append({'id':'maximum-shape','words':['ك'*32]*2048,'synthetic':True})
for row in workloads:
 row['sha256']=hashlib.sha256(json.dumps(row['words'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 row['characters']=sum(map(len,row['words']))
 row['lengthClass']=8 if max(map(len,row['words']))<=8 else 16 if max(map(len,row['words']))<=16 else 32
 row['paddingWaste']=1-row['characters']/(row['lengthClass']*len(row['words']))
result={'schemaVersion':1,'normalizationVersion':'arabic-v1','seed':42,'sourceVerificationDigest':split['roles']['verification']['sha256'],'sampling':'Frozen lexical teacher sample, without performance-based selection; not natural text. Maximum shape is explicitly synthetic.','workloads':workloads}
p=root/'packages/benchmark/performance-corpus.json'
payload=json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n'
if p.exists() and p.read_text()!=payload:raise ValueError('Refusing to overwrite frozen performance corpus')
p.write_text(payload)
