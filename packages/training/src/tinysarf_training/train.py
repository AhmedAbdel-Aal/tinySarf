import argparse, copy, json, random, time
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
from .artifacts import ROOT, DATA, GENERATED, RUNS, read_json, write_json, sha, provenance
from .model import Student, make_labels, choose_width, parameter_count, encode_words, encode_analysis
from .export import export_model, parity

def batch_loss(outputs,targets):
    parts=[]
    for name,logits in outputs.items():
        if name=='segmentation': parts.append(2*F.cross_entropy(logits.transpose(1,2),targets[name],ignore_index=-100))
        else: parts.append(F.cross_entropy(logits,targets[name]))
    return torch.stack(parts).mean()

def target_batch(rows,labels,rng):
    # Sample uniformly from the complete mappable teacher set each epoch.
    # This preserves ambiguity rather than privileging the teacher's first output.
    encoded=[encode_analysis(r,rng.choice(r['analyses']),labels) for r in rows]
    return {k:torch.as_tensor(np.stack([a[k] for a in encoded]),dtype=torch.long) for k in encoded[0]}

def run(target=50000,epochs=12,batch_size=128,seed=42,resume=None,replay=None):
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed); rng=random.Random(seed)
    torch.use_deterministic_algorithms(True)
    split=read_json(DATA/'split-manifest.json')
    for role in ['train','verification']:
        if sha(GENERATED/f'{role}.json')!=split['roles'][role]['sha256']: raise ValueError('Frozen dataset hash mismatch')
    train=read_json(GENERATED/'train.json')['records']
    verification=read_json(GENERATED/'verification.json')['records']
    if not train or not verification: raise ValueError('Nonempty train and verification sets required')
    labels=make_labels(train); base_train_count=len(train)
    if replay:
        if len(replay)>max(1,len(train)//10): raise ValueError("Replay bank exceeds 10% of base training words")
        train=train+replay
    width=choose_width(target,labels); model=Student(width,labels)
    if resume:
        checkpoint=torch.load(resume,map_location='cpu',weights_only=True)
        if checkpoint['labels']!=labels or checkpoint['width']!=width: raise ValueError('Resume architecture/label mismatch')
        model.load_state_dict(checkpoint['state'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+f'-{target}'
    path=RUNS/run_id; path.mkdir(parents=True)
    config={'targetParameters':target,'actualParameters':parameter_count(model),'width':width,'embedding':32,'epochs':epochs,'batchSize':batch_size,'seed':seed,'optimizer':'AdamW','learningRate':.003,'weightDecay':.0001,'device':'cpu','threads':4,'torch':torch.__version__,'numpy':np.__version__,'datasetDigests':split['roles'],'resume':resume,'replayWords':len(replay or []),'replayDigest':__import__('hashlib').sha256(__import__('tinysarf_training.artifacts',fromlist=['canonical']).canonical(replay)).hexdigest() if replay else None,'selection':'lowest deterministic sampled verification cross-entropy','ambiguity':'uniform allowed-analysis sampling each epoch'}
    write_json(path/'config.json',config,exclusive=True)
    start=time.perf_counter(); history=[]; best=float('inf'); best_state=None; best_epoch=None
    for epoch in range(epochs):
        model.train(); order=list(range(len(train))); rng.shuffle(order); losses=[]
        for begin in range(0,len(order),batch_size):
            rows=[train[i] for i in order[begin:begin+batch_size]]
            ids=encode_words([r['word'] for r in rows]); target_batch_=target_batch(rows,labels,rng)
            optimizer.zero_grad(set_to_none=True); outputs=model(ids); loss=batch_loss(outputs,target_batch_)
            if not torch.isfinite(loss): raise ValueError('Training diverged')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); val_losses=[]; val_rng=random.Random(seed)
        with torch.no_grad():
            for begin in range(0,len(verification),batch_size):
                rows=verification[begin:begin+batch_size]
                loss=batch_loss(model(encode_words([r['word'] for r in rows])),target_batch(rows,labels,val_rng))
                val_losses.append((float(loss),len(rows)))
        val=sum(loss*n for loss,n in val_losses)/len(verification)
        history.append({'epoch':epoch+1,'trainingLoss':sum(losses)/len(losses),'verificationLoss':val,'elapsedSeconds':time.perf_counter()-start})
        print(json.dumps({'runId':run_id,**history[-1]}),flush=True)
        if val<best: best=val; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1
        write_json(path/'history.json',history)
    model.load_state_dict(best_state); model.eval()
    torch.save({'state':best_state,'width':width,'labels':labels},path/'checkpoint.pt')
    manifest,arrays,quant=export_model(model,path,run_id,split['roles']['verification']['sha256'])
    words=[r['word'] for r in verification[:128]]+['ك','ك'*32,'وبكتابهم']
    parity_result=parity(model,arrays,quant,words)
    write_json(path/'reference-parity.json',{'schemaVersion':1,'model':manifest['id'],'data':split['roles']['verification'],**provenance(),**parity_result},exclusive=True)
    write_json(path/'parity-inputs.json',{'words':words,'float':{k:v.tolist() for k,v in __import__('tinysarf_training.model',fromlist=['numpy_reference']).numpy_reference(encode_words(words).numpy(),arrays).items()},'quantized':{k:v.tolist() for k,v in __import__('tinysarf_training.model',fromlist=['numpy_reference']).numpy_reference(encode_words(words).numpy(),quant).items()}},exclusive=True)
    result={'schemaVersion':1,'runId':run_id,'model':manifest,'data':split['roles'],'training':{'bestEpoch':best_epoch,'verificationLoss':best,'elapsedSeconds':time.perf_counter()-start},'eligibility':{'eligible':False,'reasons':['Independent human gold unavailable','Browser parity and engineering gates not yet evaluated','Clean source provenance required']},**provenance()}
    write_json(path/'result.json',result,exclusive=True)
    print(f'RUN_DIR={path}',flush=True); return path

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--target',type=int,choices=[50000,100000,250000],default=50000)
    p.add_argument('--epochs',type=int,default=12); p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--resume'); p.add_argument('--all',action='store_true')
    a=p.parse_args()
    for t in [50000,100000,250000] if a.all else [a.target]: run(t,a.epochs,a.batch_size,a.seed,a.resume)
