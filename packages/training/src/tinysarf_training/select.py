"""Stage an experimental candidate for local builds. This never promotes an active model."""
import argparse
from pathlib import Path
import torch
from .artifacts import ROOT, RUNS, DATA, read_json, write_json, sha, provenance
from .model import Student
from .export import export_model

def select(paths):
    entries=[]
    for path in paths:
        path=Path(path); config=read_json(path/'config.json'); result=read_json(path/'result.json')
        if not read_json(path/'reference-parity.json')['floatPassed']: continue
        entries.append((result['training']['verificationLoss'],path,config,result))
    if not entries: raise ValueError('No completed candidates pass float reference parity')
    objectives={bool(e[2].get('rootTraining')) for e in entries}
    if len(objectives)!=1: raise ValueError('Root marginal loss cannot be compared with multi-task cross-entropy')
    digests={e[3]['model']['verificationDigest'] for e in entries}
    if len(digests)!=1: raise ValueError('Candidates were evaluated on different verification data')
    _,path,config,result=min(entries,key=lambda x:(x[0],x[1].name))
    checkpoint=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=Student(checkpoint['width'],checkpoint['labels'],root_width=checkpoint.get('rootWidth')); model.load_state_dict(checkpoint['state']); model.eval()
    model.root_decoder=checkpoint.get('rootDecoder') or read_json(path/'manifest.json').get('rootDecoder')
    destination=ROOT/'packages/training/candidates'/path.name
    if destination.exists(): raise ValueError('Experimental candidate already staged; refusing overwrite')
    manifest,_,_=export_model(model,destination,path.name,result['model']['verificationDigest'])
    if manifest != read_json(path/'manifest.json'):
        raise ValueError('Staging changed the frozen model manifest')
    torch.save(checkpoint,destination/'checkpoint.pt')
    for filename in ['config.json','result.json','reference-parity.json','parity-inputs.json','history.json']:
        write_json(destination/filename,read_json(path/filename),exclusive=True)
    selection={'schemaVersion':1,'status':'experimental-unpromoted','id':path.name,'directory':str(destination.relative_to(ROOT)),
      'manifestSha256':sha(destination/'manifest.json'),'reason':('Root architecture and data experiment selected on teacher verification; root checkpoint selected by complete-root marginal loss and decoder by root agreement. Independent human gold remains absent; no final-test access.' if config.get('rootTraining') else 'Lowest sampled verification cross-entropy among completed candidate sizes; no final-test access.'),
      'candidates':[{'id':e[1].name,'parameters':e[2]['actualParameters'],'verificationLoss':e[0]} for e in sorted(entries)],**provenance()}
    write_json(ROOT/'packages/training/candidate.json',selection)
    print(destination)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');a=p.parse_args();select(a.runs)
