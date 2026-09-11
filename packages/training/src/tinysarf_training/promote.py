"""Re-evaluate candidates, hash eligibility evidence, and atomically switch an active pointer."""
import argparse, os, shutil, subprocess, tempfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .artifacts import ROOT,DATA,read_json,write_json,sha,provenance
from .gates import assess

REQUIRED_FILES=['manifest.json','weights.bin','float.weights.bin','float.npz','checkpoint.pt','config.json','result.json','reference-parity.json','parity-inputs.json','history.json']
def integrity(directory):
    m=read_json(directory/'manifest.json')
    for file,key in [('weights.bin','sha256'),('float.npz','floatSha256'),('float.weights.bin','floatBinarySha256')]:
        if sha(directory/file)!=m[key]:raise ValueError(f'Checkpoint hash mismatch: {file}')
    split=read_json(DATA/'split-manifest.json')
    if m['verificationDigest']!=split['roles']['verification']['sha256']:raise ValueError('Model verification digest mismatch')
    for role in ['train','verification']:
        if sha(DATA/split['directory']/f'{role}.json')!=split['roles'][role]['sha256']:raise ValueError('Dataset hash mismatch')
    for file in REQUIRED_FILES:
        if not (directory/file).is_file():raise ValueError(f'Incomplete checkpoint: {file}')
    return m

def run(command,directory):
    env={**os.environ,'TINYSARF_MODEL_DIR':str(directory)}
    p=subprocess.run(command,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout,flush=True)
    if p.returncode:raise RuntimeError(f'Validation failed: {command}')
    return p.stdout

def artifact_from(output,prefix=None):
    import json
    for line in reversed(output.splitlines()):
        if prefix and line.startswith(prefix):return Path(line[len(prefix):])
        try:
            obj=json.loads(line)
            if isinstance(obj,dict) and obj.get('file'):return Path(obj['file'])
        except json.JSONDecodeError:pass
    raise ValueError('Validation did not emit an artifact path')

def metrics(m):
    return {k:m[k]['value'] for k in ['segmentationExactMatch','rootExactMatch','fullAnalysisTop1','fullAnalysisTop3']}|{'posMacroF1':m['pos']['macroF1'],'featureMacroF1':m['featureMacroF1']}

def summarize(c,s,b,p,directory,reproduction):
    correctness=c['correctness'];gold=[correctness.get('goldNatural'),correctness.get('goldBalanced')]
    independent=all(x and x['metadata']['labelOrigin']=='human-independent' for x in gold)
    beats=independent and all(all(x['metrics'][key]['value']>base[key]['value'] for key in ['segmentationExactMatch','rootExactMatch','fullAnalysisTop3']) for x in gold for base in x['baselines'].values())
    slices={name:{**value,'metrics':metrics(value['metrics'])} for name,value in correctness['slices'].items()}
    input_slices={}
    for dataset in gold:
        if dataset:
            for name,value in dataset.get('slices',{}).items():
                if value['count']:slices[name]={**value,'metrics':metrics(value['metrics'])}
                if value['inputRejection']['count']:input_slices[name]=value['inputRejection']
    batch=next((w for w in b['warm'] if w['id']=='batch-128'),None)
    rp=read_json(directory/'reference-parity.json');fixture=read_json(DATA/'contract-fixtures.json')
    return {'integrityVerified':True,'cleanReproduction':reproduction is not None,'humanReviewedFixtures':len(fixture)==50 and all(f.get('reviewStatus')=='approved' and f.get('review',{}).get('independentHuman') is True and f.get('review',{}).get('reviewer') for f in fixture),
      'independentGold':bool(independent),'goldMetrics':{name:metrics(x['metrics']) for name,x in zip(['gold-natural','gold-balanced'],gold) if x},'goldBeatsBaselines':bool(beats),'browserContractPassed':p['passed'],'git':c['git'],'verificationDigest':c['data']['verificationDigest'],'normalizationVersion':c['data']['normalizationVersion'],
      'metrics':metrics(correctness['teacherAgreement']),'slices':slices,'inputSlices':input_slices,
      'parity':{'floatMaxAbsoluteError':max(t['floatMaxAbsoluteError'] for t in rp['tensors'].values()),'gpuArgmaxAgreement':p['parity']['headArgmaxMatches']/p['parity']['headArgmaxCount'],'deterministicRuns':p['deterministic']['runs'],'deterministic':p['deterministic']['deterministic'],'finite':p['parity']['finite']},
      'packedWeightsBytes':s['size']['packedWeightsBytes'],'brotliPackageBytes':s['size']['completePackage']['brotli'],'wrapperBrotliBytes':s['size']['wrapper']['brotli'],
      'coldP95Ms':b['cold']['summary']['webgpu']['totalMs']['p95'],'warmBatch128P95Ms':batch['webgpu']['p95'] if batch else None,
      'gpuBufferBytes':b['diagnostic'].get('peakAllocatedGPUBytes'),'coldRuns':b['protocol']['coldRuns'],'warmups':b['protocol']['warmups'],'warmIterations':b['protocol']['iterations'],'warmRepeats':b['protocol']['repeats'],
      'warmupStable':b['protocol'].get('warmupStabilityPassed') is True,'fullBrowserProtocol':b['protocol']['releaseQualifying'],
      'measuredBatchAdvantage':any(8<=w['batch']<=512 and not w['synthetic'] and w['speedup']>1 for w in b['warm']),
      'referenceDevice':{k:b['environment'].get(k) for k in ['deviceModel','osVersion','browser','adapter','headless']}}

def atomic_activate(directory,report_file,active_root,repository_root=ROOT):
    """Immutable checkpoint directories plus one atomic pointer; an old pointer always remains usable."""
    m=read_json(directory/'manifest.json');releases=active_root/'releases';releases.mkdir(parents=True,exist_ok=True)
    version=m['id']+'-'+sha(report_file)[:12];destination=releases/version
    if destination.exists():raise ValueError('Promotion already exists')
    temporary=Path(tempfile.mkdtemp(prefix='.stage-',dir=releases))
    try:
        for filename in REQUIRED_FILES:shutil.copy2(directory/filename,temporary/filename)
        shutil.copy2(report_file,temporary/'eligibility.json')
        os.replace(temporary,destination)
        pointer={'schemaVersion':1,'id':m['id'],'directory':str(destination.relative_to(repository_root)),'manifestSha256':sha(destination/'manifest.json'),'promotedAt':datetime.now(timezone.utc).isoformat(),'eligibilitySha256':sha(report_file)}
        pending=active_root/'.current.json.tmp';write_json(pending,pointer);os.replace(pending,active_root/'current.json')
        return destination
    finally:
        if temporary.exists():shutil.rmtree(temporary)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path);parser.add_argument('--gold',type=Path,action='append',default=[]);parser.add_argument('--reproduction',type=Path);parser.add_argument('--check',action='store_true',help='Collect bounded smoke evidence and report ineligibility without activating')
    args=parser.parse_args();source=provenance();policy=read_json(ROOT/'packages/training/promotion-policy.json')
    directory=(args.candidate or ROOT/read_json(ROOT/'packages/training/candidate.json')['directory']).resolve()
    report_dir=ROOT/'packages/training/promotions/local';report_dir.mkdir(parents=True,exist_ok=True)
    report_file=report_dir/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
    report={'schemaVersion':1,'candidate':str(directory),'policySha256':sha(ROOT/'packages/training/promotion-policy.json'),**source,'eligible':False,'reasons':[]}
    try:
        model=integrity(directory);report['model']=model;report['checkpointHashes']={file:sha(directory/file) for file in REQUIRED_FILES}
        if source['git']['dirty'] is not False or not source['git']['commit']:raise ValueError('Promotion requires a committed, clean working tree')
        reproduction=None
        if args.reproduction:
            reproduction=read_json(args.reproduction)
            if reproduction.get('kind')!='reproduction' or not reproduction.get('passed') or reproduction['git']!=source['git'] or reproduction.get('modelSha256')!=model['sha256'] or reproduction.get('separateCheckout') is not True:raise ValueError('Clean-clone reproduction evidence is stale or invalid')
            report['reproductionSha256']=sha(args.reproduction)
        active_pointer=ROOT/'packages/training/active/current.json';active_dir=ROOT/read_json(active_pointer)['directory'] if active_pointer.exists() else None
        evidence=[];summaries=[]
        for model_dir in [directory]+([active_dir] if active_dir else []):
            integrity(model_dir)
            run(['pnpm','test'],model_dir);run(['pnpm','build:core'],model_dir);run(['pnpm','check:package'],model_dir)
            command=['node','--import','tsx','packages/benchmark/src/correctness.ts','--local','--model-dir',str(model_dir)]
            if args.gold:
                command+=['--release-candidate']
                for file in args.gold:command+=['--gold',str(file.resolve())]
            cf=artifact_from(run(command,model_dir))
            sf=artifact_from(run(['node','--import','tsx','packages/benchmark/src/size.ts','--local'],model_dir))
            browser=['node','--import','tsx','packages/benchmark/src/browser.ts','--local']
            if args.check:browser+=['--smoke','--workloads','single-short,batch-8,batch-128']
            bf=artifact_from(run(browser,model_dir),'BROWSER_ARTIFACT=');b=read_json(bf);pf=bf.parent/b['parityArtifact']
            artifacts=[cf,sf,bf,pf];c,s,b,p=map(read_json,artifacts)
            for artifact in [c,s,b,p]:
                if artifact['model']['sha256']!=read_json(model_dir/'manifest.json')['sha256'] or artifact['git']!=source['git']:raise ValueError('Model/source artifact provenance mismatch')
            evidence.append({str(f.relative_to(ROOT)):sha(f) for f in artifacts});summaries.append(summarize(c,s,b,p,model_dir,reproduction))
        report.update(assess(summaries[0],summaries[1] if len(summaries)>1 else None,policy));report['evidence']=evidence;report['summaries']=summaries
        if len(summaries)>1:
            c,a=[read_json(ROOT/next(k for k in e if '/correctness-' in k))['correctness']['teacherAgreement']['perWord'] for e in evidence]
            rng=np.random.default_rng(42);paired={}
            for key in c:
                diffs=np.asarray(c[key])-np.asarray(a[key]);means=np.array([rng.choice(diffs,len(diffs),replace=True).mean() for _ in range(2000)])
                paired[key]={'difference':float(diffs.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'seed':42,'resamples':2000}
            report['pairedDifferences']=paired
        write_json(report_file,report,exclusive=True);(report_file.with_suffix('.sha256')).write_text(sha(report_file)+'\n')
        if report['eligible'] and not args.check:
            old=active_pointer.read_bytes() if active_pointer.exists() else None
            try:
                destination=atomic_activate(directory,report_file,ROOT/'packages/training/active')
                run(['pnpm','docs:generate'],destination)
                print(f'Active checkpoint: {destination}')
            except Exception:
                if old is None:active_pointer.unlink(missing_ok=True)
                else:
                    pending=active_pointer.with_suffix('.rollback');pending.write_bytes(old);os.replace(pending,active_pointer)
                raise
        else:print('NOT PROMOTED: '+'; '.join(report['reasons']))
    except Exception as error:
        report['eligible']=False;report['reasons'].append(str(error))
        if not report_file.exists():write_json(report_file,report,exclusive=True);report_file.with_suffix('.sha256').write_text(sha(report_file)+'\n')
        print(f'NOT PROMOTED: {error}')
    finally:print(f'ELIGIBILITY_REPORT={report_file}')
    return 0 if report['eligible'] else 1
if __name__=='__main__':raise SystemExit(main())
