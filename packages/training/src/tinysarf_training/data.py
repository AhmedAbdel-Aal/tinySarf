import argparse, collections, gzip, json, random, shutil, sys, tarfile, urllib.request
from pathlib import Path
from .artifacts import DATA, GENERATED, ROOT, sha, read_json, write_json, provenance
from .contract import normalize, map_analysis, LETTERS
from .splits import split_records

def prepare(limit=12000):
    manifest=read_json(DATA/'corpus.json'); downloads=DATA/'downloads'; downloads.mkdir(exist_ok=True)
    paths=[downloads/'camel_morph_msa_v1.0.db',downloads/'camel-tools.tar.gz']
    for source,path in zip(manifest['sources'],paths):
        compressed=path.with_suffix(path.suffix+'.gz')
        if not path.exists() and compressed.exists():
            with gzip.open(compressed,'rb') as src,path.open('wb') as dst: shutil.copyfileobj(src,dst)
        if not path.exists():
            tmp=path.with_suffix('.part'); urllib.request.urlretrieve(source['url'],tmp); tmp.rename(path)
        digest=sha(path)
        if source.get('sha256') and digest!=source['sha256']: raise ValueError('Teacher checksum mismatch')
        if not source.get('sha256'): raise ValueError('Pin source checksum in corpus.json before compilation')
    teacher=downloads/'camel_tools-source'
    if not teacher.exists():
        teacher.mkdir()
        with tarfile.open(paths[1]) as archive:
            for member in archive.getmembers():
                parts=Path(member.name).parts[1:]
                if not parts or member.issym() or member.islnk(): continue
                member.name=str(Path(*parts)); archive.extract(member,teacher,filter='data')
    sys.path.insert(0,str(teacher))
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    print('Loading pinned teacher',flush=True)
    db=MorphologyDB(str(paths[0]),'a'); analyzer=Analyzer(db,backoff='NONE')
    surfaces=sorted(s for s,entries in db.stem_hash.items() if s and all(c in LETTERS for c in s) and len(s)<=24
      and any(a.get('source')!='wiki' and a.get('pos')!='noun_prop' for _,a in entries))
    rng=random.Random(manifest['seed']); rng.shuffle(surfaces)
    words=set(); prefixes=['','و','ف','ال','بال','وال','وب']; suffixes=['','ه','هم','نا']
    for stem in surfaces:
        words.add(stem)
        words.add(rng.choice(prefixes)+stem+rng.choice(suffixes))
        if len(words)>=limit: break
    fixtures=read_json(DATA/'contract-fixtures.json'); words.update(f['word'] for f in fixtures)
    rows=[]; dropped=collections.Counter(); rejected=[]
    for i,word in enumerate(sorted(words)):
        if len(word)>32: continue
        analyses={}
        for raw in analyzer.analyze(word):
            try:
                a=map_analysis(word,raw); key=json.dumps(a,ensure_ascii=False,sort_keys=True); analyses[key]=a
            except ValueError as e:
                dropped[str(e)]+=1
                if len(rejected)<100: rejected.append({'word':word,'reason':str(e),'raw':raw})
        if analyses:
            aa=list(analyses.values()); roots={a['root'] for a in aa if a['root']}; slices=[]
            if any(any(c in root for c in 'اوىي') for root in roots): slices.append('weak-root')
            if any(any(c in root for c in 'ءأإؤئآ') for root in roots): slices.append('hamzated')
            if any(len(root)==3 and root[1]==root[2] for root in roots): slices.append('doubled')
            if any(len(root)==4 for root in roots): slices.append('quadriliteral')
            if any(len(a['spans'])>=3 for a in aa): slices.append('clitic-stack')
            rows.append({'word':word,'analyses':aa,'slices':slices,'labelOrigin':'teacher-generated'})
        else: dropped['no-mappable-analysis']+=1
        if i%1000==0: print(f'{i}/{len(words)} surfaces, {len(rows)} retained',flush=True)
    splits=split_records(rows,[f['lemmaFamily'] for f in fixtures],manifest['seed'])
    generated=GENERATED; generated.mkdir(parents=True,exist_ok=True)
    frozen={'schemaVersion':1, 'datasetId':'teacher-sampled-v2', 'directory':'generated/teacher-sampled-v2', 'normalizationVersion':'arabic-v1','sourceManifestDigest':sha(DATA/'corpus.json'), 'limit':limit,'seed':manifest['seed'],'roles':{},'sampling':manifest['sampling']}
    for role,data in splits.items():
        path=generated/f'{role}.json'
        payload={'schemaVersion':1,'role':role,'records':data}
        if path.exists() and read_json(path)!=payload: raise ValueError('Refusing to overwrite frozen split. Use a new dataset directory/version.')
        write_json(path,payload)
        frozen['roles'][role]={'sha256':sha(path),'words':len(data),'analyses':sum(len(r['analyses']) for r in data),'lemmaFamilies':len({a['lemmaFamily'] for r in data for a in r['analyses']})}
    write_json(DATA/'split-manifest.json',frozen)
    write_json(DATA/'heldout-identities.json',{'schemaVersion':1,'splitManifestSha256':sha(DATA/'split-manifest.json'),**{role:{'words':sorted(r['word'] for r in splits[role]),'lemmaFamilies':sorted({a['lemmaFamily'] for r in splits[role] for a in r['analyses']})} for role in ['verification','final-test']}})
    write_json(generated/'mapping-audit.json',{'dropped':dict(dropped),'examples':rejected,**provenance()})
    print(json.dumps(frozen,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=12000); a=p.parse_args(); prepare(a.limit)
