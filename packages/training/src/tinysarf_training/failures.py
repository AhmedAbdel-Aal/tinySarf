"""Reviewed mining/external failures enter replay; held-out words and lemma families never do."""
import argparse, hashlib
from pathlib import Path
from .artifacts import ROOT,DATA,GENERATED,read_json,write_json,sha
from .contract import normalize,SPAN_TYPES,FEATURE_MAP

def validate_failure(entry,heldout):
    review=entry.get('review',{})
    if entry.get('schemaVersion')!=1 or review.get('status')!='approved' or review.get('independentHuman') is not True or not review.get('reviewer') or not review.get('reviewedAt'):
        raise ValueError('An independent human must approve the failure labels before replay')
    if entry.get('source',{}).get('role') not in ('mining','external') or not entry.get('source',{}).get('license'):
        raise ValueError('Failure source must be licensed mining or external data')
    row=entry['record'];word=row['word']
    if normalize(word)!=word: raise ValueError('Replay uses normalized surfaces and offsets')
    if not row.get('analyses'): raise ValueError('Retain all accepted analyses')
    families={a.get('lemmaFamily') for a in row['analyses']}
    if not families or None in families or '' in families: raise ValueError('Lemma families are required')
    if word in heldout['words'] or families & heldout['families']: raise ValueError('Failure overlaps verification/final-test; never replay held-out data')
    allowed_pos={'noun','proper_noun','numeral','adjective','verb','adverb','preposition','conjunction','interjection','pronoun','particle','unknown'}
    features={target:set(values.values())|{'__missing__','__unknown__','__na__'} for _,(target,values) in FEATURE_MAP.items()}
    for analysis in row['analyses']:
        end=0
        for span in analysis['spans']:
            if span['type'] not in SPAN_TYPES or type(span['start'])!=int or type(span['end'])!=int or span['start']!=end or span['end']<=end or span['end']>len(word): raise ValueError('Invalid replay spans')
            end=span['end']
        if end!=len(word) or not any(s['type']=='stem' for s in analysis['spans']): raise ValueError('Replay requires complete stem coverage')
        if analysis['pos'] not in allowed_pos: raise ValueError('Invalid replay POS')
        root=analysis['root']
        if root is not None and (len(root) not in (3,4) or normalize(root)!=root): raise ValueError('Invalid replay root')
        if analysis['pattern'] is not None and (not isinstance(analysis['pattern'],str) or not analysis['pattern']): raise ValueError('Invalid replay pattern')
        if any(k not in features or v not in features[k] for k,v in analysis['features'].items()): raise ValueError('Invalid replay feature')
    return row

def heldout_identities():
    # Split identities only: never inspect final-test labels for mining or optimization.
    split=read_json(DATA/'split-manifest.json');words=set();families=set()
    identities=read_json(DATA/'heldout-identities.json')
    if identities['splitManifestSha256']!=sha(DATA/'split-manifest.json'): raise ValueError('Held-out identity manifest is stale; reconstruct data')
    for role in ['verification','final-test']:
        words.update(identities[role]['words']);families.update(identities[role]['lemmaFamilies'])
    for file in (DATA/'generated/audit').glob('gold-identities-*.json'):
        gold=read_json(file);words.update(gold['words']);families.update(gold['lemmaFamilies'])
    return {'words':words,'families':families}

def replay_records():
    heldout=heldout_identities();rows={}
    for file in sorted((ROOT/'packages/training/failures/reviewed').glob('*.json')):
        row=validate_failure(read_json(file),heldout)
        if row['word'] in rows: raise ValueError('Duplicate replay surface; consolidate accepted analyses')
        rows[row['word']]=row
    return list(rows.values())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--add',type=Path);p.add_argument('--run',action='store_true');p.add_argument('--epochs',type=int,default=3)
    a=p.parse_args()
    if a.add:
        entry=read_json(a.add);validate_failure(entry,heldout_identities())
        destination=ROOT/'packages/training/failures/reviewed'/f'{sha(a.add)}.json'
        write_json(destination,entry,exclusive=True);print(f'Reviewed failure retained: {destination}')
    rows=replay_records();print(f'{len(rows)} independently reviewed replay words available')
    if a.run:
        if not rows: raise ValueError('No reviewed failures; refusing empty fine-tuning')
        from .train import run
        pointer=ROOT/'packages/training/active/current.json'
        if not pointer.exists(): pointer=ROOT/'packages/training/candidate.json'
        selected=read_json(pointer);directory=ROOT/selected['directory'];config=read_json(directory/'config.json')
        run(config['targetParameters'],a.epochs,config['batchSize'],config['seed'],str(directory/'checkpoint.pt'),replay=rows)
        print('Candidate created only. Evaluate and use model:promote; replay never updates active weights.')
if __name__=='__main__':main()
