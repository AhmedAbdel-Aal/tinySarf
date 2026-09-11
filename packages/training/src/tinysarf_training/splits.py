import hashlib

ROLES=('train','verification','mining','final-test')
def digest(data): return hashlib.sha256(data).hexdigest()

def split_records(records, reserved_lemmas=(), seed=42):
    """Union all lemmas connected by an ambiguous surface before allocating families."""
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)]=min(a,b)
    for row in records:
        lemmas=sorted(set(a['lemmaFamily'] for a in row['analyses']))
        for lemma in lemmas: union(lemmas[0],lemma)
    reserved={find(x) for x in reserved_lemmas}
    result={role:[] for role in ROLES}
    for row in sorted(records,key=lambda x:x['word']):
        family=find(row['analyses'][0]['lemmaFamily'])
        bucket=int(digest(f'{seed}:{family}'.encode())[:8],16)%100
        role='verification' if family in reserved else 'train' if bucket<75 else 'verification' if bucket<85 else 'mining' if bucket<90 else 'final-test'
        result[role].append({**row,'familyGroup':family})
    validate_splits(result)
    return result

def validate_splits(splits):
    seen_surface={}; seen_lemma={}
    for role,rows in splits.items():
        for row in rows:
            if row['word'] in seen_surface: raise ValueError('surface overlap or duplicate')
            seen_surface[row['word']]=role
            for a in row['analyses']:
                lemma=a['lemmaFamily']
                if lemma in seen_lemma and seen_lemma[lemma]!=role: raise ValueError('lemma-family leakage')
                seen_lemma[lemma]=role
