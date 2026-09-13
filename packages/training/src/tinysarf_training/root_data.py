"""Root-only teacher supervision, independent of concatenative segmentation alignment."""
import argparse
import collections
import random
from pathlib import Path

from .artifacts import DATA, ROOT, read_json, write_json, sha, provenance
from .contract import LETTERS, dediac
from .corrected import pinned_teacher, teacher_families, exclude_connected, families
from .failures import heldout_identities


def compatible_candidates(db, stems, limit, rng):
    """Sample the teacher's inflection grammar, including verb person prefixes.

    Compatibility categories determine valid prefix/stem/suffix combinations;
    the resulting surface is still reanalyzed to recover its complete ambiguity.
    """
    prefixes, suffixes = collections.defaultdict(set), collections.defaultdict(set)
    for surface, entries in db.prefix_hash.items():
        if all(c in LETTERS for c in surface):
            for category, _ in entries: prefixes[category].add(surface)
    for surface, entries in db.suffix_hash.items():
        if all(c in LETTERS for c in surface):
            for category, _ in entries: suffixes[category].add(surface)
    prefixes = {k: sorted(v, key=lambda s: (len(s), s)) for k, v in prefixes.items()}
    suffixes = {k: sorted(v, key=lambda s: (len(s), s)) for k, v in suffixes.items()}
    by_stem = collections.defaultdict(set)
    for prefix, compatible in db.prefix_stem_compat.items():
        if prefix in prefixes:
            for stem in compatible: by_stem[stem].add(prefix)
    choices = {}
    for stem, prefix_categories in by_stem.items():
        choices[stem] = []
        for prefix in sorted(prefix_categories):
            suffix_categories = sorted(set(db.stem_suffix_compat.get(stem, [])) & set(db.prefix_suffix_compat.get(prefix, [])) & suffixes.keys())
            if suffix_categories: choices[stem].append((prefix, suffix_categories))
    def surface(options):
        if rng.random() < .5:
            shortest = len(options[0])
            return rng.choice([s for s in options if len(s) == shortest])
        return rng.choice(options)
    words = set()
    for stem in stems:
        words.add(stem)
        categories = sorted({category for category, raw in db.stem_hash[stem]
            if raw.get('source') != 'wiki' and raw.get('pos') != 'noun_prop' and choices.get(category)})
        for _ in range(3):
            if not categories: break
            category = rng.choice(categories)
            prefix, suffix_categories = rng.choice(choices[category])
            suffix = rng.choice(suffix_categories)
            word = surface(prefixes[prefix]) + stem + surface(suffixes[suffix])
            if len(word) <= 32: words.add(word)
        if len(words) >= limit: break
    return words


def prepare(limit=60000, seed=43, paradigms=False):
    destination = DATA / f'generated/teacher-roots-v{2 if paradigms else 1}-{limit}'
    if destination.exists(): raise ValueError('Refusing to overwrite a frozen root dataset')
    protected = heldout_identities()
    corrected = DATA / 'generated/teacher-corrected-pilot-v1'
    exclusions = read_json(corrected / 'exclusions.json')
    protected['words'].update(exclusions['blockedWords'])
    protected['families'].update(exclusions['blockedLemmaFamilies'])
    rows, counts = [], collections.Counter()
    with pinned_teacher() as (db, analyzer):
        # Only identities are retained here, including for the sealed test inputs.
        # Neither root labels nor other held-out annotations enter this dataset.
        for word in sorted(protected['words']):
            protected['families'].update(teacher_families(analyzer.analyze(word), word))
        stems = sorted(word for word, entries in db.stem_hash.items()
            if word and len(word) <= 24 and all(c in LETTERS for c in word)
            and any(raw.get('source') != 'wiki' and raw.get('pos') != 'noun_prop' for _, raw in entries))
        rng = random.Random(seed)
        rng.shuffle(stems)
        candidates = set()
        if paradigms:
            candidates = compatible_candidates(db, stems, limit, rng)
        else:
            prefixes, suffixes = ['', 'و', 'ف', 'ال', 'بال', 'وال', 'وب', 'لل', 'كال'], ['', 'ه', 'ها', 'هم', 'نا', 'كم']
            for stem in stems:
                candidates.add(stem)
                candidates.add(rng.choice(prefixes) + stem + rng.choice(suffixes))
                if len(candidates) >= limit: break
        print(f'Root data: {len(stems)} eligible stem surfaces; {len(candidates)} candidates', flush=True)
        for i, word in enumerate(sorted(candidates)):
            if word in protected['words'] or len(word) > 32: continue
            raw_analyses = analyzer.analyze(word)
            fs = teacher_families(raw_analyses, word)
            if set(fs) & protected['families']:
                counts['directProtectedFamily'] += 1
            analyses = {}
            inflections = set()
            for raw in raw_analyses:
                root = ''.join(c for c in dediac(raw.get('root', '')) if c in LETTERS)
                root = root if len(root) in (3, 4) else None
                # Same root spelling/absence policy as the frozen verification set.
                family = teacher_families([raw], word)[0]
                analyses[(root, family)] = {'root': root, 'lemmaFamily': family}
                if paradigms and raw.get('pos') == 'verb':
                    prefix_tags = [part.rsplit('/', 1)[-1] for part in raw.get('bw', '').split('+') if '/' in part and part.rsplit('/', 1)[-1].startswith('IV') and part.rsplit('/', 1)[-1] != 'IV' and not part.rsplit('/', 1)[-1].startswith('IVSUFF')]
                    inflections.add('aspect-' + str(raw.get('asp')))
                    inflections.update('prefix-' + tag for tag in prefix_tags)
                try:
                    from .contract import map_analysis
                    map_analysis(word, raw)
                except ValueError:
                    counts['analysesRecoveredWithoutSpanAlignment'] += 1
            if analyses:
                rows.append({'word': word, 'analyses': list(analyses.values()),
                    'teacherLemmaFamilies': fs, 'labelOrigin': 'teacher-generated-root-only',
                    **({'inflectionCoverage': sorted(inflections)} if paradigms else {})})
            else: counts['noTeacherAnalysis'] += 1
            if i % 5000 == 0: print(f'Root data: {i}/{len(candidates)} candidates, {len(rows)} retained before family closure', flush=True)
    rows, excluded = exclude_connected(rows, protected['words'], protected['families'])
    # Retain direct family collisions until closure so every selected ambiguity
    # component is excluded, including aliases introduced by training candidates.
    rows.sort(key=lambda r: r['word'])
    coverage = collections.Counter(label for row in rows for label in row.get('inflectionCoverage', []))
    if any(row['word'] in protected['words'] or families(row) & protected['families'] for row in rows):
        raise ValueError('Root dataset leaked an excluded identity')
    destination.mkdir(parents=True)
    write_json(destination / 'train.json', {'schemaVersion': 1, 'role': 'train', 'records': rows})
    write_json(destination / 'exclusions.json', {'blockedWords': sorted(protected['words']),
               'blockedLemmaFamilies': sorted(protected['families'])})
    verification = corrected / 'verification.json'
    metadata = {'schemaVersion': 1, 'datasetId': destination.name,
        'sourceManifestSha256': sha(DATA / 'corpus.json'), 'seed': seed, 'candidateLimit': limit,
        'sampling': 'teacher-compatible inflections and clitics' if paradigms else 'bare stems and sampled noun clitics',
        'roles': {'train': {'sha256': sha(destination / 'train.json'), 'words': len(rows),
                   'analyses': sum(len(r['analyses']) for r in rows)},
                  'verification': {'sha256': sha(verification), 'words': len(read_json(verification)['records'])}},
        'verificationPath': str(verification.relative_to(DATA)),
        'exclusionsSha256': sha(destination / 'exclusions.json'),
        'counts': dict(counts), 'excludedConnectedWords': len(excluded), 'inflectionCoverage': dict(coverage),
        'heldoutIdentitySha256': sha(DATA / 'heldout-identities.json'),
        'finalTestLabelsOpened': False, 'aiAnnotationsUsed': False,
        'limitations': ['Root-only teacher data; no independent human annotation',
            'No segment labels are fabricated for non-concatenative surface forms',
            'Root spelling and null representation match the frozen teacher contract'], **provenance()}
    write_json(destination / 'manifest.json', metadata)
    print('ROOT_DATASET=' + str(destination), flush=True)
    print(__import__('json').dumps(metadata, ensure_ascii=False), flush=True)
    return destination


def load(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    train_path = directory / 'train.json'
    verification_path = DATA / manifest['verificationPath']
    if sha(train_path) != manifest['roles']['train']['sha256'] or sha(verification_path) != manifest['roles']['verification']['sha256']:
        raise ValueError('Root dataset digest mismatch')
    if sha(directory / 'exclusions.json') != manifest['exclusionsSha256']:
        raise ValueError('Root exclusions digest mismatch')
    rows = {'train': read_json(train_path)['records'], 'verification': read_json(verification_path)['records']}
    exclusions = read_json(directory / 'exclusions.json')
    kept, dropped = exclude_connected(rows['train'], set(exclusions['blockedWords']), set(exclusions['blockedLemmaFamilies']))
    if dropped or len(kept) != len(rows['train']): raise ValueError('Root data overlaps protected identities')
    return manifest, rows


def prepare_selected():
    """Reconstruct every root supervision pool used by the selected candidate."""
    pointer = ROOT / 'packages/training/active/current.json'
    if not pointer.exists(): pointer = ROOT / 'packages/training/candidate.json'
    selected = read_json(pointer)
    config = read_json(ROOT / selected['directory'] / 'config.json')
    if not config.get('rootTraining'): return
    if not (DATA / 'generated/teacher-corrected-pilot-v1/manifest.json').exists():
        from .corrected import prepare as prepare_corrected
        prepare_corrected()
    root = config['rootTraining']
    pools = config.get('rootTrainingPools', [{'directory': root['datasetDirectory'], 'sha256': root['dataset']['train']['sha256']}])
    for pool in pools:
        directory = (ROOT / pool['directory']).resolve()
        if not directory.is_relative_to(DATA / 'generated'): raise ValueError('Root data path escapes generated datasets')
        if not directory.exists():
            audit = read_json(DATA / 'audits' / (directory.name + '-manifest.json'))
            actual = prepare(audit['candidateLimit'], audit['seed'], directory.name.startswith('teacher-roots-v2-'))
            if actual != directory: raise ValueError('Reconstructed the wrong root dataset')
        load(directory)
        if sha(directory / 'train.json') != pool['sha256']: raise ValueError('Reconstructed root training bytes differ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=60000)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--paradigms', action='store_true')
    parser.add_argument('--selected', action='store_true')
    args = parser.parse_args()
    if args.selected: prepare_selected()
    else: prepare(args.limit, args.seed, args.paradigms)
