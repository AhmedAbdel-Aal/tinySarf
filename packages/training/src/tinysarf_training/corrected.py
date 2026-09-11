"""Opt-in teacher repair and bounded pilot; never selects or promotes a model.

Only train/verification labels are read. Sealed and AI challenge identities are
used for exclusion, with AI lemma identities inferred from the pinned teacher.
The legacy preparation entry point and its output directory remain unchanged.
"""
import argparse
import collections
import contextlib
import copy
import gzip
import hashlib
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from datetime import datetime, timezone

from .artifacts import ROOT, DATA, RUNS, canonical, read_json, write_json, sha, provenance
from .contract import (CORRECTED_MAPPING, LETTERS, POS_MAP, map_analysis,
                       lemma_family, normalize)
from .data import extract_teacher
from .failures import ai_challenge_words, heldout_identities

DATASET_ID = 'teacher-corrected-pilot-v1'
COVERAGE_POS = ('preposition', 'pronoun', 'particle', 'conjunction',
                'interjection', 'adverb', 'numeral')
MAX_COVERAGE_FRACTION = 0.10


def mapped_pos(raw_pos):
    return POS_MAP.get(raw_pos, 'pronoun' if raw_pos.startswith('pron') else
                       'particle' if raw_pos.startswith('part') else 'unknown')


def load_legacy_roles(data_directory=DATA):
    """Intentionally has no final-test label-loading branch."""
    split = read_json(data_directory / 'split-manifest.json')
    if split.get('datasetId') != 'teacher-sampled-v2':
        raise ValueError('The corrected pilot requires the frozen v2 base dataset')
    directory = (data_directory / split['directory']).resolve()
    if not directory.is_relative_to(data_directory.resolve()):
        raise ValueError('Base dataset directory escapes training data')
    roles = {}
    for role in ('train', 'verification'):
        source = directory / f'{role}.json'
        if sha(source) != split['roles'][role]['sha256']:
            raise ValueError(f'Frozen {role} dataset hash mismatch')
        roles[role] = read_json(source)['records']
    return split, roles


@contextlib.contextmanager
def pinned_teacher():
    """Extract verified sources to a temporary directory, never the frozen cache."""
    sources = read_json(DATA / 'corpus.json')['sources']
    downloads = DATA / 'downloads'
    archive = downloads / 'camel-tools.tar.gz'
    if sha(archive) != sources[1]['sha256']:
        raise ValueError('Pinned teacher source archive hash mismatch')
    with tempfile.TemporaryDirectory(prefix='tinysarf-corrected-teacher-') as tmp:
        temporary = Path(tmp)
        database = temporary / 'teacher.db'
        original = downloads / 'camel_morph_msa_v1.0.db'
        opener = original.open if original.exists() else lambda mode: gzip.open(
            downloads / 'camel_morph_msa_v1.0.db.gz', mode)
        digest = hashlib.sha256()
        with opener('rb') as source, database.open('wb') as dest:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
                dest.write(chunk)
        if digest.hexdigest() != sources[0]['sha256']:
            raise ValueError('Pinned teacher database hash mismatch')
        teacher = temporary / 'source'
        extract_teacher(archive, teacher)
        if any(name == 'camel_tools' or name.startswith('camel_tools.') for name in sys.modules):
            raise ValueError('Use a fresh process so teacher imports come from the pinned archive')
        sys.path.insert(0, str(teacher))
        try:
            from camel_tools.morphology.database import MorphologyDB
            from camel_tools.morphology.analyzer import Analyzer
            db = MorphologyDB(str(database), 'a')
            yield db, Analyzer(db, backoff='NONE')
        finally:
            sys.path.remove(str(teacher))


def teacher_families(raw_analyses, word):
    return sorted({lemma_family(a.get('lex', word)) for a in raw_analyses})


def corrected_row(word, raw_analyses, rejected):
    analyses = {}
    for raw in raw_analyses:
        try:
            analysis = map_analysis(word, raw, version=CORRECTED_MAPPING)
            analyses[canonical(analysis)] = analysis
        except ValueError as error:
            rejected[str(error)] += 1
    if not analyses:
        return None
    accepted = list(analyses.values())
    roots = {a['root'] for a in accepted if a['root']}
    slices = []
    if any(any(c in root for c in 'اوىي') for root in roots): slices.append('weak-root')
    if any(any(c in root for c in 'ءأإؤئآ') for root in roots): slices.append('hamzated')
    if any(len(root) == 3 and root[1] == root[2] for root in roots): slices.append('doubled')
    if any(len(root) == 4 for root in roots): slices.append('quadriliteral')
    if any(len(a['spans']) >= 3 for a in accepted): slices.append('clitic-stack')
    return {'word': word, 'analyses': accepted, 'slices': slices,
            'labelOrigin': 'teacher-generated',
            'teacherLemmaFamilies': teacher_families(raw_analyses, word)}


def families(row):
    return set(row.get('teacherLemmaFamilies', [])) | {
        a['lemmaFamily'] for a in row['analyses']}


def exclude_connected(rows, blocked_words, blocked_families):
    """Exclude entire ambiguity-connected components, including unmappable aliases."""
    parent = {}

    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    for row in rows:
        ids = sorted(families(row))
        for value in ids:
            left, right = find(ids[0]), find(value)
            parent[max(left, right)] = min(left, right)
    blocked = {find(value) for value in blocked_families}
    for row in rows:
        if row['word'] in blocked_words:
            blocked.update(find(value) for value in families(row))
    kept, dropped = [], []
    for row in rows:
        if row['word'] in blocked_words or any(find(value) in blocked for value in families(row)):
            dropped.append(row)
        else:
            kept.append({**row, 'familyGroup': find(next(iter(families(row))))})
    return kept, dropped


def coverage_candidates(db, seed=42, per_pos=128):
    """Get teacher POS strata from lexical tables, without a hand-picked word list."""
    if not 1 <= per_pos <= 256:
        raise ValueError('Candidate bound must be between 1 and 256 per POS')
    groups = {pos: set() for pos in COVERAGE_POS}
    for table_name in ('stem_hash', 'prefix_hash', 'suffix_hash'):
        for word, entries in getattr(db, table_name).items():
            if not word or len(word) > 32 or any(c not in LETTERS for c in word):
                continue
            for _, raw in entries:
                pos = mapped_pos(raw.get('pos', ''))
                if pos in groups and raw.get('source') != 'wiki':
                    groups[pos].add(word)
    rng = random.Random(seed)
    chosen = set()
    report = {}
    for pos in COVERAGE_POS:
        values = sorted(groups[pos])
        rng.shuffle(values)
        selected = values[:per_pos]
        chosen.update(selected)
        report[pos] = {'availableSurfaces': len(values), 'candidateSurfaces': len(selected)}
    return sorted(chosen), report


def choose_supplement(rows, base_words, seed=42, per_pos=32):
    if not 0 <= per_pos <= 64:
        raise ValueError('Supplement bound must be between 0 and 64 per POS')
    rng = random.Random(seed)
    groups = {pos: [] for pos in COVERAGE_POS}
    for row in rows:
        if row['word'] not in base_words:
            for pos in {a['pos'] for a in row['analyses']} & groups.keys():
                groups[pos].append(row['word'])
    selected = set()
    for pos in COVERAGE_POS:
        values = sorted(groups[pos])
        rng.shuffle(values)
        selected.update(values[:per_pos])
    return [row for row in rows if row['word'] in selected]


def supervision_coverage(rows):
    support, expected = collections.Counter(), collections.Counter()
    for row in rows:
        support.update({a['pos'] for a in row['analyses']})
        for analysis in row['analyses']:
            expected[analysis['pos']] += 1 / len(row['analyses'])
    all_pos = sorted(set(support) | set(COVERAGE_POS))
    return {pos: {'wordSupport': support[pos],
                  'expectedOrdinaryTargetsPerEpoch': expected[pos]} for pos in all_pos}


def epoch_samples(rows, rng, coverage_fraction=0.10, targets_per_pos=64):
    """At least 90.9% of draws remain ordinary uniform-analysis supervision."""
    if not 0 <= coverage_fraction <= MAX_COVERAGE_FRACTION:
        raise ValueError('Coverage draws must be at most 10% of ordinary draws')
    if not 0 <= targets_per_pos <= 128:
        raise ValueError('Coverage targets must be bounded at 128 per POS')
    pools = {pos: [] for pos in COVERAGE_POS}
    samples = []
    for index, row in enumerate(rows):
        samples.append((index, rng.randrange(len(row['analyses'])), False))
        by_pos = collections.defaultdict(list)
        for analysis_index, analysis in enumerate(row['analyses']):
            if analysis['pos'] in pools:
                by_pos[analysis['pos']].append(analysis_index)
        for pos, indices in by_pos.items():
            pools[pos].append((index, indices))
    supported = [pos for pos in COVERAGE_POS if pools[pos]]
    budget = min(math.floor(len(rows) * coverage_fraction), targets_per_pos * len(supported))
    forced = collections.Counter()
    for i in range(budget):
        pos = supported[i % len(supported)]
        row_index, indices = rng.choice(pools[pos])
        samples.append((row_index, rng.choice(indices), True))
        forced[pos] += 1
    rng.shuffle(samples)
    counts = collections.Counter(rows[i]['analyses'][a]['pos'] for i, a, _ in samples)
    return samples, {'ordinarySamples': len(rows), 'coverageSamples': budget,
                     'ordinaryFraction': len(rows) / len(samples) if samples else 0,
                     'sampledPosCounts': dict(counts), 'forcedPosCounts': dict(forced),
                     'unsupportedCoveragePOS': [pos for pos in COVERAGE_POS if not pools[pos]]}


def prepare(output=None, seed=42, candidate_limit=128, supplement_per_pos=32):
    output = Path(output or DATA / 'generated' / DATASET_ID).resolve()
    if (not output.is_relative_to((DATA / 'generated').resolve()) or
            not output.name.startswith('teacher-corrected-') or output.exists()):
        raise ValueError('Use a new teacher-corrected-* directory under training/data/generated')
    split, base = load_legacy_roles()
    protected = heldout_identities()  # Identities only, including sealed final-test metadata.
    ai_words = ai_challenge_words(DATA / 'ai-review')
    fixtures = read_json(DATA / 'contract-fixtures.json')
    protected['words'].update(normalize(row['word']) for row in fixtures)
    protected['families'].update(row['lemmaFamily'] for row in fixtures)
    rejected = collections.Counter()
    ai_families = set()
    repaired = {}
    verification_words = {row['word'] for row in base['verification']}
    with pinned_teacher() as (db, analyzer):
        for word in sorted(ai_words):
            # Never retain or train on the AI words' teacher labels: identities only.
            ai_families.update(teacher_families(analyzer.analyze(word), word))
        candidates, candidate_report = coverage_candidates(db, seed, candidate_limit)
        words = sorted({row['word'] for role in base.values() for row in role} | set(candidates))
        for i, word in enumerate(words):
            # Excluded AI identities never enter the data-building path.
            if word in ai_words and word not in verification_words:
                continue
            row = corrected_row(word, analyzer.analyze(word), rejected)
            if row is not None:
                repaired[word] = row
            if i % 1000 == 0:
                print(f'corrected teacher {i}/{len(words)} surfaces', flush=True)
    verification = [repaired[row['word']] for row in base['verification'] if row['word'] in repaired]
    if len(verification) != len(base['verification']):
        raise ValueError('Correction lost a base verification surface; investigate before running a pilot')
    protected['families'].update(ai_families)
    protected['families'].update(value for row in verification for value in families(row))
    train_candidates = [row for word, row in repaired.items()
                        if word not in verification_words]
    allowed, excluded = exclude_connected(train_candidates, protected['words'], protected['families'])
    base_words = {row['word'] for row in base['train']}
    supplement = choose_supplement(allowed, base_words, seed, supplement_per_pos)
    train = sorted([row for row in allowed if row['word'] in base_words] + supplement,
                   key=lambda row: row['word'])
    if not train or len({r['word'] for r in train}) != len(train):
        raise ValueError('Corrected training set is empty or has duplicate surfaces')
    if any(row['word'] in protected['words'] or families(row) & protected['families'] for row in train):
        raise ValueError('Training overlaps an excluded identity')
    coverage = supervision_coverage(train)
    roles = {'train': train, 'verification': verification}
    original_verification = {row['word']: row for row in base['verification']}
    changes = collections.Counter()
    for row in verification:
        before = original_verification[row['word']]['analyses']
        for field in ('root', 'pos', 'pattern', 'spans', 'features'):
            if {canonical(a[field]) for a in before} != {canonical(a[field]) for a in row['analyses']}:
                changes[field] += 1
    metadata = {'schemaVersion': 1, 'datasetId': output.name,
                'directory': str(output.relative_to(DATA.resolve())),
                'mappingVersion': CORRECTED_MAPPING, 'normalizationVersion': 'arabic-v1',
                'sourceManifestSha256': sha(DATA / 'corpus.json'),
                'sourceDigests': {s['name']: s['sha256'] for s in read_json(DATA / 'corpus.json')['sources'] if 'sha256' in s},
                'baseSplitManifestSha256': sha(DATA / 'split-manifest.json'),
                'baseRoles': {role: split['roles'][role] for role in ('train', 'verification')},
                'heldoutIdentitySha256': sha(DATA / 'heldout-identities.json'),
                'aiInputSha256': sha(DATA / 'ai-review/words.json'),
                'seed': seed, 'candidateLimitPerPOS': candidate_limit,
                'supplementLimitPerPOS': supplement_per_pos, 'supplementWords': len(supplement),
                'excludedTrainingCandidateWords': len(excluded),
                'aiNormalizedIdentities': len(ai_words), 'aiTeacherLemmaIdentities': len(ai_families),
                'verificationSurfaceOrderPreserved': True,
                'verificationWordsChangedByField': dict(changes),
                'coverage': coverage, 'candidateCoverage': candidate_report,
                'rejectedTeacherAnalyses': dict(rejected), 'roles': {},
                'limitations': ['Teacher-generated diagnostic data, not independent human gold',
                    'No final-test labels or AI annotations were opened',
                    'AI inputs and inferred lemma identities are used solely to exclude training data',
                    'Strict surface alignment is retained; this does not recover rejected spelling alternations',
                    'Pattern separators are removed and unsupported/non-Arabic patterns become null',
                    'Root spelling follows the pinned teacher and is not canonicalized in this pilot'],
                **provenance()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.teacher-corrected-', dir=output.parent) as tmp:
        staged = Path(tmp) / output.name
        staged.mkdir()
        for role, rows in roles.items():
            target = staged / f'{role}.json'
            write_json(target, {'schemaVersion': 1, 'role': role, 'records': rows}, exclusive=True)
            metadata['roles'][role] = {'file': target.name, 'sha256': sha(target), 'words': len(rows),
                'analyses': sum(len(row['analyses']) for row in rows),
                'lemmaFamilies': len({value for row in rows for value in families(row)})}
        write_json(staged / 'exclusions.json', {'schemaVersion': 1,
            'blockedWords': sorted(protected['words']), 'blockedLemmaFamilies': sorted(protected['families']),
            'excludedCandidateWords': sorted(row['word'] for row in excluded),
            'aiTeacherLemmaFamilies': sorted(ai_families)}, exclusive=True)
        metadata['exclusionsSha256'] = sha(staged / 'exclusions.json')
        write_json(staged / 'manifest.json', metadata, exclusive=True)
        if output.exists():
            raise ValueError('Refusing to overwrite an existing corrected dataset')
        os.rename(staged, output)
    print(f'CORRECTED_DATASET={output}', flush=True)
    return output


def load_corrected_dataset(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('mappingVersion') != CORRECTED_MAPPING or set(manifest['roles']) != {'train', 'verification'}:
        raise ValueError('Pilot requires corrected train/verification data only')
    exclusions_path = directory / 'exclusions.json'
    if sha(exclusions_path) != manifest['exclusionsSha256']:
        raise ValueError('Corrected exclusion digest mismatch')
    exclusions = read_json(exclusions_path)
    rows = {}
    for role in ('train', 'verification'):
        if manifest['roles'][role].get('file') != f'{role}.json':
            raise ValueError('Unexpected corrected role filename')
        source = directory / f'{role}.json'
        if sha(source) != manifest['roles'][role]['sha256']:
            raise ValueError('Corrected dataset digest mismatch')
        rows[role] = read_json(source)['records']
        if not rows[role]: raise ValueError('Corrected roles must be nonempty')
    blocked_words, blocked_families = set(exclusions['blockedWords']), set(exclusions['blockedLemmaFamilies'])
    blocked_words.update(row['word'] for row in rows['verification'])
    blocked_families.update(value for row in rows['verification'] for value in families(row))
    kept, dropped = exclude_connected(rows['train'], blocked_words, blocked_families)
    if dropped or len(kept) != len(rows['train']):
        raise ValueError('Corrected pilot training overlaps excluded identities')
    return manifest, rows


def pilot(directory, epochs=12, batch_size=128, seed=42, coverage_fraction=0.10, targets_per_pos=64):
    import numpy as np
    import torch
    from .model import Student, make_labels, choose_width, parameter_count, encode_words, encode_analysis, numpy_reference
    from .train import batch_loss, target_batch
    from .export import export_model, parity

    if not 1 <= epochs <= 12 or not 1 <= batch_size <= 256:
        raise ValueError('Pilot is bounded to 1–12 epochs and batch sizes up to 256')
    dataset, rows = load_corrected_dataset(directory)
    train, verification = rows['train'], rows['verification']
    # Validate bounds before creating artifacts or starting optimization.
    epoch_samples(train, random.Random(seed), coverage_fraction, targets_per_pos)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = random.Random(seed)
    labels = make_labels(train)
    width = choose_width(250000, labels)
    model = Student(width, labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-250000-corrected-v1'
    path = RUNS / run_id
    path.mkdir(parents=True, exist_ok=False)
    config = {'schemaVersion': 1, 'experiment': 'corrected-teacher-pilot-v1',
        'targetParameters': 250000, 'actualParameters': parameter_count(model),
        'width': width, 'embedding': 32, 'epochs': epochs, 'batchSize': batch_size,
        'seed': seed, 'optimizer': 'AdamW', 'learningRate': .003, 'weightDecay': .0001,
        'device': 'cpu', 'threads': 4, 'torch': torch.__version__, 'numpy': np.__version__,
        'datasetDirectory': str(Path(directory).resolve()),
        'datasetManifestSha256': sha(Path(directory) / 'manifest.json'),
        'datasetDigests': dataset['roles'], 'mappingVersion': CORRECTED_MAPPING,
        'coverageFractionOfOrdinary': coverage_fraction, 'coverageTargetsPerPOS': targets_per_pos,
        'coveragePOS': list(COVERAGE_POS), 'supervisionCoverage': supervision_coverage(train),
        'selection': 'lowest deterministic sampled corrected-verification cross-entropy',
        'comparisonWarning': 'This loss uses corrected targets; compare model outputs on the same corrected verification records',
        'resume': None, 'aiAnnotationsUsed': False, 'finalTestOpened': False,
        'automaticSelectionOrPromotion': False}
    write_json(path / 'config.json', config, exclusive=True)
    start = time.perf_counter()
    history, best_state = [], None
    best, best_epoch = float('inf'), None
    for epoch in range(epochs):
        model.train()
        samples, sampling = epoch_samples(train, rng, coverage_fraction, targets_per_pos)
        losses = []
        for begin in range(0, len(samples), batch_size):
            batch = samples[begin:begin + batch_size]
            batch_rows = [train[i] for i, _, _ in batch]
            encoded = [encode_analysis(train[i], train[i]['analyses'][a], labels) for i, a, _ in batch]
            targets = {key: torch.as_tensor(np.stack([a[key] for a in encoded]), dtype=torch.long)
                       for key in encoded[0]}
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model(encode_words([r['word'] for r in batch_rows])), targets)
            if not torch.isfinite(loss): raise ValueError('Corrected pilot training diverged')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append((float(loss.detach()), len(batch)))
        model.eval()
        val_rng, val_losses = random.Random(seed), []
        with torch.no_grad():
            for begin in range(0, len(verification), batch_size):
                batch_rows = verification[begin:begin + batch_size]
                loss = batch_loss(model(encode_words([r['word'] for r in batch_rows])),
                                  target_batch(batch_rows, labels, val_rng))
                val_losses.append((float(loss), len(batch_rows)))
        val = sum(loss * n for loss, n in val_losses) / len(verification)
        entry = {'epoch': epoch + 1,
            'trainingLoss': sum(loss * n for loss, n in losses) / len(samples),
            'verificationLoss': val, 'elapsedSeconds': time.perf_counter() - start,
            'sampling': sampling}
        history.append(entry)
        print(__import__('json').dumps({'runId': run_id, **entry}), flush=True)
        if val < best:
            best, best_epoch, best_state = val, epoch + 1, copy.deepcopy(model.state_dict())
        write_json(path / 'history.json', history)
    model.load_state_dict(best_state)
    model.eval()
    torch.save({'state': best_state, 'width': width, 'labels': labels}, path / 'checkpoint.pt')
    manifest, arrays, quant = export_model(model, path, run_id, dataset['roles']['verification']['sha256'])
    words = [r['word'] for r in verification[:128]] + ['ك', 'ك' * 32]
    reference_parity = parity(model, arrays, quant, words)
    write_json(path / 'reference-parity.json', {'schemaVersion': 1, 'model': manifest['id'],
        'data': dataset['roles']['verification'], **provenance(), **reference_parity}, exclusive=True)
    ids = encode_words(words).numpy()
    write_json(path / 'parity-inputs.json', {'words': words,
        'float': {k: v.tolist() for k, v in numpy_reference(ids, arrays).items()},
        'quantized': {k: v.tolist() for k, v in numpy_reference(ids, quant).items()}}, exclusive=True)
    sampled_totals = collections.Counter()
    for entry in history[:best_epoch]: sampled_totals.update(entry['sampling']['sampledPosCounts'])
    write_json(path / 'result.json', {'schemaVersion': 1, 'runId': run_id, 'model': manifest,
        'data': dataset['roles'], 'datasetDirectory': str(Path(directory).resolve()),
        'datasetManifestSha256': config['datasetManifestSha256'],
        'training': {'bestEpoch': best_epoch, 'verificationLoss': best,
                    'elapsedSeconds': time.perf_counter() - start,
                    'selectedCheckpointPosTargets': dict(sampled_totals)},
        'eligibility': {'eligible': False, 'reasons': ['Experimental pilot; never auto-selected',
            'Corrected verification comparison and browser/engineering gates not yet evaluated',
            'Independent human gold unavailable']},
        'finalTestOpened': False, 'aiAnnotationsUsed': False, **provenance()}, exclusive=True)
    print(f'CORRECTED_RUN={path}', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--output', type=Path)
    prepare_parser.add_argument('--seed', type=int, default=42)
    prepare_parser.add_argument('--candidate-limit', type=int, default=128)
    prepare_parser.add_argument('--supplement-per-pos', type=int, default=32)
    train_parser = sub.add_parser('pilot')
    train_parser.add_argument('--dataset', type=Path, required=True)
    train_parser.add_argument('--epochs', type=int, default=12)
    train_parser.add_argument('--batch-size', type=int, default=128)
    train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--coverage-fraction', type=float, default=.10)
    train_parser.add_argument('--targets-per-pos', type=int, default=64)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output, args.seed, args.candidate_limit, args.supplement_per_pos)
    else:
        pilot(args.dataset, args.epochs, args.batch_size, args.seed,
              args.coverage_fraction, args.targets_per_pos)


if __name__ == '__main__':
    main()
