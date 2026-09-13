"""Diagnostic: learn surface-to-radical transformations without a word/root dictionary."""
import argparse
import collections
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from .artifacts import DATA, ROOT, read_json, write_json, sha, provenance
from .contract import LETTERS
from .root_experiment import load_base, encode_words, roots_for, report, decode_roots, RootPointer, CNNRootPointer, StandaloneRootPointer
from .root_data import load


def transformations(word, root):
    """Enumerate ordered copies; a missing radical can be generated literally.

    Each retained transformation must copy at least two distinct positions and
    generalize across multiple different training roots before it is exported.
    """
    results = []
    def visit(slot, previous, output):
        if slot == len(root):
            copied = {value for value in output if isinstance(value, int)}
            if len(copied) < 2: return
            mask = ''.join('*' if i in copied else c for i, c in enumerate(word))
            results.append((mask, tuple(output)))
            return
        positions = [i for i, c in enumerate(word) if c == root[slot] and
                     (i > previous or i == previous and slot and root[slot] == root[slot - 1])]
        for position in positions: visit(slot + 1, position, output + [position])
        if not any(isinstance(value, str) for value in output):
            visit(slot + 1, previous, output + [root[slot]])
    visit(0, -1, [])
    if not results: return []
    # Prefer complete copies; missing-radical rules are learned when the surface
    # really needs them, not as arbitrary shortcuts past visible consonants.
    best = min(sum(isinstance(v, str) for v in out) for _, out in results)
    return list({item for item in results if sum(isinstance(v, str) for v in item[1]) == best})


def learn(rows, minimum_roots=3):
    support = collections.defaultdict(set)
    for row in rows:
        for root in roots_for([row])[0]:
            if root:
                for template in transformations(row['word'], root): support[template].add(root)
    return [{'surface': mask, 'output': list(output), 'distinctRoots': len(roots)}
            for (mask, output), roots in sorted(support.items(), key=lambda entry: repr(entry[0]))
            if len(roots) >= minimum_roots]


def proposals(word, templates):
    scores = {}
    for template in templates:
        mask = template['surface']
        if len(word) != len(mask) or any(c != '*' and c != word[i] for i, c in enumerate(mask)): continue
        root = ''.join(word[value] if isinstance(value, int) else value for value in template['output'])
        scores[root] = scores.get(root, 0) + template['distinctRoots']
    total = sum(scores.values())
    return {root: math.log(value / total) for root, value in scores.items()}


def run(directory, output, minimum_roots=3, root_run=None, stem_templates=None):
    torch.set_num_threads(4)
    dataset, roles = load(directory)
    templates = learn(roles['train'], minimum_roots)
    grouped = collections.defaultdict(list)
    for template in templates: grouped[len(template['surface'])].append(template)
    base, manifest, selected = load_base()
    root_model = None
    if root_run:
        config = read_json(Path(root_run) / 'config.json')
        cls = {'pointer': RootPointer, 'cnn-pointer': CNNRootPointer, 'standalone': StandaloneRootPointer}[config['mode']]
        root_model = cls(base)
        root_model.load_state_dict(torch.load(Path(root_run) / 'root-head.pt', weights_only=True)['state'])
        root_model.eval()
    with torch.no_grad():
        log_batches = []
        for batch in encode_words([r['word'] for r in roles['verification']]).split(128):
            values = base(batch, trace=True)
            log_batches.append(root_model(values['conv2'], values['pooled'], batch) if root_model else torch.stack([values[f'root{i}'].log_softmax(-1) for i in range(4)], 1))
        logp = torch.cat(log_batches)
    stem_grouped = collections.defaultdict(list)
    predicted_stems = {}
    if stem_templates:
        for template in read_json(Path(stem_templates) / 'templates.json'): stem_grouped[len(template['surface'])].append(template)
        baseline_file = ROOT / 'packages/benchmark/results/correctness-2026-09-11T212223821Z.json'
        artifact = read_json(baseline_file)
        if artifact['model']['sha256'] != manifest['sha256']: raise ValueError('Segmentation artifact does not match base model')
        for row in artifact['predictions']:
            top = row['analyses'][0]
            predicted_stems[row['word']] = [row['word'][s['start']:s['end']] for s in top['spans'] if s['type'] == 'stem']
    rows = roles['verification']
    allowed = roots_for(rows)
    old_roots = decode_roots(logp)
    old_scores = []
    candidate_scores = []
    for i, row in enumerate(rows):
        def neural(root): return sum(float(logp[i, slot, LETTERS.index(root[slot]) + 1 if slot < len(root) else 0]) for slot in range(4))
        old_scores.append(neural(old_roots[i]))
        options = proposals(row['word'], grouped[len(row['word'])])
        if stem_templates:
            combined = {root: math.exp(prior) for root, prior in options.items()}
            for stem in predicted_stems[row['word']]:
                for root, prior in proposals(stem, stem_grouped[len(stem)]).items(): combined[root] = combined.get(root, 0) + math.exp(prior)
            total = sum(combined.values())
            options = {root: math.log(value / total) for root, value in combined.items()}
        options[''] = 0.0
        candidate_scores.append({root: (neural(root), prior) for root, prior in options.items()})
    training_roots = set(root for choices in roots_for(roles['train']) for root in choices)
    grid = []
    for weight, penalty, null_penalty in itertools.product([0, .25, .5, 1], [0, 2, 4, 8, 16, 1000], [0, 1, 2, 4]):
        predictions = []
        for i, candidates in enumerate(candidate_scores):
            ranked = {root: likelihood + weight * prior for root, (likelihood, prior) in candidates.items()}
            old = old_roots[i]
            ranked[old] = max(ranked.get(old, -math.inf), old_scores[i] - (penalty if old not in candidates else 0))
            ranked[''] -= null_penalty
            predictions.append(max(ranked, key=ranked.get))
        metrics = report(rows, predictions, training_roots)
        grid.append({'priorWeight': weight, 'unsupportedPenalty': penalty, 'nullPenalty': null_penalty, 'metrics': metrics, 'predictions': predictions})
    best = max(grid, key=lambda entry: entry['metrics']['all']['exactMatch'])
    coverage = sum(any(root in candidates for root in choices) for candidates, choices in zip(candidate_scores, allowed))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'templates.json', templates)
    write_json(output / 'grid.json', [{k: v for k, v in entry.items() if k != 'predictions'} for entry in grid])
    write_json(output / 'predictions.json', [{'word': row['word'], 'root': root or None} for row, root in zip(rows, best['predictions'])])
    result = {'templateCount': len(templates), 'minimumDistinctRoots': minimum_roots,
              'candidateCoverage': {'correct': coverage, 'count': len(rows)},
              'baseline': report(rows, old_roots, training_roots),
              'best': {k: v for k, v in best.items() if k != 'predictions'},
              'data': dataset['roles'], 'sourceDatasetDirectory': str(directory), 'sourceDatasetSha256': sha(Path(directory) / 'manifest.json'),
              'baseModel': manifest['id'], 'rootRun': root_run, 'stemTemplates': stem_templates,
              'finalTestOpened': False, 'wordDictionary': False, **provenance()}
    write_json(output / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--minimum-roots', type=int, default=3)
    parser.add_argument('--root-run')
    parser.add_argument('--stem-templates')
    args = parser.parse_args()
    run(args.dataset, args.output, args.minimum_roots, args.root_run, args.stem_templates)
