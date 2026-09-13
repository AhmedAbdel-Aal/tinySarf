"""Learn reusable root transformations from teacher-aligned stems in training only."""
import argparse
from pathlib import Path

from .artifacts import read_json, write_json, sha, provenance
from .contract import map_analysis
from .corrected import pinned_teacher, exclude_connected
from .root_data import load
from .root_templates import learn


def run(directory, output):
    directory, output = Path(directory), Path(output)
    if output.exists(): raise ValueError('Refusing to overwrite a stem experiment')
    manifest, roles = load(directory)
    exclusions = read_json(directory / 'exclusions.json')
    blocked_words = set(exclusions['blockedWords'])
    rows = {}
    with pinned_teacher() as (_, analyzer):
        for i, row in enumerate(roles['train']):
            for raw in analyzer.analyze(row['word']):
                try: a = map_analysis(row['word'], raw)
                except ValueError: continue
                if not a['root']: continue
                for span in a['spans']:
                    if span['type'] != 'stem': continue
                    stem = row['word'][span['start']:span['end']]
                    if stem in blocked_words: continue
                    entry = rows.setdefault(stem, {'word': stem, 'analyses': {}})
                    entry['analyses'][(a['root'], a['lemmaFamily'])] = {'root': a['root'], 'lemmaFamily': a['lemmaFamily']}
            if i % 5000 == 0: print(f'Root stem induction: {i}/{len(roles["train"])}', flush=True)
    records = [{**row, 'analyses': list(row['analyses'].values())} for row in rows.values()]
    records, dropped = exclude_connected(records, blocked_words, set(exclusions['blockedLemmaFamilies']))
    templates = learn(records)
    output.mkdir(parents=True)
    write_json(output / 'training-stems.json', records)
    write_json(output / 'templates.json', templates)
    write_json(output / 'manifest.json', {'sourceManifestSha256': sha(directory / 'manifest.json'),
        'templatesSha256': sha(output / 'templates.json'), 'trainingStemsSha256': sha(output / 'training-stems.json'),
        'trainingStems': len(records), 'excludedConnectedStems': len(dropped),
        'templates': len(templates), 'minimumDistinctRoots': 3, 'finalTestOpened': False, **provenance()})
    print(f'STEM_TEMPLATES={output}; stems={len(records)} templates={len(templates)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.dataset, args.output)
