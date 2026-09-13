import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tinysarf_training import root_data


class RootPreparationTest(unittest.TestCase):
    def test_cold_preparation_isolates_each_pinned_teacher_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = root / 'packages/training/data'
            def write(file, value):
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(json.dumps(value))
            pools = []
            for version, limit in [(1, 60000), (2, 120000)]:
                name = f'teacher-roots-v{version}-{limit}'
                digest = hashlib.sha256(str(limit).encode()).hexdigest()
                pools.append({'directory': f'packages/training/data/generated/{name}', 'sha256': digest})
                write(data / 'audits' / (name + '-manifest.json'), {'candidateLimit': limit, 'seed': 43})
            write(root / 'packages/training/candidate.json', {'directory': 'candidate'})
            write(root / 'candidate/config.json', {'rootTraining': {'datasetDirectory': pools[0]['directory'],
                  'dataset': {'train': {'sha256': pools[0]['sha256']}}}, 'rootTrainingPools': pools})
            calls = []
            def run(command, *, cwd, check):
                self.assertEqual(cwd, root)
                self.assertTrue(check)
                calls.append(command)
                if command[2] == 'tinysarf_training.corrected':
                    write(data / 'generated/teacher-corrected-pilot-v1/manifest.json', {})
                else:
                    limit = command[command.index('--limit') + 1]
                    version = 2 if '--paradigms' in command else 1
                    target = data / 'generated' / f'teacher-roots-v{version}-{limit}' / 'train.json'
                    target.parent.mkdir(parents=True)
                    target.write_text(limit)
            with patch.object(root_data, 'ROOT', root), patch.object(root_data, 'DATA', data), \
                 patch.object(root_data.subprocess, 'run', side_effect=run), \
                 patch.object(root_data, 'load') as validate, \
                 patch.object(root_data, 'prepare', side_effect=AssertionError('Teacher reused in caller')):
                root_data.prepare_selected()
                self.assertEqual(validate.call_count, 2)
                self.assertEqual([c[2] for c in calls], ['tinysarf_training.corrected',
                    'tinysarf_training.root_data', 'tinysarf_training.root_data'])
                # Once reconstructed, verification must not recompile anything.
                root_data.prepare_selected()
                self.assertEqual(len(calls), 3)


if __name__ == '__main__': unittest.main()
