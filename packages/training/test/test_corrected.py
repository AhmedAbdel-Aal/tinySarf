import collections
import copy
import random
import tempfile
import unittest
from pathlib import Path

from tinysarf_training.artifacts import sha, write_json
from tinysarf_training.contract import (
    CORRECTED_MAPPING, LEGACY_MAPPING, map_analysis, mapped_pattern, segment_type)
from tinysarf_training.corrected import (
    COVERAGE_POS, epoch_samples, exclude_connected, load_legacy_roles,
    supervision_coverage, choose_supplement)


def row(word, *pos, family=None):
    return {'word': word, 'analyses': [{'pos': p, 'lemmaFamily': family or word}
                                     for p in pos]}


class CorrectedMappingTest(unittest.TestCase):
    def test_legacy_default_is_the_frozen_mapping(self):
        raw = {'bw': 'فَ/CONNEC_PART+بِ/PREP+ال/DET+قَلَم/NOUN',
               'root': 'ق.ل.م', 'pos': 'noun', 'lex': 'قَلَم_1',
               'pattern': 'NTWS', 'pattern_abstract': 'ٌطWص+ِيّ'}
        expected = {
            'spans': [{'type': 'stem', 'start': 0, 'end': 1},
                      {'type': 'preposition', 'start': 1, 'end': 2},
                      {'type': 'article', 'start': 2, 'end': 4},
                      {'type': 'stem', 'start': 4, 'end': 7}],
            'root': 'قلم', 'pattern': 'ٌطWص+ِيّ', 'pos': 'noun',
            'features': {key: '__missing__' for key in
                         ('person', 'gender', 'number', 'aspect', 'mood', 'voice', 'case', 'state')},
            'lemmaFamily': 'قلم'}
        self.assertEqual(map_analysis('فبالقلم', raw), expected)
        self.assertEqual(map_analysis('فبالقلم', raw, LEGACY_MAPPING), expected)
        corrected = map_analysis('فبالقلم', raw, CORRECTED_MAPPING)
        self.assertEqual(corrected['spans'][0]['type'], 'particle')
        self.assertIsNone(corrected['pattern'])
        self.assertEqual(corrected['root'], expected['root'])

    def test_corrected_pattern_rejects_corrupted_sentinels_and_non_arabic(self):
        for pattern in ('NTWS', 'NOAN', 'na'):
            raw = {'pattern': pattern, 'pattern_abstract': 'ٌطWص+ِيّ'}
            self.assertIsNone(mapped_pattern(raw, CORRECTED_MAPPING))
        for pattern in ('ٌطWص+ِيّ', 'فعل5', 'فاعِل😀', 'Latin', 'َُ'):
            self.assertIsNone(mapped_pattern({'pattern_abstract': pattern}, CORRECTED_MAPPING))
        self.assertEqual(mapped_pattern({'pattern_abstract': '1َعْ2+ِيّ'}, CORRECTED_MAPPING), 'فَعْعِيّ')
        self.assertEqual(mapped_pattern({'pattern_abstract': 'فَعْل+ِيّ'}, CORRECTED_MAPPING), 'فَعْلِيّ')
        self.assertEqual(mapped_pattern({'pattern_abstract': 'ٱِفْتِعال'}, CORRECTED_MAPPING), 'ٱِفْتِعال')
        self.assertEqual(mapped_pattern({'pattern_abstract': 'فَعْل+ِيّ'}), 'فَعْل+ِيّ')

    def test_only_requested_mapping_version_enables_connective_particles(self):
        for tag in ('CONNEC_PART', 'RC_PART'):
            self.assertEqual(segment_type(tag), 'stem')
            self.assertEqual(segment_type(tag, CORRECTED_MAPPING), 'particle')
        with self.assertRaisesRegex(ValueError, 'version'):
            map_analysis('كتب', {'bw': 'كتب/VERB'}, version='future-version')


class CorrectedIsolationTest(unittest.TestCase):
    def test_transitive_ambiguity_and_unmappable_aliases_cannot_enter_training(self):
        first = row('كتب', 'noun', family='كتاب')
        first['teacherLemmaFamilies'] = ['كتاب', 'محظور']
        second = row('كتاب', 'noun', family='كتاب')
        safe = row('ذهب', 'verb')
        kept, dropped = exclude_connected([first, second, safe], set(), {'محظور'})
        self.assertEqual([r['word'] for r in kept], ['ذهب'])
        self.assertEqual([r['word'] for r in dropped], ['كتب', 'كتاب'])
        kept, _ = exclude_connected([first, second, safe], {'كتب'}, set())
        self.assertEqual([r['word'] for r in kept], ['ذهب'])

    def test_loader_reads_only_train_and_verification_even_when_final_test_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / 'generated/teacher-sampled-v2'
            data.mkdir(parents=True)
            roles = {}
            for role in ('train', 'verification'):
                file = data / f'{role}.json'
                write_json(file, {'records': [row(role, 'noun')]})
                roles[role] = {'sha256': sha(file)}
            roles['final-test'] = {'sha256': 'sealed-not-accessed'}
            write_json(base / 'split-manifest.json', {'datasetId': 'teacher-sampled-v2',
                'directory': 'generated/teacher-sampled-v2', 'roles': roles})
            _, loaded = load_legacy_roles(base)
            self.assertEqual(set(loaded), {'train', 'verification'})
            self.assertFalse((data / 'final-test.json').exists())
            (data / 'train.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_legacy_roles(base)


class CoverageSamplerTest(unittest.TestCase):
    def test_coverage_draws_are_bounded_reproducible_and_correctly_labelled(self):
        rows = [row(str(i), 'noun') for i in range(200)]
        for i, pos in enumerate(COVERAGE_POS):
            rows[i]['analyses'].append({'pos': pos, 'lemmaFamily': str(i)})
        before = copy.deepcopy(rows)
        first, report = epoch_samples(rows, random.Random(42), targets_per_pos=64)
        second, again = epoch_samples(rows, random.Random(42), targets_per_pos=64)
        self.assertEqual((first, report), (second, again))
        self.assertEqual(rows, before)
        self.assertEqual(report['ordinarySamples'], 200)
        self.assertEqual(report['coverageSamples'], 20)
        self.assertGreaterEqual(report['ordinaryFraction'], .9)
        self.assertEqual(set(report['forcedPosCounts']), set(COVERAGE_POS))
        forced = collections.Counter(rows[i]['analyses'][a]['pos'] for i, a, coverage in first if coverage)
        self.assertEqual(dict(forced), report['forcedPosCounts'])
        for fraction in (-.01, .11):
            with self.assertRaises(ValueError): epoch_samples(rows, random.Random(1), fraction)

    def test_empty_or_absent_strata_do_not_invent_supervision(self):
        samples, report = epoch_samples([row('كتب', 'noun')], random.Random(1))
        self.assertEqual(len(samples), 1)
        self.assertEqual(report['coverageSamples'], 0)
        self.assertEqual(report['unsupportedCoveragePOS'], list(COVERAGE_POS))
        self.assertEqual(epoch_samples([], random.Random(1))[0], [])

    def test_coverage_measures_ambiguity_and_supplement_is_bounded(self):
        rows = [row('كتب', 'noun', 'verb'), row('لن', 'particle', 'verb')]
        measured = supervision_coverage(rows)
        self.assertEqual(measured['particle']['wordSupport'], 1)
        self.assertEqual(measured['particle']['expectedOrdinaryTargetsPerEpoch'], .5)
        more = rows + [row(f'particle-{i}', 'particle') for i in range(100)]
        supplement = choose_supplement(more, {'كتب', 'لن'}, per_pos=3)
        self.assertEqual(len(supplement), 3)
        self.assertTrue(all(r['word'].startswith('particle-') for r in supplement))


if __name__ == '__main__':
    unittest.main()
