import unittest
import numpy as np
import torch
from tinysarf_training.model import encode_words
from tinysarf_training.root_model import RootModel, export_root, numpy_root
from tinysarf_training.root_experiment import root_loss, targets_for
from tinysarf_training.root_templates import transformations, learn, proposals


class RootTests(unittest.TestCase):
    def test_complete_root_marginal_does_not_reward_incompatible_radical_mixtures(self):
        probabilities = torch.tensor([[[.01, .79, .20], [.01, .70, .29], [.01, .01, .98], [.98, .01, .01]]])
        targets = torch.tensor([[[1, 2, 2, 0], [2, 1, 2, 0]]])
        valid = torch.tensor([[True, True]])
        expected = -np.log((.79 * .29 + .20 * .70) * .98 * .98)
        self.assertAlmostEqual(float(root_loss(probabilities.log(), targets, valid)), expected, places=6)

    def test_duplicate_analyses_do_not_multiply_root_probability(self):
        rows = [{'analyses': [{'root': 'كتب'}, {'root': 'كتب'}, {'root': 'كبت'}]}]
        targets, valid = targets_for(rows)
        self.assertEqual(targets.shape, (1, 2, 4))
        self.assertTrue(valid.all())

    def test_numpy_parity_masked_context_and_probability_mass(self):
        torch.manual_seed(83)
        model = RootModel().eval()
        words = ['ك', 'كتب', 'كتاب', 'وبكتابهم', 'ك' * 32]
        ids = encode_words(words)
        with torch.no_grad(): expected, trace = model(ids, trace=True)
        actual = numpy_root(ids.numpy(), export_root(model))
        for name, tensor in trace.items(): np.testing.assert_allclose(actual[name], tensor.numpy(), atol=1e-5, rtol=1e-5)
        for slot in range(4): np.testing.assert_allclose(actual[f'root{slot}'], expected[:, slot].numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(expected.exp().sum(-1).numpy(), 1, atol=1e-6)
        with torch.no_grad(): short = model(encode_words(words[:4], length=8))
        np.testing.assert_allclose(short.numpy(), expected[:4].numpy(), atol=1e-5, rtol=1e-5)

    def test_ordered_copies_doubled_and_restored_radicals(self):
        self.assertIn(('**ا*', (0, 1, 3)), transformations('كتاب', 'كتب'))
        self.assertIn(('**', (0, 1, 1)), transformations('مد', 'مدد'))
        self.assertIn(('*ا*', (0, 'و', 2)), transformations('قال', 'قول'))
        self.assertNotIn(('***', (2, 1, 0)), transformations('كتب', 'بتك'))

    def test_transformations_require_cross_root_support(self):
        rows = [{'word': word, 'analyses': [{'root': root}]} for word, root in [('كتاب', 'كتب'), ('حساب', 'حسب'), ('شراب', 'شرب')]]
        rules = learn(rows)
        self.assertIn('ضرب', proposals('ضراب', rules))
        self.assertEqual(learn(rows[:1]), [])


if __name__ == '__main__': unittest.main()
