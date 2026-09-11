import unittest
import numpy as np
import torch
from tinysarf_training.model import Student, make_labels, export_float, numpy_reference, encode_words

class ModelTest(unittest.TestCase):
 def test_numpy_matches_pytorch_and_padding_does_not_change_outputs(self):
  torch.manual_seed(42); labels=make_labels([]); m=Student(16,labels).eval()
  words=['كتب','وبكتابهم']; a=export_float(m)
  with torch.no_grad(): expected=m(encode_words(words),trace=True)
  actual=numpy_reference(encode_words(words).numpy(),a,trace=True)
  for key in expected: np.testing.assert_allclose(expected[key].numpy(),actual[key],atol=1e-5,rtol=1e-5)
  short=numpy_reference(encode_words(words,8).numpy(),a)
  for key in short:
   if key=='segmentation': np.testing.assert_allclose(short[key],actual[key][:,:8],atol=1e-5)
   else: np.testing.assert_allclose(short[key],actual[key],atol=1e-5)
 def test_all_shapes_are_finite(self):
  m=Student(16,make_labels([])).eval()
  with torch.no_grad():
   for words in [['ك'],['ك'*32],['كتب']*17]:
    for out in m(encode_words(words)).values(): self.assertTrue(torch.isfinite(out).all())
