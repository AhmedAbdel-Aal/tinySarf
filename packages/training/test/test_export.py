import tempfile,unittest
from pathlib import Path
import numpy as np
from tinysarf_training.model import Student,make_labels,export_float
from tinysarf_training.export import export_model
from tinysarf_training.artifacts import sha
class ExportTest(unittest.TestCase):
 def test_packed_ranges_and_reconstruction(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);m=Student(16,make_labels([]));manifest,arrays,quant=export_model(m,p,'test','a'*64)
   self.assertEqual(sha(p/'weights.bin'),manifest['sha256'])
   signed=np.frombuffer((p/'weights.bin').read_bytes(),dtype=np.int8);end=0
   for name,s in manifest['tensors'].items():
    self.assertEqual(s['offset'],end);self.assertEqual(s['offset']%4,0)
    values=signed[s['offset']:s['offset']+s['length']].astype(np.float32)*np.float32(s['scale'])
    np.testing.assert_array_equal(values,quant[name].reshape(-1))
    self.assertLessEqual(float(np.max(np.abs(arrays[name]-quant[name]))),s['scale']/2+1e-6)
    end=s['offset']+((s['length']+3)//4)*4
   self.assertEqual(end,len(signed))
