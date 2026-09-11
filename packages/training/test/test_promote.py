import tempfile,unittest
from pathlib import Path
from tinysarf_training.promote import atomic_activate,REQUIRED_FILES
from tinysarf_training.artifacts import write_json,read_json,sha
class PromoteTest(unittest.TestCase):
 def test_atomic_pointer_and_immutable_checkpoint(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);candidate=root/'candidate';candidate.mkdir();active=root/'active'
   for file in REQUIRED_FILES:(candidate/file).write_bytes(b'fixture')
   write_json(candidate/'manifest.json',{'id':'unit-test'})
   report=root/'report.json';write_json(report,{'eligible':True})
   destination=atomic_activate(candidate,report,active,root)
   pointer=read_json(active/'current.json');self.assertEqual(root/pointer['directory'],destination)
   self.assertEqual(pointer['manifestSha256'],sha(candidate/'manifest.json'))
   self.assertEqual((destination/'weights.bin').read_bytes(),b'fixture')
   original=(active/'current.json').read_bytes()
   with self.assertRaises(ValueError):atomic_activate(candidate,report,active,root)
   self.assertEqual((active/'current.json').read_bytes(),original)
