import io,tarfile,tempfile,unittest
from pathlib import Path
from tinysarf_training.data import extract_teacher
class ArchiveTest(unittest.TestCase):
 def archive(self,path,name,link=False):
  with tarfile.open(path,'w:gz') as tar:
   info=tarfile.TarInfo(name)
   if link:info.type=tarfile.SYMTYPE;info.linkname='/tmp/outside';tar.addfile(info)
   else:info.size=4;tar.addfile(info,io.BytesIO(b'test'))
 def test_portable_extraction_and_traversal_rejection(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);archive=root/'teacher.tar.gz';dest=root/'teacher'
   self.archive(archive,'root/camel_tools/morphology/database.py');extract_teacher(archive,dest)
   self.assertEqual((dest/'camel_tools/morphology/database.py').read_bytes(),b'test')
   for name,link in [('root/../../outside',False),('/absolute',False),('root/link',True)]:
    self.archive(archive,name,link)
    with self.assertRaises(ValueError):extract_teacher(archive,dest)
    self.assertEqual((dest/'camel_tools/morphology/database.py').read_bytes(),b'test')
