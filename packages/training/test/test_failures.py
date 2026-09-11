import copy,unittest
from tinysarf_training.failures import validate_failure
from tinysarf_training.contract import normalize
class FailureTest(unittest.TestCase):
 def setUp(self):
  self.entry={'schemaVersion':1,'source':{'role':'mining','license':'MIT'},'review':{'status':'approved','independentHuman':True,'reviewer':'unit-test fixture','reviewedAt':'2026-01-01'},'record':{'word':'كتب','analyses':[{'spans':[{'type':'stem','start':0,'end':3}],'root':'كتب','pattern':None,'pos':'verb','features':{},'lemmaFamily':'كتب'}]}}
  self.empty={'words':set(),'families':set()}
 def test_unreviewed_and_heldout_are_rejected(self):
  self.assertEqual(validate_failure(self.entry,self.empty)['word'],'كتب')
  for h in [{'words':{'كتب'},'families':set()},{'words':set(),'families':{'كتب'}}]:
   with self.assertRaises(ValueError):validate_failure(self.entry,h)
  bad=copy.deepcopy(self.entry);bad['review']['independentHuman']=False
  with self.assertRaises(ValueError):validate_failure(bad,self.empty)
  bad=copy.deepcopy(self.entry);bad['source']['role']='verification'
  with self.assertRaises(ValueError):validate_failure(bad,self.empty)
 def test_python_matches_input_rejections(self):
  self.assertEqual(normalize('كِتَاب'),'كتاب')
  for word in ['','َكتب','ـكتب','a','😀','كتب جيد','ك'*33,'ك'+'َ'*512]:
   with self.assertRaises(ValueError):normalize(word)
