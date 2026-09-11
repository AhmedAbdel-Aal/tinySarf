import unittest
from tinysarf_training.contract import map_analysis, internal_feature
from tinysarf_training.splits import split_records, validate_splits

class DataTest(unittest.TestCase):
 def test_mapping_and_missing_states(self):
  a=map_analysis('وبكتابهم',{'bw':'وَ/CONJ+بِ/PREP+كِتاب/NOUN+هُم/POSS_PRON_3MP','root':'ك.ت.ب','lex':'كِتاب_1','pos':'noun','gen':'m','num':'s','stt':'c'})
  self.assertEqual(a['root'],'كتب'); self.assertEqual(a['lemmaFamily'],'كتاب')
  self.assertEqual([s['end'] for s in a['spans']],[1,2,6,8])
  self.assertEqual(internal_feature({},'gen'),'__missing__')
  self.assertEqual(internal_feature({'gen':'na'},'gen'),'__na__')
  self.assertEqual(internal_feature({'gen':'x'},'gen'),'__unknown__')
  with self.assertRaises(ValueError): map_analysis('كتاب',{'bw':'بَيْت/NOUN'})
 def test_ambiguity_connects_families(self):
  rows=[{'word':'كتب','analyses':[{'lemmaFamily':'كتاب'},{'lemmaFamily':'كتب'}]}, {'word':'يكتب','analyses':[{'lemmaFamily':'كتب'}]}, {'word':'الكتاب','analyses':[{'lemmaFamily':'كتاب'}]}]
  split=split_records(rows,['كتاب'])
  self.assertEqual(len(split['verification']),3)
  self.assertEqual(split,split_records(list(reversed(rows)),['كتاب']))
 def test_detect_leakage(self):
  with self.assertRaises(ValueError): validate_splits({'train':[{'word':'كتب','analyses':[{'lemmaFamily':'كتب'}]}], 'verification':[{'word':'يكتب','analyses':[{'lemmaFamily':'كتب'}]}]})
