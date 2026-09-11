import copy, unittest
from tinysarf_training.gates import assess
from tinysarf_training.artifacts import ROOT,read_json

class GatesTest(unittest.TestCase):
 def setUp(self):
  self.policy=read_json(ROOT/'packages/training/promotion-policy.json')
  metrics={k:.95 for k in self.policy['floors']}
  self.good={'integrityVerified':True,'cleanReproduction':True,'humanReviewedFixtures':True,'independentGold':True,'goldBeatsBaselines':True,'browserContractPassed':True,'git':{'commit':'a'*40,'dirty':False},'verificationDigest':'b'*64,'normalizationVersion':'arabic-v1','referenceDevice':'test-device','metrics':metrics,'goldMetrics':{n:metrics.copy() for n in ['gold-natural','gold-balanced']},
   'slices':{s:{'count':10,'digest':s,'metrics':metrics.copy()} for s in self.policy['protectedSlices']},
   'parity':{'floatMaxAbsoluteError':.00001,'gpuArgmaxAgreement':1,'deterministicRuns':100,'deterministic':True,'finite':True},
   **{k:10 for k in self.policy['budgets']},'coldRuns':20,'warmups':20,'warmIterations':100,'warmRepeats':3,'measuredBatchAdvantage':True,'warmupStable':True,'fullBrowserProtocol':True,'inputSlices':{s:{'count':10,'accuracy':1} for s in self.policy['protectedInputSlices']}}
 def test_missing_evidence_blocks_first_release(self):
  self.assertFalse(assess({},None,self.policy)['eligible'])
  self.assertTrue(assess(self.good,None,self.policy)['eligible'])
  for field in ['independentGold','cleanReproduction','humanReviewedFixtures']:
   bad=copy.deepcopy(self.good);bad[field]=False;self.assertFalse(assess(bad,None,self.policy)['eligible'])
 def test_aggregate_improvement_cannot_hide_slice_regression(self):
  active=copy.deepcopy(self.good);active['metrics']={k:.90 for k in self.policy['floors']}
  self.assertTrue(assess(self.good,active,self.policy)['eligible'])
  bad=copy.deepcopy(self.good);bad['slices']['weak-root']['metrics']['rootExactMatch']=.1
  self.assertFalse(assess(bad,active,self.policy)['eligible'])
 def test_nan_and_mismatched_data_are_rejected(self):
  bad=copy.deepcopy(self.good);bad['metrics']['rootExactMatch']=float('nan');self.assertFalse(assess(bad,None,self.policy)['eligible'])
  bad=copy.deepcopy(self.good);bad['verificationDigest']='other';self.assertFalse(assess(bad,self.good,self.policy)['eligible'])
