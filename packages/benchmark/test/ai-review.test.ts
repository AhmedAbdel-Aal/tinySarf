import test from 'node:test';
import assert from 'node:assert/strict';
import {validateAIReview,fieldMatches,type AIReview} from '../src/ai-review';
const sample=():AIReview=>({schemaVersion:1,name:'ai-reviewed-challenge-v1',labelOrigin:'ai-reviewed',humanReviewed:false,independentOfModelAndTeacher:true,annotation:{annotators:['ai-a','ai-b'],adjudicator:'ai-c',completedAt:'2026-09-11',inputSha256:'hash',annotatorFiles:[{file:'a',sha256:'a'},{file:'b',sha256:'b'}]},records:[{word:'قرأ',roots:['قرء'],pos:['verb'],segmentations:[[{type:'stem',start:0,end:3}]],confidence:'high',notes:'hamza canonicalization'}],limitations:[]});
test('AI review stays separate from human gold and requires independent review evidence',()=>{
 const d=sample();validateAIReview(d,['قرأ']);
 assert.throws(()=>validateAIReview({...d,humanReviewed:true} as any,['قرأ']),/human gold/);
 assert.throws(()=>validateAIReview({...d,annotation:{...d.annotation,adjudicator:'ai-a'}},['قرأ']),/adjudicator/);
 assert.throws(()=>validateAIReview({...d,annotation:{...d.annotation,annotatorFiles:[d.annotation.annotatorFiles[0],d.annotation.annotatorFiles[0]]}},['قرأ']),/Two blind/);
 assert.throws(()=>validateAIReview(d,['كتب']),/frozen word list/);
 const malformed=sample();malformed.records[0]={...malformed.records[0],word:'كَتَبَ',segmentations:[[{type:'stem',start:0,end:1},{type:'inflectional_suffix',start:1,end:6}]]};
 assert.throws(()=>validateAIReview(malformed,['كَتَبَ']),/attached mark/);
});
test('field agreement preserves exact radicals and separates canonical hamza, POS and span alternatives',()=>{
 const r=sample().records[0],matches=fieldMatches(r,[{root:'قرأ',pos:'noun',spans:r.segmentations[0],pattern:null,features:{},score:0}]);
 assert.deepEqual(matches,{rootExact:[false],rootCanonical:[true],pos:[false],segmentation:[true]});
});
