import test from 'node:test';
import assert from 'node:assert/strict';
import type {ModelManifest} from '../src/model';
import {LETTERS} from '../src/normalize';
import {constrainRoots} from '../src/root-decode';

function scores(root:string) {
  return Object.fromEntries([0,1,2,3].map(slot=>{
    const values=Array(LETTERS.length+1).fill(-12);values[slot<root.length?LETTERS.indexOf(root[slot])+1:0]=-.1;
    return [`root${slot}`,values];
  }));
}
function manifest():ModelManifest {
  return {vocabulary:['__pad__',...LETTERS],rootDecoder:{format:'learned-transforms-v1',minimumDistinctRoots:3,priorWeight:.25,unsupportedPenalty:40,nullPenalty:0,
    whole:[],stem:[{surface:'**ا*',output:[0,1,3],distinctRoots:20}]}} as unknown as ModelManifest;
}
test('learned root transformations use the predicted stem and restore input radicals',()=>{
  const values=scores('كبب');values.root1[LETTERS.indexOf('ت')+1]=-2;
  const result=constrainRoots('وبكتابهم',[0,2,4,4,4,4,7,7],values,manifest(),[{root:'كبب',score:-.4}]);
  assert.equal(result[0].root,'كتب');
});
test('a supported neural root keeps its ranking and an unmatched word retains a fallback',()=>{
  const values=scores('كتب'),m=manifest();
  const result=constrainRoots('كتاب',[4,4,4,4],values,m,[{root:'كتب',score:-.4}]);
  assert.equal(result[0].root,'كتب');assert.equal(result[0].score,-.4);
  const fallback=constrainRoots('كتب',[4,4,4],values,m,[{root:'كتب',score:-.4}]);
  assert.ok(fallback.some(r=>r.root==='كتب'));
});
test('legacy manifests preserve legacy root candidates exactly',()=>{
  const original=[{root:'كتب',score:-3}];
  assert.equal(constrainRoots('كتاب',[4,4,4,4],scores('كتب'),{} as ModelManifest,original),original);
});
