import assert from "node:assert/strict";
import {test} from "node:test";
import {evaluate} from "../src/metrics";
import {stats,pairedBootstrap,bootstrap} from "../src/math";
import type {MorphAnalysis} from "../../core/src/schema";
const a:MorphAnalysis={spans:[{type:"stem",start:0,end:3}],root:"كتب",pattern:"فعل",pos:"verb",features:{person:"third"},score:-.1};
test("accepts alternate valid analyses and reports counts",()=>{
 const b={...a,pos:"noun"}; const r=evaluate([{word:"كتب",analyses:[a,b]}],[[b]],{bootstrap:false});
 assert.equal(r.fullAnalysisTop1.value,1); assert.equal(r.rootExactMatch.numerator,1); assert.equal(r.boundary.f1,1); assert.equal(r.pos.macroF1,1);
});
test("top3 does not conceal poor top1; abstention risk denominator is covered words",()=>{
 const b={...a,root:"دخل",score:-10};
 const r=evaluate([{word:"كتب",analyses:[a]},{word:"كتب",analyses:[a]}],[[b,a],[a]],{threshold:-1,bootstrap:false});
 assert.equal(r.fullAnalysisTop1.value,.5); assert.equal(r.fullAnalysisTop3.value,1);
 assert.equal(r.coverage.value,.5); assert.equal(r.risk.value,0); assert.equal(r.risk.denominator,1);
});
test("known boundary precision/recall and partial root",()=>{
 const g={...a,spans:[{type:"conjunction" as const,start:0,end:1},{type:"stem" as const,start:1,end:3}]};
 const r=evaluate([{word:"كتب",analyses:[g]}],[[{...a,root:"كتف"}]],{bootstrap:false});
 assert.equal(r.boundary.recall,0); assert.equal(r.rootCharacterAccuracy.value,2/3); assert.equal(r.segmentationExactMatch.value,0);
});
test("unmeasured empty data is null, not measured zero",()=>{
 const r=evaluate([],[],{bootstrap:false}); assert.equal(r.rootExactMatch.value,null); assert.equal(r.risk.value,null);
});
test("bootstrap is deterministic and paired differences use the same words",()=>{
 assert.deepEqual(bootstrap([0,1,1],42,100),bootstrap([0,1,1],42,100));
 const r=pairedBootstrap([1,0,1],[1,0,1],42,100); assert.equal(r.difference,0); assert.equal(r.ci95?.high,0);
 assert.throws(()=>pairedBootstrap([1],[])); assert.equal(stats([1,2,3,4])?.p50,2.5);
});
