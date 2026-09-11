import assert from "node:assert/strict";
import {test} from "node:test";
import {validateArtifact} from "../src/artifact";
test("release artifacts require provenance and finite measured values",()=>{
 const a={schemaVersion:1,runId:"test",createdAt:"2026-09-11T00:00:00Z",commands:[],git:{commit:null,dirty:null},environment:{browser:null},model:{id:null},data:{goldDigest:null},correctness:null};
 validateArtifact(a); assert.throws(()=>validateArtifact({...a,metric:NaN})); assert.throws(()=>validateArtifact({metric:0}));
});
