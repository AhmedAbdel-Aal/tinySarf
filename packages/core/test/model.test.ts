import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {validateManifest,loadModel} from '../src/model';
import {selectedModel} from '../../benchmark/src/selection';
const selected=await selectedModel(),manifest=JSON.parse(await readFile(path.join(selected.directory,'manifest.json'),'utf8')),bytes=await readFile(path.join(selected.directory,'weights.bin'));
test('frozen exported model validates and corrupt weights fail SHA-256',async()=>{
 const model=await loadModel(manifest,bytes);assert.equal(Object.values(model.tensors).reduce((n,t)=>n+t.length,0),manifest.trainingParameters);
 const corrupt=Uint8Array.from(bytes);corrupt[12]^=1;await assert.rejects(()=>loadModel(manifest,corrupt),/SHA-256/);
});
test('architecture, class order, packed ranges and finite scales fail closed',()=>{
 for(const mutate of [(m:any)=>m.labels.heads.pos.reverse(),(m:any)=>m.labels.heads.root0.reverse(),(m:any)=>m.labels.heads.gender.reverse(),(m:any)=>m.tensors.embedding.offset=4,(m:any)=>m.tensors.embedding.scale=NaN,(m:any)=>m.tensors.embedding.shape=[1,1],(m:any)=>m.reachableWeights++]){
  const m=structuredClone(manifest);mutate(m);assert.throws(()=>validateManifest(m,bytes.length));
 }
});
