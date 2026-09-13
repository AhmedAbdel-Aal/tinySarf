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
test('root transform metadata rejects unsupported copies, memorized exceptions and invalid priors',()=>{
 const decoder={format:'learned-transforms-v1',priorWeight:.25,unsupportedPenalty:4,nullPenalty:0,minimumDistinctRoots:3,whole:[],stem:[{surface:'**ا*',output:[0,1,3],distinctRoots:3}]};
 const clean={...structuredClone(manifest),rootDecoder:decoder};validateManifest(clean,bytes.length);
 for(const mutate of [(d:any)=>d.priorWeight=NaN,(d:any)=>d.stem[0].output[2]=20,(d:any)=>d.stem[0].output=['ك','ت','ب'],(d:any)=>d.stem[0].distinctRoots=1,(d:any)=>d.stem.push(structuredClone(d.stem[0]))]){
  const m=structuredClone(clean);mutate(m.rootDecoder);assert.throws(()=>validateManifest(m,bytes.length));
 }
 if(manifest.rootArchitecture)for(const mutate of [(m:any)=>m.rootArchitecture.width=0,(m:any)=>m.rootArchitecture.embedding=32,(m:any)=>m.format='cnn-v1']){
  const m=structuredClone(manifest);mutate(m);assert.throws(()=>validateManifest(m,bytes.length));
 }
});
