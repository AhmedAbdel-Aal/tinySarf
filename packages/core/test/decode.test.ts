import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {decode} from '../src/decode';
import {normalize} from '../src/normalize';
import {validateAnalysis} from '../src/schema';
import type {Logits} from '../src/cpu';
import type {ModelManifest} from '../src/model';
import {selectedModel} from '../../benchmark/src/selection';
const selected=await selectedModel(),manifest:ModelManifest=JSON.parse(await readFile(path.join(selected.directory,'manifest.json'),'utf8'));
manifest.labels.heads.pattern=['__missing__','__unknown__','__na__','ٌطWص+ِيّ','فَعَل','ٱِفْتَعَل','فِعال+ِيّ'];
function oracle(word:string,tags:number[],pattern:string|null=null):Logits {
 const heads=Object.fromEntries(Object.entries(manifest.labels.heads).map(([name,labels])=>{
  const values=new Float32Array(labels.length).fill(-40);
  values[name==='pos'?labels.indexOf('noun'):name==='pattern'&&pattern?labels.indexOf(pattern):0]=40;
  return [name,values];
 }));
 const segmentation=new Float32Array(word.length*8).fill(-40);tags.forEach((tag,i)=>segmentation[i*8+tag]=40);
 return {heads,segmentation};
}
test('decoder admits teacher-attested particle and oath-preposition prefixes',()=>{
 for(const [word,tag] of [['بكتاب',1],['وكتاب',1],['فكتاب',1],['وكتاب',2]] as const){
  const input=normalize(word),[result]=decode(input,oracle(word,[tag,...Array(word.length-1).fill(4)]),manifest,1);
  assert.equal(result.spans[0].type,tag===1?'particle':'preposition');assert.equal(result.spans[0].end,1);validateAnalysis(word,result);
 }
});
test('corrupted legacy pattern labels become null while valid Arabic patterns survive',()=>{
 const malformed=manifest.labels.heads.pattern.find(p=>p.includes('W'));
 const valid=manifest.labels.heads.pattern.find(p=>/^[ء-غف-ي\u064b-\u065f\u0670]+$/u.test(p));
 const wasla=manifest.labels.heads.pattern.find(p=>p.includes('ٱ'));
 const derived=manifest.labels.heads.pattern.find(p=>p.includes('+')&&!p.includes('W'));
 assert.ok(malformed);assert.ok(valid);
 assert.ok(wasla);assert.ok(derived);
 for(const pattern of [malformed,valid,wasla,derived]){
  const [result]=decode(normalize('كتاب'),oracle('كتاب',[4,4,4,4],pattern),manifest,1);
  assert.equal(result.pattern,pattern===malformed?null:pattern);
 }
});
