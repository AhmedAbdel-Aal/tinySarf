/** Paired root comparison on unchanged words and exact teacher root strings. */
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {parseArgs} from 'node:util';
import {ROOT,hash,json,provenance,writeArtifact} from './artifact';
import {bootstrap,pairedBootstrap} from './math';

const {values}=parseArgs({options:{baseline:{type:'string',default:'packages/benchmark/results/correctness-2026-09-11T212223821Z.json'},candidate:{type:'string'},local:{type:'boolean',default:false}}});
if(!values.candidate)throw new Error('--candidate correctness artifact is required');
const baseline=await json(path.resolve(ROOT,values.baseline!)),candidate=await json(path.resolve(ROOT,values.candidate));
const split=await json(path.join(ROOT,'packages/training/data/split-manifest.json'));
const bytes=await readFile(path.join(ROOT,'packages/training/data',split.directory,'verification.json'));
if(hash(bytes)!==baseline.data.verificationDigest||hash(bytes)!==candidate.data.verificationDigest)throw new Error('Comparison labels must be identical');
const rows=JSON.parse(bytes.toString()).records;
if([baseline,candidate].some(report=>JSON.stringify(report.predictions.map((r:any)=>r.word))!==JSON.stringify(rows.map((r:any)=>r.word))))throw new Error('Comparison words must be identical and ordered');
const baselineTrain=(await json(path.join(ROOT,'packages/training/data',split.directory,'train.json'))).records;
const seen=new Set(baselineTrain.flatMap((r:any)=>r.analyses.map((a:any)=>a.root)).filter(Boolean));
const a=rows.map((r:any,i:number)=>Number(r.analyses.some((x:any)=>x.root===baseline.predictions[i].analyses[0].root)));
const b=rows.map((r:any,i:number)=>Number(r.analyses.some((x:any)=>x.root===candidate.predictions[i].analyses[0].root)));
const groups:Record<string,number[]>={all:rows.map((_:any,i:number)=>i),'root-required':[],'root-optional':[],'null-only':[],'unseen-in-original-training':[],'seen-in-original-training':[]};
rows.forEach((r:any,i:number)=>{
 const roots=r.analyses.map((x:any)=>x.root);
 groups[roots.every(Boolean)?'root-required':roots.some(Boolean)?'root-optional':'null-only'].push(i);
 groups[roots.some((x:any)=>seen.has(x))?'seen-in-original-training':'unseen-in-original-training'].push(i);
 for(const label of r.slices??[])(groups[label]??=[]).push(i);
});
const rate=(xs:number[])=>({numerator:xs.reduce((x,y)=>x+y,0),denominator:xs.length,value:xs.length?xs.reduce((x,y)=>x+y,0)/xs.length:null,ci95:bootstrap(xs)});
const slices=Object.fromEntries(Object.entries(groups).map(([name,indices])=>[name,{baseline:rate(indices.map(i=>a[i])),candidate:rate(indices.map(i=>b[i])),delta:pairedBootstrap(indices.map(i=>b[i]),indices.map(i=>a[i]))}]));
const nonRootFields=['spans','pattern','pos','features'];
const changedNonRootTop1=Object.fromEntries(nonRootFields.map(field=>[field,rows.filter((_:any,i:number)=>JSON.stringify(baseline.predictions[i].analyses[0][field])!==JSON.stringify(candidate.predictions[i].analyses[0][field])).length]));
const outcomes={fixed:0,regressed:0,bothCorrect:0,bothWrong:0};
for(let i=0;i<a.length;i++)outcomes[a[i]?b[i]?'bothCorrect':'regressed':b[i]?'fixed':'bothWrong']++;
const artifact={...provenance(),kind:'root-comparison',model:candidate.model,baselineModel:baseline.model,runtime:candidate.runtime,
 data:candidate.data,protocol:{labelOrigin:'teacher-generated',finalTestOpened:false,sameWordsAndLabels:true,rootCanonicalization:'none',bootstrap:{seed:42,resamples:2000,unit:'word'},unseenRootDefinition:'Fixed original multi-task training roots for both models; candidate-specific unseen roots are reported separately by correctness.ts'},
 sources:{baseline:{file:values.baseline,sha256:hash(await readFile(path.resolve(ROOT,values.baseline!)))},candidate:{file:values.candidate,sha256:hash(await readFile(path.resolve(ROOT,values.candidate))) }},
 slices,outcomes,changedNonRootTop1,relativeRootErrorReduction:(a.length-a.reduce((x:number,y:number)=>x+y,0))?((b.reduce((x:number,y:number)=>x+y,0)-a.reduce((x:number,y:number)=>x+y,0))/(a.length-a.reduce((x:number,y:number)=>x+y,0))):null,
 changes:rows.flatMap((r:any,i:number)=>a[i]===b[i]?[]:[{word:r.word,baselineRoot:baseline.predictions[i].analyses[0].root,candidateRoot:candidate.predictions[i].analyses[0].root,outcome:b[i]?'fixed':'regressed'}]),
 limitations:['Repeated teacher verification used for model selection; not independent human accuracy.','Data, root architecture, loss and decoding changed together; the full-model comparison does not isolate their individual causal effects.','The inherited non-root encoder retains its original training provenance.']};
const file=await writeArtifact('root-comparison',artifact,values.local?'packages/benchmark/local':'packages/benchmark/results');
console.log(JSON.stringify({file,root:slices.all,outcomes,changedNonRootTop1,errorReduction:artifact.relativeRootErrorReduction}));
