import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {parseArgs} from 'node:util';
import {ROOT,hash,provenance,validateArtifact,writeArtifact} from './artifact';
import {pairedBootstrap} from './math';
const {values}=parseArgs({options:{baseline:{type:'string'},candidate:{type:'string'},local:{type:'boolean',default:false}}});
if(!values.baseline||!values.candidate)throw new Error('Provide --baseline and --candidate correctness JSON artifacts');
const inputs=await Promise.all([values.baseline,values.candidate].map(async file=>{
 const bytes=await readFile(path.resolve(ROOT,file)),artifact=JSON.parse(bytes.toString());validateArtifact(artifact);
 if(artifact.kind!=='correctness'||artifact.data.evaluationRole!=='verification')throw new Error('Only teacher verification comparisons are supported');
 return {file:path.relative(ROOT,path.resolve(ROOT,file)),sha256:hash(bytes),artifact};
}));
const [baseline,candidate]=inputs.map(i=>i.artifact);
if(baseline.data.evaluationDigest!==candidate.data.evaluationDigest||baseline.runtime?.sha256!==candidate.runtime?.sha256||!baseline.runtime?.sha256||JSON.stringify(baseline.predictions.map((r:any)=>r.word))!==JSON.stringify(candidate.predictions.map((r:any)=>r.word)))throw new Error('Compare identical verification words, labels and decoder/runtime only');
const a=baseline.correctness.teacherAgreement,b=candidate.correctness.teacherAgreement;
const deltas=Object.fromEntries(Object.keys(a.perWord).map(name=>[name,pairedBootstrap(b.perWord[name],a.perWord[name])]));
const fields=['segmentationExactMatch','rootExactMatch','patternExactMatch','posAccuracy','fullAnalysisTop1','fullAnalysisTop3'];
const metrics=Object.fromEntries(fields.map(name=>[name,{baseline:a[name],candidate:b[name]}]));
const slices=Object.fromEntries(Object.keys(baseline.correctness.slices).map(name=>{
 const left=baseline.correctness.slices[name],right=candidate.correctness.slices[name];
 if(!right||left.digest!==right.digest)return [name,{comparable:false,reason:'Membership depends on the model training data'}];
 return [name,{comparable:true,count:left.count,metrics:Object.fromEntries(fields.map(field=>[field,{baseline:left.metrics[field],candidate:right.metrics[field]}]))}];
}));
const report={...provenance(),kind:'corrected-comparison',status:'experimental-unpromoted',model:candidate.model,baselineModel:baseline.model,runtime:baseline.runtime,data:{...candidate.data,goldDigest:null},inputs:inputs.map(({file,sha256})=>({file,sha256})),metrics,deltas,slices,protocol:{comparison:'Same words and corrected teacher mapping; current reference and a separately trained pilot',humanGold:false,finalTestOpened:false,selectionEligible:false,bootstrap:{resamples:2000,seed:42,unit:'word'}},limitations:['Corrected-label agreement is not directly comparable to the legacy-label headline score.','Pilot changes label mapping, training coverage and target sampling together; this is not a controlled single-factor ablation.','No automatic model selection or promotion is performed. Human gold and release gates remain mandatory.']};
const file=await writeArtifact('corrected-comparison',report,values.local?'packages/benchmark/local':'packages/benchmark/results');
console.log(JSON.stringify({file,metrics,deltas}));
