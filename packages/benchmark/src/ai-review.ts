/** AI diagnostic agreement is deliberately separate from the human-gold loader and promotion. */
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {parseArgs} from 'node:util';
import {ROOT,hash,json,provenance,writeArtifact} from './artifact';
import {selectedModel} from './selection';
import {bootstrap} from './math';
import {normalize} from '../../core/src/normalize';
import {validateAnalysis,type MorphSpan,type MorphAnalysis} from '../../core/src/schema';
import {loadModel} from '../../core/src/model';
import {CPUBackend} from '../../core/src/cpu';
import {createAnalyze} from '../../core/src/scheduler';
import {rulesBaseline,frequencyBaseline} from './baselines';
import type {EvaluationWord} from './metrics';

export interface AIRecord {
 word:string; expectedError?:boolean; roots:(string|null)[]; pos:string[];
 segmentations:MorphSpan[][]; confidence:'high'|'medium'|'low'; notes:string;
}
export interface AIReview {
 schemaVersion:1; name:'ai-reviewed-challenge-v1'; labelOrigin:'ai-reviewed';
 independentOfModelAndTeacher:true; humanReviewed:false;
 annotation:{annotators:string[];adjudicator:string;completedAt:string;inputSha256:string;annotatorFiles:{file:string;sha256:string}[]};
 records:AIRecord[];limitations:string[];
}
export function validateAIReview(d:AIReview,words:string[]):void {
 if(d.schemaVersion!==1||d.name!=='ai-reviewed-challenge-v1'||d.labelOrigin!=='ai-reviewed'||d.humanReviewed!==false||d.independentOfModelAndTeacher!==true)throw new Error('AI review must not claim human gold provenance');
 if(d.annotation?.annotators?.length!==2||new Set(d.annotation.annotators).size!==2||!d.annotation.adjudicator||d.annotation.annotators.includes(d.annotation.adjudicator)||!d.annotation.completedAt||d.annotation.annotatorFiles?.length!==2||new Set(d.annotation.annotatorFiles.map(a=>a.file)).size!==2)throw new Error('Two blind annotations and a separate adjudicator are required');
 if(!Array.isArray(d.records)||JSON.stringify(d.records.map(r=>r.word))!==JSON.stringify(words))throw new Error('AI review must preserve the complete frozen word list and order');
 for(const r of d.records){
  if(!['high','medium','low'].includes(r.confidence)||typeof r.notes!=='string')throw new Error('AI confidence and notes are required');
  if(r.expectedError){let invalid=false;try{normalize(r.word);}catch{invalid=true;}if(!invalid||r.roots.length||r.pos.length||r.segmentations.length)throw new Error('Invalid rejection annotation');continue;}
  normalize(r.word);if(!r.roots.length||!r.pos.length||!r.segmentations.length)throw new Error('Missing accepted AI field values');
  for(const root of r.roots)for(const pos of r.pos)for(const spans of r.segmentations)validateAnalysis(r.word,{root,pos,spans,pattern:null,features:{},score:0});
 }
}
const canonicalRoot=(root:string|null)=>root?.replace(/[أإؤئ]/g,'ء')??null;
const spanKey=(spans:MorphSpan[])=>JSON.stringify(spans.map(s=>[s.type,s.start,s.end]));
export function fieldMatches(row:AIRecord,predictions:MorphAnalysis[]){
 return {rootExact:predictions.map(p=>row.roots.includes(p.root)),rootCanonical:predictions.map(p=>row.roots.map(canonicalRoot).includes(canonicalRoot(p.root))),pos:predictions.map(p=>row.pos.includes(p.pos)),segmentation:predictions.map(p=>row.segmentations.some(s=>spanKey(s)===spanKey(p.spans)))};
}
const rate=(values:number[])=>({numerator:values.reduce((a,b)=>a+b,0),denominator:values.length,value:values.length?values.reduce((a,b)=>a+b,0)/values.length:null,ci95:bootstrap(values)});
export async function evaluateAI(file:string,local=false){
 const bytes=await readFile(file),dataset=JSON.parse(bytes.toString()) as AIReview;
 const inputFile=path.join(ROOT,'packages/training/data/ai-review/words.json'),inputBytes=await readFile(inputFile),inputs=JSON.parse(inputBytes.toString());
 validateAIReview(dataset,inputs.words);if(hash(inputBytes)!==dataset.annotation.inputSha256)throw new Error('Frozen AI input hash changed');
 for(const [i,source] of dataset.annotation.annotatorFiles.entries()){
  const sourcePath=path.resolve(ROOT,source.file);if(!sourcePath.startsWith(path.join(ROOT,'packages/training/data/ai-review')+path.sep))throw new Error('Blind annotation path escapes dataset');
  const sourceBytes=await readFile(sourcePath),annotation=JSON.parse(sourceBytes.toString());
  if(hash(sourceBytes)!==source.sha256||annotation.schemaVersion!==1||annotation.annotator!==dataset.annotation.annotators[i]||annotation.labelOrigin!=='ai-authored'||annotation.independentOfModelAndTeacher!==true||JSON.stringify(annotation.records.map((r:AIRecord)=>r.word))!==JSON.stringify(inputs.words))throw new Error('Blind annotation evidence mismatch');
 }
 const selected=await selectedModel(),manifest=await json(path.join(selected.directory,'manifest.json')),model=await loadModel(manifest,await readFile(path.join(selected.directory,'weights.bin')));
 const analyze=createAnalyze(async()=>({model,backend:new CPUBackend(model)}));
 const split=await json(path.join(ROOT,'packages/training/data/split-manifest.json'));
 const identities:Record<string,Set<string>>={};let training:EvaluationWord[]=[];
 for(const role of ['train','verification','mining']){const data=await readFile(path.join(ROOT,'packages/training/data',split.directory,`${role}.json`));if(hash(data)!==split.roles[role].sha256)throw new Error('Split identity audit digest changed');const records=JSON.parse(data.toString()).records;identities[role]=new Set(records.map((r:AIRecord)=>normalize(r.word).text));if(role==='train')training=records.map((r:any)=>({...r,analyses:r.analyses.map((a:any)=>({...a,features:Object.fromEntries(Object.entries(a.features).filter(([,v])=>!(v as string).startsWith('__')))}))}));}
 const predictions=[];
 for(const row of dataset.records){
  if(row.expectedError){let rejected=false;try{await analyze(row.word,{backend:'cpu'});}catch{rejected=true;}predictions.push({word:row.word,expectedError:true,rejected,confidence:row.confidence});continue;}
  const analyses=await analyze(row.word,{backend:'cpu',topK:3}),normalized=normalize(row.word).text;
  predictions.push({word:row.word,analyses,confidence:row.confidence,agreement:fieldMatches(row,analyses),overlap:Object.fromEntries(Object.entries(identities).map(([role,words])=>[role,words.has(normalized)]))});
 }
 const accepted=predictions.filter((p):p is Extract<typeof p,{analyses:MorphAnalysis[]} >=>p.agreement!==undefined),summarize=(rows:typeof accepted)=>Object.fromEntries(['rootExact','rootCanonical','pos','segmentation'].map(key=>[key,{top1:rate(rows.map(r=>Number((r.agreement as any)[key][0]))),top3:rate(rows.map(r=>Number((r.agreement as any)[key].some(Boolean))))}]));
 const validRows=dataset.records.filter(r=>!r.expectedError),normalizedRows=validRows.map(r=>({word:normalize(r.word).text,analyses:[]}));
 const baselines=Object.fromEntries(Object.entries({rules:rulesBaseline(normalizedRows),frequency:frequencyBaseline(training,normalizedRows)}).map(([name,outputs])=>{
  const rows=validRows.map((row,i)=>{const input=normalize(row.word),analyses=outputs[i].map(a=>({...a,spans:a.spans.map(s=>({...s,start:input.offsets[s.start].start,end:input.offsets[s.end-1].end}))}));analyses.forEach(a=>validateAnalysis(row.word,a));return {...accepted[i],analyses,agreement:fieldMatches(row,analyses)};});
  return [name,{all:summarize(rows),highConfidence:summarize(rows.filter(r=>r.confidence==='high')),unseenTrainingSurface:summarize(rows.filter(r=>!r.overlap?.train)),predictions:rows.map(r=>({word:r.word,analyses:r.analyses}))}];
 }));
 const rejected=predictions.filter(p=>p.expectedError);
 const artifact={...provenance(),kind:'ai-review',runtime:{sha256:hash(await readFile(path.join(ROOT,'packages/core/dist/index.js')))},status:'diagnostic-only-not-human-gold',model:manifest,data:{aiReviewDigest:hash(bytes),inputDigest:hash(inputBytes),verificationDigest:manifest.verificationDigest,goldDigest:null},
  protocol:{labelOrigin:'ai-reviewed',distribution:inputs.distribution,humanReviewed:false,finalTestOpened:false,bootstrap:{seed:42,resamples:2000,unit:'word'},rootCanonicalization:'Seated hamza أإؤئ maps to ء only in the secondary canonical-root metric; exact-root agreement also retained',scoring:'Accepted values scored separately by field; top-3 means at least one candidate matches that field. No full-analysis or gold accuracy claim.'},
  annotation:dataset.annotation,metrics:{all:summarize(accepted),highConfidence:summarize(accepted.filter(r=>r.confidence==='high')),unseenTrainingSurface:summarize(accepted.filter(r=>!r.overlap?.train)),inputRejection:rate(rejected.map(r=>Number(r.rejected)))},
  counts:{inputs:dataset.records.length,valid:accepted.length,rejections:rejected.length,highConfidence:accepted.filter(r=>r.confidence==='high').length,lowConfidence:accepted.filter(r=>r.confidence==='low').length,overlap:Object.fromEntries(Object.keys(identities).map(role=>[role,accepted.filter(r=>r.overlap?.[role]).length]))},predictions,baselines,
  limitations:[...dataset.limitations,'Same-family AI annotators have correlated knowledge and errors; multiple agents are not independent humans.','Curated inputs can overlap training and verification; surface-disjoint does not establish lemma-family disjointness.','Field-level accepted sets omit pattern/features and may be incomplete or permissive. This report cannot satisfy human-gold promotion gates.']};
 const output=await writeArtifact('ai-review',artifact,local?'packages/benchmark/local':'packages/benchmark/results');console.log(JSON.stringify({file:output,counts:artifact.counts,metrics:artifact.metrics}));return output;
}
if(import.meta.url===`file://${process.argv[1]}`){const {values}=parseArgs({options:{file:{type:'string',default:'packages/training/data/ai-review/adjudicated.json'},local:{type:'boolean',default:false}}});await evaluateAI(values.file!,values.local);}
