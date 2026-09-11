import {selectedModel} from './selection';
import {readFile,appendFile,mkdir} from "node:fs/promises";
import path from "node:path";
import {parseArgs} from "node:util";
import {ROOT,json,hash,provenance,writeArtifact} from "./artifact";
import {evaluate,analysisKey,type EvaluationWord} from "./metrics";
import {pairedBootstrap} from "./math";
import {frequencyBaseline,rulesBaseline} from "./baselines";
import {loadModel,type LoadedModel} from "../../core/src/model";
import {CPUBackend} from "../../core/src/cpu";
import {createAnalyze} from "../../core/src/scheduler";
import {validateAnalysis,type MorphAnalysis} from "../../core/src/schema";
import {loadGold} from './gold';
function publicRows(rows:any[]):EvaluationWord[] {return rows.map(r=>({...r,analyses:r.analyses.map((a:any)=>({spans:a.spans,root:a.root,pattern:a.pattern,pos:a.pos,features:Object.fromEntries(Object.entries(a.features).filter(([,v])=>!(v as string).startsWith("__"))),score:0}))}));}
function subset(rows:EvaluationWord[], predictions:MorphAnalysis[][],indices:number[]) {return evaluate(indices.map(i=>rows[i]),indices.map(i=>predictions[i]),{bootstrap:false});}
export async function correctness(modelDirectory?:string,local=false,goldFiles:string[]=[],releaseCandidate=false,role:'verification'|'final-test'='verification',splitManifest?:string,crossDataset=false) {
  if(role==='final-test'&&!releaseCandidate)throw new Error('Final-test access requires explicit --release-candidate');
  if((splitManifest||crossDataset)&&(role!=='verification'||goldFiles.length))throw new Error('Dataset comparisons are limited to teacher verification');
  const selection=await selectedModel(modelDirectory),dir=selection.directory;
  const manifest=await json(path.join(dir,"manifest.json")),packed=await readFile(path.join(dir,"weights.bin"));
  const model=await loadModel(manifest,packed),split=await json(path.resolve(ROOT,splitManifest??"packages/training/data/split-manifest.json"));
  const base=path.join(ROOT,"packages/training/data",split.directory);
  const verificationBytes=await readFile(path.join(base,`${role}.json`)),trainBytes=await readFile(path.join(base,"train.json"));
  if((manifest.verificationDigest!==split.roles.verification.sha256&&!crossDataset) || hash(verificationBytes)!==split.roles[role].sha256 || hash(trainBytes)!==split.roles.train.sha256) throw new Error("Frozen dataset/checkpoint digest mismatch; evaluating an explicitly different verification mapping requires --cross-dataset");
  if(role==='final-test'){const audit=path.join(ROOT,'packages/training/data/generated/audit');await mkdir(audit,{recursive:true});await appendFile(path.join(audit,'final-test-access.jsonl'),JSON.stringify({createdAt:new Date().toISOString(),role,model:manifest.id,digest:hash(verificationBytes),purpose:'explicit release-candidate evaluation'})+'\n');}
  const rawRows=JSON.parse(verificationBytes.toString()).records,rawTrain=JSON.parse(trainBytes.toString()).records;
  const rows=publicRows(rawRows),train=publicRows(rawTrain);
  for(const row of rows) for(const a of row.analyses) validateAnalysis(row.word,a);
  const floatBytes=await readFile(path.join(dir,"float.weights.bin"));
  if(hash(floatBytes)!==manifest.floatBinarySha256) throw new Error("Float checkpoint hash mismatch");
  const floatModel:LoadedModel={...model,tensors:{}};
  for(const [name,t] of Object.entries(manifest.tensors) as [string,{offset:number;length:number}][]) {
    const data=floatBytes.subarray(t.offset*4,(t.offset+t.length)*4);
    floatModel.tensors[name]=new Float32Array(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength));
  }
  const predict=async(m:LoadedModel,inputRows:EvaluationWord[]=rows)=>{
    const backend=new CPUBackend(m),analyze=createAnalyze(async()=>({model:m,backend}));const outputs:MorphAnalysis[][]=[];
    for(let i=0;i<inputRows.length;i+=64) {outputs.push(...await analyze.batch(inputRows.slice(i,i+64).map(r=>r.word),{backend:"cpu",topK:3}));if(i%256===0) console.log(`${m===model?'int8':'float'} ${i}/${inputRows.length}`);}
    inputRows.forEach((row,i)=>outputs[i].forEach(a=>validateAnalysis(row.word,a)));return outputs;
  };
  const floatPredictions=await predict(floatModel),quantizedPredictions=await predict(model);
  const fp=evaluate(rows,floatPredictions),qp=evaluate(rows,quantizedPredictions);
  const seenRoots=new Set(rawTrain.flatMap((r:any)=>r.analyses.map((a:any)=>a.root)).filter(Boolean));
  const seenLemmas=new Set(rawTrain.flatMap((r:any)=>r.analyses.map((a:any)=>a.lemmaFamily)));
  const gold:Record<string,unknown>={},goldDigests:Record<string,string>={};
  for(const file of goldFiles) {
    const {dataset,digest}=await loadGold(file,releaseCandidate);
    if(dataset.records.some(r=>r.lemmaFamilies?.some(l=>seenLemmas.has(l))))throw new Error('Gold lemma family overlaps training');
    const accepted=dataset.records.filter(r=>!r.expectedError),rejected=dataset.records.filter(r=>r.expectedError);
    const predictions=await predict(model,accepted),metrics=evaluate(accepted,predictions);
    const backend=new CPUBackend(model),analyze=createAnalyze(async()=>({model,backend}));let rejectionMatches=0;const rejectedMatches=new Set<string>();
    for(const r of rejected){try{await analyze(r.word,{backend:'cpu'});}catch{rejectionMatches++;rejectedMatches.add(r.word);}}
    const goldSlices=Object.fromEntries([...new Set(dataset.records.flatMap(r=>r.slices??[]))].map(name=>{
      const indices=accepted.flatMap((r,i)=>r.slices?.includes(name)?[i]:[]),rejections=rejected.filter(r=>r.slices?.includes(name));
      return [name,{count:indices.length,digest:hash(JSON.stringify(indices.map(i=>accepted[i].word))),metrics:subset(accepted,predictions,indices),inputRejection:{count:rejections.length,accuracy:rejections.length?rejections.filter(r=>rejectedMatches.has(r.word)).length/rejections.length:null}}];
    }));
    gold[dataset.name]={slices:goldSlices,metadata:{source:dataset.source,annotation:dataset.annotation,labelOrigin:dataset.labelOrigin},metrics,baselines:{rules:evaluate(accepted,rulesBaseline(accepted)),frequency:evaluate(accepted,frequencyBaseline(train,accepted))},inputRejection:{numerator:rejectionMatches,denominator:rejected.length,value:rejected.length?rejectionMatches/rejected.length:null}};
    goldDigests[dataset.name]=digest;
  }
  const slices:Record<string,number[]>={};
  const add=(name:string,i:number)=>{(slices[name]??=[]).push(i);};
  rawRows.forEach((row:any,i:number)=>{
    for(const name of row.slices??[]) add(name,i);
    add(row.analyses.some((a:any)=>seenRoots.has(a.root))?'seen-root':'unseen-root',i);
    add(row.analyses.some((a:any)=>seenLemmas.has(a.lemmaFamily))?'seen-lemma':'unseen-lemma',i);
    add(row.analyses.some((a:any)=>a.root!==null)?'non-null-root':'null-root-only',i);
    if(row.analyses.some((a:any)=>a.pos==='proper_noun')) add('named-entity',i);
    add('undiacritized',i);
    add(`length-${row.word.length<=5?'1-5':row.word.length<=8?'6-8':row.word.length<=16?'9-16':'17-32'}`,i);
    add(`ambiguity-${row.analyses.length===1?'1':row.analyses.length<=3?'2-3':row.analyses.length<=10?'4-10':'11+'}`,i);
  });
  for(const name of ['weak-root','hamzated','doubled','quadriliteral','broken-plural','clitic-stack','diacritized','named-entity','foreign','malformed','non-MSA','seen-lemma','unseen-lemma','seen-root','unseen-root']) slices[name]??=[];
  const breakdowns=Object.fromEntries(Object.entries(slices).map(([name,indices])=>[name,{count:indices.length,digest:hash(JSON.stringify(indices.map(i=>rows[i].word))),metrics:subset(rows,quantizedPredictions,indices)}]));
  const deltas=Object.fromEntries(Object.keys(fp.perWord).map(k=>[k,pairedBootstrap(qp.perWord[k as keyof typeof qp.perWord],fp.perWord[k as keyof typeof fp.perWord])]));
  const outputDifferences=rows.flatMap((row,i)=>JSON.stringify(quantizedPredictions[i].map(analysisKey))===JSON.stringify(floatPredictions[i].map(analysisKey))?[]:[{word:row.word,float:floatPredictions[i],int8:quantizedPredictions[i]}]);
  const artifact={...provenance(),kind:"correctness",runtime:{sha256:hash(await readFile(path.join(ROOT,'packages/core/dist/index.js')))},status:"experimental-unpromoted",model:manifest,data:{datasetId:split.datasetId,verificationDigest:split.roles.verification.sha256,checkpointVerificationDigest:manifest.verificationDigest,crossDatasetEvaluation:crossDataset,splitManifest:splitManifest??'packages/training/data/split-manifest.json',evaluationRole:role,evaluationDigest:hash(verificationBytes),goldDigest:goldFiles.length?goldDigests:null,trainDigest:hash(trainBytes),normalizationVersion:"arabic-v1"},
    protocol:{labelOrigin:"teacher-generated",distribution:"teacher-sampled lexical surfaces; not natural text",seenSlicesAndFrequencyBaseline:'Reference training pool from the evaluation split; cross-dataset comparisons may differ from the checkpoint training pool',scoreSemantics:"ranking",threshold:null,bootstrap:{seed:42,resamples:2000,unit:"word"},finalTestOpened:role==='final-test'||goldFiles.length>0},
    correctness:{teacherAgreement:qp,goldNatural:gold['gold-natural']??null,goldBalanced:gold['gold-balanced']??null,float:fp,floatToQuantizedDeltas:deltas,completeLabelAgreement:{numerator:rows.length-outputDifferences.length,denominator:rows.length,value:(rows.length-outputDifferences.length)/rows.length},slices:breakdowns,
      baselines:{rules:evaluate(rows,rulesBaseline(rows)),frequency:evaluate(rows,frequencyBaseline(train,rows))}},outputDifferences,predictions:quantizedPredictions.map((analyses,i)=>({word:rows[i].word,analyses})),
    limitations:[...(!goldFiles.length?["No independent gold annotation","No final-test evaluation"]:[]),"Teacher verification is not a natural text-distribution sample","Missing challenge slices remain unmeasured","Macro-F1 includes the explicit absent-feature class; per-class support is retained"]};
  const file=await writeArtifact(splitManifest||crossDataset?'comparison-correctness':role==='final-test'?'final-test-correctness':"correctness",artifact,local?"packages/benchmark/local":"packages/benchmark/results");
  console.log(JSON.stringify({file,segmentation:qp.segmentationExactMatch,root:qp.rootExactMatch,posMacroF1:qp.pos.macroF1,fullTop1:qp.fullAnalysisTop1,fullTop3:qp.fullAnalysisTop3,rootNonNull:breakdowns['non-null-root'].metrics.rootExactMatch}));return file;
}
if(import.meta.url===`file://${process.argv[1]}`) {
  const {values}=parseArgs({options:{"model-dir":{type:"string"},local:{type:"boolean",default:false},gold:{type:'string',multiple:true},'release-candidate':{type:'boolean',default:false},'dataset-role':{type:'string',default:'verification'},'split-manifest':{type:'string'},'cross-dataset':{type:'boolean',default:false}}});if(!['verification','final-test'].includes(values['dataset-role']!))throw new Error('Unknown evaluation role');await correctness(values['model-dir'],values.local,values.gold,values['release-candidate'],values['dataset-role'] as 'verification'|'final-test',values['split-manifest'],values['cross-dataset']);
}
