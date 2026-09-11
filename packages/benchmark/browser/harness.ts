import {loadModel,type LoadedModel} from '../../core/src/model';
import {CPUBackend,type Logits} from '../../core/src/cpu';
import {GPUBackend} from '../../core/src/gpu';
import {normalize} from '../../core/src/normalize';
import {decode} from '../../core/src/decode';
import {validateAnalysis} from '../../core/src/schema';
import {analysisKey} from '../src/metrics';

let debug:Promise<{model:LoadedModel;cpu:CPUBackend;gpu:GPUBackend}>|undefined;
const coreURL='/tinysarf/index.js';
let publicModule:Promise<typeof import('../../core/src/index')>|undefined;
let lastAdapter:Record<string,unknown>|null=null;
if(navigator.gpu) {
  const native=navigator.gpu.requestAdapter.bind(navigator.gpu);
  navigator.gpu.requestAdapter=async(options)=>{
    const adapter=await native(options);
    if(adapter) {const i=adapter.info;lastAdapter={vendor:i.vendor,architecture:i.architecture,device:i.device,description:i.description,isFallbackAdapter:i.isFallbackAdapter};}
    return adapter;
  };
}
function getDebug() {
  return debug??=(async()=>{
    const [manifest,weights]=await Promise.all([fetch('/manifest.json').then(r=>r.json()),fetch('/weights.bin').then(r=>r.arrayBuffer())]);
    const model=await loadModel(manifest,new Uint8Array(weights));return {model,cpu:new CPUBackend(model),gpu:await GPUBackend.create(model)};
  })();
}
function getPublic():Promise<typeof import('../../core/src/index')> {return publicModule??=import(/* @vite-ignore */ coreURL);}
async function digest(value:unknown) {
  const bytes=new TextEncoder().encode(JSON.stringify(value));const buffer=await crypto.subtle.digest('SHA-256',bytes);
  return Array.from(new Uint8Array(buffer),b=>b.toString(16).padStart(2,'0')).join('');
}
function compare(expected:Float32Array,actual:Float32Array,classes:number) {
  if(expected.length!==actual.length) throw new Error('Parity tensor length mismatch');
  let maxAbsoluteError=0,maxRelativeError=0,argmaxMatches=0,finite=true;
  for(let i=0;i<expected.length;i++) {
    const difference=Math.abs(expected[i]-actual[i]);maxAbsoluteError=Math.max(maxAbsoluteError,difference);maxRelativeError=Math.max(maxRelativeError,difference/Math.max(Math.abs(expected[i]),1e-6));
    finite&&=Number.isFinite(expected[i])&&Number.isFinite(actual[i]);
  }
  const count=expected.length/classes;
  for(let i=0;i<expected.length;i+=classes) {
    let a=0,b=0;for(let c=1;c<classes;c++){if(expected[i+c]>expected[i+a])a=c;if(actual[i+c]>actual[i+b])b=c;}
    argmaxMatches+=Number(a===b);
  }
  return {elements:expected.length,maxAbsoluteError,maxRelativeError,argmaxMatches,argmaxCount:count,argmaxAgreement:count?argmaxMatches/count:null,finite};
}
async function publicRun(words:string[],backend:'cpu'|'webgpu'='webgpu',topK=3) {
  const imported=performance.now(),{analyze}=await getPublic(),importMs=performance.now()-imported;
  const start=performance.now(),outputs=await analyze.batch(words,{backend,topK}),apiMs=performance.now()-start;
  for(let i=0;i<words.length;i++) outputs[i].forEach(a=>validateAnalysis(words[i],a));
  return {importMs,apiMs,outputHash:await digest(outputs.map(a=>a.map(analysisKey))),outputs};
}
async function parity(words:string[],capture=false) {
  const {model,cpu,gpu}=await getDebug(),prepared=words.map(normalize),m=model.manifest;
  const pending:{name:string;buffer:GPUBuffer;batch:number;length:number;elements:number}[]=[];
  if(capture && words.length>256) throw new Error('Tensor trace is limited to one physical batch');
  if(capture) gpu.onStage=(encoder,name,source,elements,batch,length)=>{
    if(name==='heads')return;
    const buffer=gpu.device.createBuffer({size:elements*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});encoder.copyBufferToBuffer(source,0,buffer,0,elements*4);pending.push({name,buffer,batch,length,elements});
  };
  let gr;
  try {gr=await gpu.run(prepared);} finally {gpu.onStage=undefined;}
  const cr=await cpu.run(prepared,capture),tensors:Record<string,ReturnType<typeof compare>>={};
  for(const item of pending) {
    await item.buffer.mapAsync(GPUMapMode.READ);const padded=new Float32Array(item.buffer.getMappedRange());
    const values:number[]=[],channels=item.name==='embedding'?m.embedding:m.width;
    for(let b=0;b<item.batch;b++) {
      if(item.name==='pool') values.push(...padded.slice(b*channels,(b+1)*channels));
      else for(let p=0;p<prepared[b].text.length;p++) values.push(...padded.slice((b*item.length+p)*channels,(b*item.length+p+1)*channels));
    }
    const name=item.name==='pool'?'pooled':item.name;tensors[name]=compare(cr.trace![name],Float32Array.from(values),channels);item.buffer.unmap();item.buffer.destroy();
  }
  for(const [name,classes] of Object.entries({segmentation:m.labels.segmentation,...m.labels.heads})) {
    const field=(x:Logits)=>name==='segmentation'?x.segmentation:x.heads[name];
    const a=Float32Array.from(cr.words.flatMap(x=>Array.from(field(x)))),b=Float32Array.from(gr.words.flatMap(x=>Array.from(field(x))));tensors[name]=compare(a,b,classes.length);
  }
  const cpuAnalyses=prepared.map((word,i)=>decode(word,cr.words[i],m,3)),gpuAnalyses=prepared.map((word,i)=>decode(word,gr.words[i],m,3));
  const disagreements=words.flatMap((word,i)=>JSON.stringify(cpuAnalyses[i].map(analysisKey))===JSON.stringify(gpuAnalyses[i].map(analysisKey))?[]:[{word,cpu:cpuAnalyses[i],gpu:gpuAnalyses[i]}]);
  gpuAnalyses.forEach((aa,i)=>aa.forEach(a=>validateAnalysis(words[i],a)));
  const heads=Object.entries(tensors).filter(([k])=>k==='segmentation'||k in m.labels.heads).map(([,v])=>v);
  return {words:words.length,tensors,headArgmaxMatches:heads.reduce((s,h)=>s+h.argmaxMatches,0),headArgmaxCount:heads.reduce((s,h)=>s+h.argmaxCount,0),completeAnalysisMatches:words.length-disagreements.length,completeAnalysisCount:words.length,disagreements,finite:Object.values(tensors).every(t=>t.finite),metadata:gpu.metadata()};
}
async function referenceParity() {
  const expected=await fetch('/parity.json').then(r=>r.json()),{model,cpu}=await getDebug(),words:string[]=expected.words;
  const actual=await cpu.run(words.map(normalize)),reports:Record<string,ReturnType<typeof compare>>={};
  for(const [name,labels] of Object.entries({segmentation:model.manifest.labels.segmentation,...model.manifest.labels.heads})) {
    const values:number[]=[];
    for(let i=0;i<words.length;i++) {
      if(name==='segmentation') for(let p=0;p<words[i].length;p++) values.push(...expected.quantized[name][i][p]);
      else values.push(...expected.quantized[name][i]);
    }
    reports[name]=compare(Float32Array.from(values),Float32Array.from(actual.words.flatMap(x=>Array.from(name==='segmentation'?x.segmentation:x.heads[name]))),labels.length);
  }
  return reports;
}
async function contracts() {
  const {analyze}=await getPublic(),checks:Record<string,boolean>={};
  const rejects=async(fn:()=>Promise<unknown>)=>{try{await fn();return false;}catch{return true;}};
  checks.emptyBatch=(await analyze.batch([])).length===0;
  for(const word of ['', ' ', 'foreign', '😀','كتب جيد','َكتب','ك'.repeat(33)]) checks[`invalid:${word}`]=await rejects(()=>analyze(word));
  checks.invalidTopK=await rejects(()=>analyze('كتب',{topK:4}));
  checks.maximumBatchRejected=await rejects(()=>analyze.batch(Array(2049).fill('كتب')));
  checks.concurrent=JSON.stringify(await Promise.all([analyze('كتب'),analyze('دخل')])).length>0;
  const marks=await analyze('وَبِكِتَابِهِمْ');marks.forEach(a=>validateAnalysis('وَبِكِتَابِهِمْ',a));checks.diacritizedOffsets=true;
  const maximum='ك'.repeat(32),maxResult=await analyze.batch(Array(2048).fill(maximum));
  checks.maximumShapes=maxResult.length===2048 && maxResult.every(aa=>{aa.forEach(a=>validateAnalysis(maximum,a));return aa.length>0;});
  const native=navigator.gpu.requestAdapter.bind(navigator.gpu);
  navigator.gpu.requestAdapter=async()=>null;
  try {const path=`${coreURL}?adapter-failure`;const fresh=await import(/* @vite-ignore */ path);checks.adapterFailure=await rejects(()=>fresh.analyze('كتب'));checks.explicitCPUFallback=(await fresh.analyze('كتب',{backend:'cpu'})).length>0;} finally {navigator.gpu.requestAdapter=native;}
  const own=Object.getOwnPropertyDescriptor(navigator,'gpu');const originalGPU=navigator.gpu;
  try {
    Object.defineProperty(navigator,'gpu',{value:undefined,configurable:true});
    const path=`${coreURL}?gpu-unavailable`;const fresh=await import(/* @vite-ignore */ path);checks.webgpuUnavailable=await rejects(()=>fresh.analyze('كتب'));checks.emptyWithoutGPU=(await fresh.analyze.batch([])).length===0;
  } finally {if(own)Object.defineProperty(navigator,'gpu',own);else {delete (navigator as unknown as Record<string,unknown>).gpu;}if(navigator.gpu!==originalGPU)throw new Error('Failed to restore WebGPU');}
  let captured:GPUDevice|undefined;
  navigator.gpu.requestAdapter=async(options)=>{
    const adapter=await native(options);if(adapter){const rd=adapter.requestDevice.bind(adapter);adapter.requestDevice=async(descriptor)=>{captured=await rd(descriptor);return captured;};}return adapter;
  };
  try {
    const path=`${coreURL}?device-loss`;const fresh=await import(/* @vite-ignore */ path);await fresh.analyze('كتب');
    if(!captured)throw new Error('No production device acquired');captured.destroy();await captured.lost;
    checks.deviceLossRecovery=(await fresh.analyze('كتب')).length>0;
  } finally {navigator.gpu.requestAdapter=native;}
  const {gpu}=await getDebug();gpu.device.destroy();await gpu.device.lost;
  checks.deviceLossError=await rejects(()=>gpu.run([normalize('كتب')]));debug=undefined;
  return {checks,passed:Object.values(checks).every(Boolean)};
}
async function deterministic(words:string[],runs=100) {
  const baseline=(await publicRun(words)).outputHash;const hashes:string[]=[];
  for(let i=0;i<runs;i++) hashes.push((await publicRun(words)).outputHash);
  return {runs,deterministic:hashes.every(h=>h===baseline),hash:baseline,hashes};
}
function environment() {
  const p=performance as Performance&{memory?:{usedJSHeapSize:number;totalJSHeapSize:number;jsHeapSizeLimit:number}};
  return {userAgent:navigator.userAgent,secureContext:isSecureContext,adapter:lastAdapter,visibility:document.visibilityState,devicePixelRatio,screen:{width:screen.width,height:screen.height},heap:p.memory?{usedBytes:p.memory.usedJSHeapSize,totalBytes:p.memory.totalJSHeapSize,limitBytes:p.memory.jsHeapSizeLimit,api:'performance.memory (Chromium-specific)'}:null,powerState:null,pluggedIn:null,measureUserAgentSpecificMemory:null};
}
async function diagnostics(words:string[]) {const {gpu}=await getDebug();await gpu.run(words.map(normalize));return gpu.metadata();}
export const lab={publicRun,parity,referenceParity,contracts,deterministic,environment,diagnostics};
declare global {interface Window {lab:typeof lab}}
window.lab=lab;
