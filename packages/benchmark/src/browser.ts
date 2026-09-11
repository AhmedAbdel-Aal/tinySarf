import {selectedModel} from './selection';
import {chromium,type Page} from 'playwright';
import {readFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {parseArgs} from 'node:util';
import path from 'node:path';
import {serveBenchmark} from './server';
import {ROOT,json,hash,provenance,writeArtifact} from './artifact';
import {stats,random,warmupStable} from './math';
import type {lab} from '../browser/harness';

declare global {interface Window {lab:typeof lab}}
const query=(command:string,args:string[])=>{try{return execFileSync(command,args,{encoding:'utf8',stdio:['ignore','pipe','ignore']}).trim();}catch{return null;}};
export async function browserBenchmark(smoke=false,onlyParity=false,local=false,workloadsFilter?:string) {
 const selection=await selectedModel(),model=await json(path.join(selection.directory,'manifest.json'));
 const corpusFile=path.join(ROOT,'packages/benchmark/performance-corpus.json'),corpus=await json(corpusFile),fixtures=await json(path.join(ROOT,'packages/training/data/contract-fixtures.json'));
 const allWorkloadIds=corpus.workloads.map((w:any)=>w.id);
 if(workloadsFilter){const ids=workloadsFilter.split(',');if(ids.some(id=>!allWorkloadIds.includes(id)))throw new Error('Unknown workload');corpus.workloads=corpus.workloads.filter((w:any)=>ids.includes(w.id));}
 const inputs=await json(path.join(selection.directory,'parity-inputs.json'));
 const settings={coldRuns:smoke?3:20,warmups:smoke?3:20,iterations:smoke?5:100,repeats:smoke?1:3,seed:42};
 const server=await serveBenchmark();const executable=process.env.BROWSER_PATH??'/Applications/Brave Browser.app/Contents/MacOS/Brave Browser';
 const browser=await chromium.launch({executablePath:executable,headless:true});
 const errors:string[]=[];
 const configure=async(page:Page)=>{page.on('pageerror',e=>errors.push(e.message));await page.goto(server.url);await page.waitForFunction(()=>!!window.lab);};
 const source=provenance(),runtime={sha256:hash(await readFile(path.join(ROOT,'packages/core/dist/index.js'))),modelModuleSha256:hash(await readFile(path.join(ROOT,'packages/core/dist/checkpoint.js')))};
 try {
  const page=await browser.newPage();await configure(page);
  console.log('Comparing Python-exported int8 reference and browser CPU tensors');
  const pythonCPU=await page.evaluate(()=>window.lab.referenceParity());
  console.log('Comparing quantized CPU/WebGPU tensors on frozen inputs');
  const parity=await page.evaluate(words=>window.lab.parity(words,true),inputs.words);
  const fixtureParity=await page.evaluate(words=>window.lab.parity(words),fixtures.map((f:any)=>f.word));
  console.log('Testing minimum, irregular and maximum shapes, error paths, and device recovery');
  const shapes=[];
  for(const words of [['ك'],['كتب','كتاب','المدرسة'],Array(17).fill('وبكتابهم'),Array(257).fill('كتب')]) shapes.push(await page.evaluate(words=>window.lab.parity(words),words));
  const contracts=await page.evaluate(()=>window.lab.contracts());
  console.log('Checking determinism over 100 warm calls');
  const deterministic=await page.evaluate(()=>window.lab.deterministic(['كتب','وبكتابهم','المدرسة'],100));
  const parityEnvironment=await page.evaluate(()=>window.lab.environment());
  const pythonCPUPassed=Object.values(pythonCPU).every(t=>t.finite && t.argmaxAgreement!==null && t.argmaxAgreement>=.9999 && t.maxAbsoluteError<=1e-4);
  const parityArtifact={...source,kind:'parity',runtime,status:'experimental-unpromoted',model,data:{verificationDigest:model.verificationDigest,parityInputDigest:hash(await readFile(path.join(selection.directory,'parity-inputs.json'))),fixtureDigest:hash(await readFile(path.join(ROOT,'packages/training/data/contract-fixtures.json'))),goldDigest:null,normalizationVersion:'arabic-v1'},environment:{...source.environment,browser:browser.version(),...parityEnvironment},pythonCPU,parity,fixtureParity,shapes,contracts,deterministic,errors,passed:pythonCPUPassed && parity.finite && parity.headArgmaxMatches/parity.headArgmaxCount>=.9999 && parity.completeAnalysisMatches===parity.completeAnalysisCount && fixtureParity.completeAnalysisMatches===fixtureParity.completeAnalysisCount && shapes.every(s=>s.completeAnalysisMatches===s.completeAnalysisCount) && contracts.passed && deterministic.deterministic && !errors.length};
  const parityFile=await writeArtifact('parity',parityArtifact,local?'packages/benchmark/local':'packages/benchmark/results');console.log(`PARITY_ARTIFACT=${parityFile}`);
  if(!parityArtifact.passed) throw new Error('Browser parity/contract gate failed; disagreements retained in artifact');
  await page.close();if(onlyParity)return {parityFile};
  const cold:Record<string,unknown[]>={webgpu:[],cpu:[]};
  for(const backend of ['webgpu','cpu'] as const) for(let i=0;i<settings.coldRuns;i++) {
   const context=await browser.newContext({serviceWorkers:'block'}),page=await context.newPage();await configure(page);
   const sample=await page.evaluate(backend=>window.lab.publicRun(['وبكتابهم'],backend),backend);
   const resources=await page.evaluate(()=>performance.getEntriesByType('resource').filter(e=>e.name.includes('/tinysarf/')).map(entry=>{const e=entry as PerformanceResourceTiming;return {url:new URL(e.name).pathname,durationMs:e.duration,transferBytes:e.transferSize,encodedBodyBytes:e.encodedBodySize,decodedBodyBytes:e.decodedBodySize};}));
   cold[backend].push({run:i,...sample,outputs:undefined,resources,totalImportAndAPI:sample.importMs+sample.apiMs});await context.close();
   console.log(`Cold ${backend} ${i+1}/${settings.coldRuns}`);
  }
  const context=await browser.newContext({serviceWorkers:'block'}),warmPage=await context.newPage();await configure(warmPage);
  // Initialize both public backends before warmups; all measured calls use the production package entry.
  await warmPage.evaluate(async()=>{await window.lab.publicRun(['كتب'],'webgpu');await window.lab.publicRun(['كتب'],'cpu');});
  const raw:{repeat:number;workload:string;backend:'webgpu'|'cpu';samples:number[];hashes:string[];warmupSamples:number[];stable:boolean}[]=[];
  const rng=random(settings.seed);
  for(let repeat=0;repeat<settings.repeats;repeat++) {
   const workloads=[...corpus.workloads];for(let i=workloads.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[workloads[i],workloads[j]]=[workloads[j],workloads[i]];}
   for(const workload of workloads) {
    const backends=repeat%2?['cpu','webgpu'] as const:['webgpu','cpu'] as const;
    for(const backend of backends) {
     const record={repeat,workload:workload.id,backend,samples:[] as number[],hashes:[] as string[],warmupSamples:[] as number[],stable:false};
     for(let i=0;i<(smoke?settings.warmups:100);i++) {
      const result=await warmPage.evaluate(({words,backend})=>window.lab.publicRun(words,backend),{words:workload.words,backend});
      record.warmupSamples.push(result.apiMs);
      if(i+1>=settings.warmups && (smoke || warmupStable(record.warmupSamples))){record.stable=!smoke;break;}
     }
     for(let i=0;i<settings.iterations;i++) {
      const result=await warmPage.evaluate(({words,backend})=>window.lab.publicRun(words,backend),{words:workload.words,backend});
      record.samples.push(result.apiMs);record.hashes.push(result.outputHash);
     }
     raw.push(record);console.log(`Warm ${repeat+1}/${settings.repeats} ${workload.id} ${backend}: p50 ${stats(record.samples)!.p50.toFixed(2)} ms`);
    }
   }
  }
  const warm=corpus.workloads.map((workload:any)=>{
   const group=(backend:string)=>raw.filter(r=>r.workload===workload.id&&r.backend===backend);
   const gpu=group('webgpu'),cpu=group('cpu'),gpuStats=stats(gpu.flatMap(r=>r.samples))!,cpuStats=stats(cpu.flatMap(r=>r.samples))!;
   const allHashes=[...gpu,...cpu].flatMap(r=>r.hashes),outputHashesMatch=new Set(allHashes).size===1;
   return {id:workload.id,batch:workload.words.length,characters:workload.characters,lengthClass:workload.lengthClass,paddingWaste:workload.paddingWaste,synthetic:workload.synthetic??false,inputDigest:workload.sha256,webgpu:gpuStats,cpu:cpuStats,speedup:cpuStats.p50/gpuStats.p50,webgpuWordsPerSecond:workload.words.length/(gpuStats.p50/1000),webgpuCharactersPerSecond:workload.characters/(gpuStats.p50/1000),outputHashesMatch,outputHash:allHashes[0]};
  });
  const diagnostic=await warmPage.evaluate(words=>window.lab.diagnostics(words),Array(2048).fill('ك'.repeat(32)));
  const environment=await warmPage.evaluate(()=>window.lab.environment());
  const power=query('pmset',['-g','batt']);
  const artifact={...provenance(),kind:'browser',runtime,status:'experimental-unpromoted',model,data:{verificationDigest:model.verificationDigest,goldDigest:null,performanceCorpusDigest:hash(await readFile(corpusFile)),normalizationVersion:'arabic-v1'},environment:{...source.environment,deviceModel:query('sysctl',['-n','hw.model']),osVersion:query('sw_vers',['-productVersion']),browser:browser.version(),browserExecutable:executable,headless:true,...environment,powerState:power,pluggedIn:power?power.includes('AC Power'):null},protocol:{...settings,mode:smoke?'smoke':'full',releaseQualifying:!smoke && !workloadsFilter && raw.every(r=>r.stable) && !errors.length,shaderCompilationIncluded:true,cacheState:'Fresh contexts for cold runs; HTTP no-store; driver shader cache not controlled',networkScope:'Loopback static server, not internet transfer latency',devToolsOpen:false,warmupStabilityAssessed:!smoke,warmupStabilityPassed:raw.every(r=>r.stable),warmupRule:'Two consecutive windows of 10 calls: median shift <=5%, CV <=10%; minimum20, maximum100',workloadSubset:!!workloadsFilter,availableWorkloads:allWorkloadIds,workloadRotation:'Seeded shuffle per repeat; backend order alternates across repeats'},cold:{raw:cold,summary:Object.fromEntries(Object.entries(cold).map(([backend,samples])=>[backend,{apiMs:stats(samples.map((s:any)=>s.apiMs)),moduleImportMs:stats(samples.map((s:any)=>s.importMs)),totalMs:stats(samples.map((s:any)=>s.totalImportAndAPI))}]))},warm,raw,diagnostic,parityArtifact:path.basename(parityFile),errors,limitations:['One headless Chromium browser on one integrated GPU','Independent discrete GPU, Android, Safari and Firefox measurements are unavailable',...(!raw.every(r=>r.stable)?['Warmup stability was not established for all workload/backend pairs']:[]),...workloadsFilter?['Only the explicitly selected workload subset was timed']:[],'Reference CPU is straightforward JavaScript, not optimized WASM','Timing runs are experimental; a controlled release-device run is still required']};
  const file=await writeArtifact('browser',artifact,local?'packages/benchmark/local':'packages/benchmark/results');await context.close();console.log(`BROWSER_ARTIFACT=${file}`);
  if(warm.some((w:any)=>!w.outputHashesMatch)) throw new Error('CPU/GPU warm output hashes differ');return {parityFile,file};
 } finally {await browser.close();await server.close();}
}
if(import.meta.url===`file://${process.argv[1]}`){const {values}=parseArgs({options:{smoke:{type:'boolean',default:false},'parity-only':{type:'boolean',default:false},local:{type:'boolean',default:false},workloads:{type:'string'}}});await browserBenchmark(values.smoke,values['parity-only'],values.local,values.workloads);}
