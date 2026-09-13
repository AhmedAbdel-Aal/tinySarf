import {execFileSync,spawn} from 'node:child_process';
import {mkdtemp,mkdir,readFile,writeFile,copyFile,readdir} from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {parseArgs} from 'node:util';
import {ROOT,hash,json,provenance} from '../packages/benchmark/src/artifact';
import {selectedModel} from '../packages/benchmark/src/selection';
const source=provenance();if(!source.git.commit || source.git.dirty!==false)throw new Error('Commit source and evidence before clean-clone reproduction');
const {values}=parseArgs({options:{checkout:{type:'string'}}});
const temporary=await mkdtemp(path.join(os.tmpdir(),'tinysarf-reproduce-')),checkout=values.checkout?path.resolve(values.checkout):path.join(temporary,'checkout');
if(values.checkout){
 const top=execFileSync('git',['rev-parse','--show-toplevel'],{cwd:checkout,encoding:'utf8'}).trim();
 const remote=execFileSync('git',['remote','get-url','origin'],{cwd:checkout,encoding:'utf8'}).trim();
 const dirty=execFileSync('git',['status','--porcelain'],{cwd:checkout,encoding:'utf8'}).trim();
 if(path.resolve(top)!==checkout || path.resolve(remote)!==ROOT || dirty || checkout===ROOT)throw new Error('Reuse requires a clean separate checkout cloned from this source repository');
 execFileSync('git',['fetch','--quiet','origin'],{cwd:checkout,stdio:'inherit'});
}else execFileSync('git',['clone','--no-hardlinks','--quiet',ROOT,checkout],{stdio:'inherit'});
execFileSync('git',['checkout','--detach',source.git.commit],{cwd:checkout,stdio:'inherit'});
const logs=path.join(temporary,'logs');await mkdir(logs,{recursive:true});
const frozenIndex=await json(path.join(ROOT,'packages/benchmark/reports/current.json'));
const commands:string[][]=[];
async function run(args:string[]) {
 commands.push(args);const number=commands.length;console.log(`Reproduction ${number}: ${args.join(' ')}`);
 const log:string[]=[];await new Promise<void>((resolve,reject)=>{
  const child=spawn(args[0],args.slice(1),{cwd:checkout,env:{...process.env,CI:'true'},stdio:['ignore','pipe','pipe']});
  child.stdout.on('data',d=>{process.stdout.write(d);log.push(d.toString());});child.stderr.on('data',d=>{process.stderr.write(d);log.push(d.toString());});
  child.on('error',reject);child.on('close',async code=>{await writeFile(path.join(logs,`${number}.log`),log.join(''));code===0?resolve():reject(new Error(`Reproduction command failed (${code}): ${args.join(' ')}`));});
 });
}
let passed=false,error:string|null=null;const artifacts:any={};
try {
 await run(['bash','scripts/setup.sh']);
 // Reuse only downloaded upstream source bytes, still checked against the pinned SHA-256 by the compiler.
 const cache=path.join(ROOT,'packages/training/data/downloads'),destination=path.join(checkout,'packages/training/data/downloads');await mkdir(destination,{recursive:true});
 for(const file of ['camel_morph_msa_v1.0.db.gz','camel-tools.tar.gz']){try{await copyFile(path.join(cache,file),path.join(destination,file));}catch{ /* compiler downloads any missing source */ }}
 await run(['pnpm','data:prepare']);
 await run(['.venv/bin/python','-m','tinysarf_training.root_data','--selected']);
 await run(['pnpm','test']);await run(['pnpm','build:core']);await run(['pnpm','check:package']);
 await run(['node','--import','tsx','packages/benchmark/src/correctness.ts','--local']);
 await run(['node','--import','tsx','packages/benchmark/src/size.ts','--local']);
 await run(['node','--import','tsx','packages/benchmark/src/browser.ts','--local','--parity-only']);
 if(frozenIndex.supplemental?.['ai-review'])await run(['pnpm','benchmark:ai-review','--local']);
 await run(['pnpm','docs:generate']);await run(['npm','--prefix','apps/website','run','build']);
 const resultDir=path.join(checkout,'packages/benchmark/local'),index=frozenIndex;
 for(const kind of ['correctness','size','parity',...(index.supplemental?.['ai-review']?['ai-review']:[])]){
  const filename=(await readdir(resultDir)).filter(f=>f.startsWith(kind+'-')).sort().at(-1)!;const bytes=await readFile(path.join(resultDir,filename)),actual=JSON.parse(bytes.toString());
  if(actual.git.commit!==source.git.commit || actual.git.dirty!==false)throw new Error('Reproduction changed tracked source');
  const expected=await json(path.join(ROOT,(index.artifacts[kind]??index.supplemental[kind]).file));
  if(kind==='correctness'){
   if(JSON.stringify(actual.predictions)!==JSON.stringify(expected.predictions) || JSON.stringify(actual.correctness.teacherAgreement)!==JSON.stringify(expected.correctness.teacherAgreement))throw new Error('Frozen correctness predictions/metrics were not reproduced');
  }else if(kind==='size'){
   for(const field of ['packedWeightsBytes','javascriptMinifiedBytes','npmUnpackedBytes'])if(actual.size[field]!==expected.size[field])throw new Error(`Size changed: ${field}`);
   if(actual.runtime.sha256!==expected.runtime.sha256 || actual.size.completePackage.brotli!==expected.size.completePackage.brotli)throw new Error('Frozen runtime/package bytes were not reproduced');
  }else if(kind==='ai-review'){
   if(JSON.stringify(actual.predictions)!==JSON.stringify(expected.predictions)||JSON.stringify(actual.metrics)!==JSON.stringify(expected.metrics))throw new Error('AI diagnostic predictions/metrics were not reproduced');
  }else if(!actual.passed)throw new Error('Clean-clone browser parity failed');
  artifacts[kind]={file:filename,sha256:hash(bytes)};
 }
 const dirty=execFileSync('git',['status','--porcelain'],{cwd:checkout,encoding:'utf8'}).trim();if(dirty)throw new Error(`Clean reproduction modified tracked files: ${dirty}`);
 passed=true;
}catch(e){error=String(e);}
const selection=await selectedModel(),model=await json(path.join(selection.directory,'manifest.json'));
let runtimeSha256:string|null=null;try{runtimeSha256=hash(await readFile(path.join(checkout,'packages/core/dist/index.js')));}catch{}
const report={...source,kind:'reproduction',runtime:{sha256:runtimeSha256},separateCheckout:true,checkout,modelSha256:model.sha256,model,data:{verificationDigest:model.verificationDigest,goldDigest:null},passed,error,commands:commands.map(c=>c.join(' ')),artifacts,upstreamCache:'Only pinned source archive/database bytes reused; dependencies installed from lockfiles; generated datasets reconstructed; final-test labels never evaluated'};
const out=path.join(ROOT,'packages/benchmark/local');await mkdir(out,{recursive:true});const file=path.join(out,`reproduction-${source.runId}.json`);await writeFile(file,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(`REPRODUCTION_ARTIFACT=${file}`);console.log(`REPRODUCTION_CHECKOUT=${checkout}`);if(!passed)throw new Error(error!);
