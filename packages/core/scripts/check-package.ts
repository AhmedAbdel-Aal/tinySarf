import {selectedModel} from '../../benchmark/src/selection';
import {readFile,mkdtemp,mkdir,writeFile,rename} from "node:fs/promises";
import {execFileSync} from "node:child_process";
import path from "node:path";
import os from "node:os";
import {brotliCompressSync,constants} from "node:zlib";
import {ROOT,json,hash} from "../../benchmark/src/artifact";
const core=path.join(ROOT,"packages/core"),pkg=await json(path.join(core,"package.json"));
if(Object.keys(pkg.exports).join()!==".") throw new Error("Only the primary API may be exported");
const module=await import(path.join(core,"dist/index.js"));
if(Object.keys(module).join()!=="analyze" || typeof module.analyze.batch!=="function") throw new Error("Unexpected public exports");
const selection=await selectedModel(),m=await json(path.join(selection.directory,"manifest.json"));
const bytes=await readFile(path.join(selection.directory,"weights.bin"));
if(hash(bytes)!==m.sha256 || bytes.length>=1048576) throw new Error("Checkpoint integrity/weight budget failed");
const wrapper=await readFile(path.join(core,"dist/index.js")),checkpoint=await readFile(path.join(core,"dist/checkpoint.js"));
const compress=(b:Uint8Array)=>brotliCompressSync(b,{params:{[constants.BROTLI_PARAM_QUALITY]:11}}).length;
if(compress(wrapper)>=102400 || compress(wrapper)+compress(checkpoint)>=1572864) throw new Error("Browser transfer budget exceeded");
const temporary=await mkdtemp(path.join(os.tmpdir(),"tinysarf-package-"));
const packed=JSON.parse(execFileSync("npm",["pack","--json","--ignore-scripts","--pack-destination",temporary],{cwd:core,encoding:"utf8",env:{...process.env,npm_config_cache:path.join(temporary,"cache")}}))[0];
if(packed.files.some((f:{path:string})=>/test|\.pt$|\.npz$|\.bin$|\.meta\.json$|\.map$/.test(f.path))) throw new Error("Unexpected development/training files in package");
const modules=path.join(temporary,"node_modules");await mkdir(modules,{recursive:true});
execFileSync("tar",["-xzf",path.join(temporary,packed.filename),"-C",temporary]);
await rename(path.join(temporary,"package"),path.join(modules,"tinysarf"));
await writeFile(path.join(temporary,"package.json"),JSON.stringify({name:"tinysarf-package-consumer",private:true,type:"module"}));
await writeFile(path.join(temporary,"smoke.mjs"),"import {analyze} from 'tinysarf'; const r=await analyze('كتب',{backend:'cpu',topK:3}); if(!r.length||!r[0].spans.length) throw Error('Missing analysis'); console.log('Tarball CPU API passed');");
execFileSync(process.execPath,[path.join(temporary,"smoke.mjs")],{cwd:temporary,stdio:"inherit"});
await writeFile(path.join(temporary,"usage.ts"),`import {analyze, type AnalyzeOptions, type MorphAnalysis} from 'tinysarf';
const options: AnalyzeOptions = {backend:'cpu',topK:3};
const r: MorphAnalysis[] = await analyze('كتب',options);
r[0].spans[0].start satisfies number;
await analyze.batch(['كتب','وَبِكِتَابِهِمْ'] as const,options);
// @ts-expect-error only documented backends are accepted
await analyze('كتب',{backend:'remote'});
// @ts-expect-error return values retain their declared shape
r[0].inventedField;
// @ts-expect-error no debug entry is publicly exported
import {GPUBackend} from 'tinysarf';
// @ts-expect-error package internals cannot bypass the public exports map
import {GPUBackend as InternalGPUBackend} from 'tinysarf/dist/gpu.js';
`);
for(const [module,moduleResolution] of [["ESNext","Bundler"],["NodeNext","NodeNext"]] as const) {
  const config=path.join(temporary,`tsconfig-${moduleResolution.toLowerCase()}.json`);
  await writeFile(config,JSON.stringify({compilerOptions:{module,moduleResolution,target:"ES2022",lib:["ES2022","DOM"],strict:true,skipLibCheck:false,noEmit:true,types:[]},files:["usage.ts"]}));
  execFileSync(process.execPath,[path.join(ROOT,"node_modules/typescript/bin/tsc"),"--project",config],{cwd:temporary,stdio:"inherit"});
  console.log(`Tarball strict ${moduleResolution} TypeScript API passed`);
}
let browserSmoke;
if(process.env.TINYSARF_PACKAGE_BROWSER==="1") {
  const {checkBrowserConsumer}=await import("./browser-consumer");
  browserSmoke=await checkBrowserConsumer(path.join(modules,"tinysarf"));
  console.log("Tarball browser CPU API and module-loading smoke passed (not hardware evidence)");
}
console.log(JSON.stringify({package:packed.filename,files:packed.files.length,tarballBytes:packed.size,wrapperBrotliBytes:compress(wrapper),firstLoadBrotliBytes:compress(wrapper)+compress(checkpoint),typeConsumers:["Bundler","NodeNext"],...(browserSmoke?{browserSmoke}:{})}));
