import {selectedModel} from './selection';
import {build} from 'esbuild';
import {brotliCompressSync,gzipSync,gunzipSync,constants} from 'node:zlib';
import {execFileSync} from 'node:child_process';
import {readFile,mkdtemp} from 'node:fs/promises';
import {parseArgs} from 'node:util';
import path from 'node:path';
import os from 'node:os';
import {ROOT,json,hash,provenance,writeArtifact} from './artifact';
import {shaders} from '../../core/src/shaders';
const compressed=(bytes:Uint8Array)=>({raw:bytes.length,gzip:gzipSync(bytes,{level:9,mtime:0} as any).length,brotli:brotliCompressSync(bytes,{params:{[constants.BROTLI_PARAM_QUALITY]:11}}).length});
export async function sizeBenchmark(local=false) {
 execFileSync(process.execPath,['--import','tsx','scripts/build.ts'],{cwd:ROOT,stdio:'inherit'});
 const selection=await selectedModel(),dir=selection.directory,model=await json(path.join(dir,'manifest.json'));
 const temporary=await mkdtemp(path.join(os.tmpdir(),'tinysarf-size-'));
 const raw=await build({entryPoints:[path.join(ROOT,'packages/core/src/index.ts')],bundle:true,format:'esm',platform:'browser',target:'es2022',minify:false,external:['./checkpoint.js'],write:false});
 const wrapper=await readFile(path.join(ROOT,'packages/core/dist/index.js')),modelModule=await readFile(path.join(ROOT,'packages/core/dist/checkpoint.js')),metadata=await readFile(path.join(dir,'manifest.json')),weights=await readFile(path.join(dir,'weights.bin'));
 const packed=JSON.parse(execFileSync('npm',['pack','--json','--ignore-scripts','--pack-destination',temporary],{cwd:path.join(ROOT,'packages/core'),encoding:'utf8',env:{...process.env,npm_config_cache:path.join(temporary,'cache')}}))[0];
 const tarball=await readFile(path.join(temporary,packed.filename)),tar=gunzipSync(tarball);
 const wrapperSizes=compressed(wrapper),modelSizes=compressed(modelModule),packageSizes=compressed(tar);
 const size={trainingParameters:model.trainingParameters,reachableWeights:model.reachableWeights,packedWeightsBytes:weights.length,wgslSourceBytes:Object.values(shaders(model)).reduce((s,x)=>s+Buffer.byteLength(x),0),javascriptRawBytes:raw.outputFiles[0].contents.length,javascriptMinifiedBytes:wrapper.length,wrapper:wrapperSizes,modelModule:modelSizes,modelMetadata:compressed(metadata),modelBinary:compressed(weights),npmTarballBytes:tarball.length,npmUnpackedBytes:packed.unpackedSize,npmTarballSha256:hash(tarball),completePackage:packageSizes,wasmFallbackBytes:0,firstLoad:{webgpu:{raw:wrapper.length+modelModule.length,gzip:wrapperSizes.gzip+modelSizes.gzip,brotli:wrapperSizes.brotli+modelSizes.brotli},cpu:{raw:wrapper.length+modelModule.length,gzip:wrapperSizes.gzip+modelSizes.gzip,brotli:wrapperSizes.brotli+modelSizes.brotli}}};
 const policy=await json(path.join(ROOT,'packages/training/promotion-policy.json'));
 const budgetPassed=weights.length<policy.budgets.packedWeightsBytes && packageSizes.brotli<policy.budgets.brotliPackageBytes && wrapperSizes.brotli<policy.budgets.wrapperBrotliBytes;
 const artifact={...provenance(),kind:'size',status:'experimental-unpromoted',model,data:{verificationDigest:model.verificationDigest,goldDigest:null},runtime:{sha256:hash(wrapper),modelModuleSha256:hash(modelModule)},size,compression:{gzipLevel:9,brotliQuality:11,zlib:process.versions.zlib,brotli:process.versions.brotli,npm:queryNpmVersion(),esbuild:(await import('esbuild')).version},packageFiles:packed.files,budgetPassed,notes:['Raw wrapper is a non-minified build of the same production entry','First-load totals sum the separately compressed runtime and checkpoint modules','Complete package Brotli is Brotli of the exact uncompressed npm tar; npm distributes the measured gzip tarball','No WASM fallback is included; CPU reference is JavaScript','Generated workspace benchmark tables are excluded from the tarball to avoid circular size claims']};
 const file=await writeArtifact('size',artifact,local?'packages/benchmark/local':'packages/benchmark/results');console.log(JSON.stringify({file,size,budgetPassed}));if(!budgetPassed)throw new Error('Package size budget failed');return file;
}
function queryNpmVersion(){return execFileSync('npm',['--version'],{encoding:'utf8'}).trim();}
if(import.meta.url===`file://${process.argv[1]}`){const {values}=parseArgs({options:{local:{type:'boolean',default:false}}});await sizeBenchmark(values.local);}
