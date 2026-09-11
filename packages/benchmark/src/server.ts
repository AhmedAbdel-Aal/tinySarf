import {selectedModel} from './selection';
import {build} from 'esbuild';
import {createServer} from 'node:http';
import {readFile,mkdtemp} from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {ROOT,json} from './artifact';
export async function serveBenchmark(modelDirectory?:string) {
 const selected=await selectedModel(modelDirectory);
 const directory=selected.directory;
 const temporary=await mkdtemp(path.join(os.tmpdir(),'tinysarf-browser-'));
 await build({entryPoints:[path.join(ROOT,'packages/benchmark/browser/harness.ts')],outfile:path.join(temporary,'harness.js'),bundle:true,format:'esm',target:'es2022',platform:'browser'});
 const routes:Record<string,[string,string]>={
  '/harness.js':[path.join(temporary,'harness.js'),'text/javascript'],
  '/tinysarf/index.js':[path.join(ROOT,'packages/core/dist/index.js'),'text/javascript'],
  '/tinysarf/checkpoint.js':[path.join(ROOT,'packages/core/dist/checkpoint.js'),'text/javascript'],
  '/manifest.json':[path.join(directory,'manifest.json'),'application/json'],
  '/weights.bin':[path.join(directory,'weights.bin'),'application/octet-stream'],
  '/parity.json':[path.join(directory,'parity-inputs.json'),'application/json'],
  '/performance.json':[path.join(ROOT,'packages/benchmark/performance-corpus.json'),'application/json'],
 };
 const server=createServer(async(req,res)=>{
  res.setHeader('Cache-Control','no-store');res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
  const route=new URL(req.url??'/','http://localhost').pathname;
  if(route==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><html lang="en"><meta charset="UTF-8"><title>TinySarf browser benchmark</title><body><h1>TinySarf browser benchmark</h1><p>Real local inference. Results are collected by the benchmark runner.</p><script type="module" src="/harness.js"></script></body></html>');return;}
  const file=routes[route];if(!file){res.writeHead(404);res.end('Not found');return;}
  try{res.setHeader('Content-Type',file[1]);res.end(await readFile(file[0]));}catch{res.writeHead(500);res.end('Benchmark artifact unavailable');}
 });
 await new Promise<void>((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
 const address=server.address();if(!address||typeof address==='string')throw new Error('No benchmark address');
 return {url:`http://127.0.0.1:${address.port}`,close:()=>new Promise<void>((resolve,reject)=>server.close(error=>error?reject(error):resolve()))};
}
