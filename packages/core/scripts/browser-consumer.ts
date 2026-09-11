import {createServer} from "node:http";
import {readFile} from "node:fs/promises";
import path from "node:path";
import {chromium,type Browser} from "playwright";

/** Consumer integration only: explicit CPU inference does not establish WebGPU support or hardware performance. */
export async function checkBrowserConsumer(packageDirectory:string) {
  const assets=new Map<string,Buffer>(await Promise.all(["index.js","checkpoint.js"].map(async filename=>[
    `/tinysarf/${filename}`,await readFile(path.join(packageDirectory,"dist",filename)),
  ] as const)));
  const requested=new Set<string>();
  const server=createServer((request,response)=>{
    const pathname=new URL(request.url??"/","http://127.0.0.1").pathname;
    response.setHeader("Cache-Control","no-store");
    const asset=assets.get(pathname);
    if(asset) {
      requested.add(pathname);
      response.writeHead(200,{"Content-Type":"text/javascript; charset=utf-8"});
      response.end(asset);
    } else if(pathname==="/") {
      response.writeHead(200,{"Content-Type":"text/html; charset=utf-8"});
      response.end('<!doctype html><meta charset="utf-8"><title>TinySarf package consumer smoke</title>');
    } else {
      response.writeHead(404);response.end("Not found");
    }
  });
  await new Promise<void>((resolve,reject)=>{
    server.once("error",reject);
    server.listen(0,"127.0.0.1",()=>{server.off("error",reject);resolve();});
  });
  let browser:Browser|undefined;
  try {
    const executable=process.env.BROWSER_PATH;
    try {
      browser=await chromium.launch({headless:true,...(executable?{executablePath:executable}:{})});
    } catch(error) {
      throw new Error("Could not launch the package-smoke browser. Run pnpm exec playwright install chromium, or set BROWSER_PATH to a Chromium executable.",{cause:error});
    }
    const address=server.address();
    if(!address||typeof address==="string")throw new Error("Package smoke server has no TCP address");
    const origin=`http://127.0.0.1:${address.port}`;
    const page=await browser.newPage();
    const pageErrors:string[]=[];
    page.on("pageerror",error=>pageErrors.push(error.message));
    const responses=new Map<string,number>();
    page.on("response",response=>{
      const pathname=new URL(response.url()).pathname;
      if(assets.has(pathname))responses.set(pathname,response.status());
    });
    await page.goto(origin,{waitUntil:"load"});
    const result=await page.evaluate(async moduleURL=>{
      const {analyze}=await import(moduleURL);
      // Object methods avoid tsx's module-scoped function-name helper in this serialized callback.
      const checks={validate(word:string,candidates:any[]) {
        if(!Array.isArray(candidates)||candidates.length<1||candidates.length>3)throw new Error("Missing package analysis");
        for(const candidate of candidates) {
          if(!Array.isArray(candidate.spans)||!Number.isFinite(candidate.score))throw new Error("Invalid package result");
          let end=0,stem=false;
          for(const span of candidate.spans) {
            if(!Number.isInteger(span.start)||!Number.isInteger(span.end)||span.start!==end||span.end<=span.start||span.end>word.length)throw new Error("Invalid original-input span coverage");
            if(span.end<word.length&&/^[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]$/u.test(word[span.end]))throw new Error("A package span splits an attached mark");
            end=span.end;stem||=span.type==="stem";
          }
          if(end!==word.length||!stem)throw new Error("Incomplete package spans");
        }
      }};
      const options={backend:"cpu",topK:3};
      const word="وَبِكِتَابِهِمْ",words=["كتب",word,"مدرسة"];
      const single=await analyze(word,options);
      checks.validate(word,single);
      const batch=await analyze.batch(words,options);
      if(!Array.isArray(batch)||batch.length!==words.length)throw new Error("Package batch length changed");
      batch.forEach((candidates:any[],index:number)=>checks.validate(words[index],candidates));
      if(JSON.stringify(batch[1])!==JSON.stringify(single))throw new Error("Single and batch package results disagree");
      const empty=await analyze.batch([],options);
      if(!Array.isArray(empty)||empty.length)throw new Error("Empty package batch changed");
      let invalidRejected=false;
      try{await analyze.batch(["كتب","hello"],options);}catch{invalidRejected=true;}
      if(!invalidRejected)throw new Error("Malformed package input was accepted");
      return {singleCandidates:single.length,batchWords:batch.length,emptyBatch:true,invalidInputRejected:true};
    },`${origin}/tinysarf/index.js`);
    if(pageErrors.length)throw new Error(`Package browser errors: ${pageErrors.join("; ")}`);
    for(const asset of assets.keys())if(!requested.has(asset)||responses.get(asset)!==200)throw new Error(`Browser did not load packaged module ${asset}`);
    return {kind:"package-consumer-smoke",backend:"cpu",hardwareEvidence:false,browser:browser.version(),browserSource:executable?"BROWSER_PATH":"playwright-installed-chromium",modules:[...assets.keys()],...result};
  } finally {
    try{await browser?.close();}
    finally {
      server.closeAllConnections();
      await new Promise<void>((resolve,reject)=>server.close(error=>error?reject(error):resolve()));
    }
  }
}
