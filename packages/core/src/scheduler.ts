import {MAX_BATCH,normalize} from "./normalize";
import type {LoadedModel} from "./model";
import type {Backend} from "./cpu";
import {decode} from "./decode";
import type {AnalyzeOptions,MorphAnalysis} from "./schema";
export interface Analyze {
  (word:string,options?:AnalyzeOptions):Promise<MorphAnalysis[]>;
  batch(words:readonly string[],options?:AnalyzeOptions):Promise<MorphAnalysis[][]>;
}
export function createAnalyze(load:(backend:"cpu"|"webgpu")=>Promise<{model:LoadedModel;backend:Backend}>):Analyze {
  let tail:Promise<unknown>=Promise.resolve();
  async function batch(words:readonly string[],options:AnalyzeOptions={}):Promise<MorphAnalysis[][]> {
    if(!Array.isArray(words)) throw new TypeError("Expected an array of Arabic words");
    if(words.length>MAX_BATCH) throw new RangeError(`Maximum batch size is ${MAX_BATCH}`);
    if(!options || typeof options!=="object") throw new TypeError("Expected analyze options");
    const topK=options.topK??1,backend=options.backend??"webgpu",threshold=options.threshold;
    if(!Number.isInteger(topK) || topK<1 || topK>3) throw new RangeError("topK must be 1, 2, or 3");
    if(backend!=="webgpu" && backend!=="cpu") throw new TypeError("backend must be webgpu or cpu");
    if(threshold!==undefined && !Number.isFinite(threshold)) throw new RangeError("threshold must be finite");
    const prepared=words.map(word=>normalize(word));
    if(!prepared.length) return [];
    const task=tail.then(async()=>{
      const runtime=await load(backend),result=await runtime.backend.run(prepared);
      return prepared.map((word,i)=>decode(word,result.words[i],runtime.model.manifest,topK,threshold));
    });
    tail=task.catch(()=>{}); return task;
  }
  const analyze=async(word:string,options?:AnalyzeOptions)=>(await batch([word],options))[0];
  analyze.batch=batch; return analyze;
}
