import type {LoadedModel} from "./model";
import type {NormalizedWord} from "./normalize";
import {rootCPU} from './root-cpu';
export interface Logits {segmentation:Float32Array;heads:Record<string,Float32Array>}
export interface InferenceResult {words:Logits[];trace?:Record<string,Float32Array>}
export interface Backend {
  run(words:NormalizedWord[], trace?:boolean):Promise<InferenceResult>;
  dispose():void;
  metadata():Record<string,unknown>;
}
export class CPUBackend implements Backend {
  constructor(readonly model:LoadedModel) {}
  async run(words:NormalizedWord[],debug=false):Promise<InferenceResult> {
    const {manifest:m,tensors:t}=this.model; const results:Logits[]=[]; const traces:Record<string,number[]>={};
    const record=(name:string,array:Float32Array)=>{ if(debug) (traces[name]??=[]).push(...array); };
    for(const word of words) {
      const length=word.text.length; let channels=m.embedding; let x=new Float32Array(length*channels);
      for(let p=0;p<length;p++) for(let c=0;c<channels;c++) x[p*channels+c]=t.embedding[(word.packed[p]&255)*channels+c]+t.position[p*channels+c];
      record("embedding",x);
      for(let layer=0;layer<3;layer++) {
        const y=new Float32Array(length*m.width),weight=t[`conv${layer}.weight`],bias=t[`conv${layer}.bias`],dilation=m.dilations[layer];
        for(let p=0;p<length;p++) for(let o=0;o<m.width;o++) {
          let sum=0;
          for(let k=0;k<3;k++) { const pos=p+(k-1)*dilation; if(pos<0 || pos>=length) continue;
            const a=pos*channels,b=(o*3+k)*channels;
            for(let c=0;c<channels;c++) sum+=x[a+c]*weight[b+c];
          }
          y[p*m.width+o]=Math.max(0,sum+bias[o]);
        }
        x=y; channels=m.width; record(`conv${layer}`,x);
      }
      const pooled=new Float32Array(m.width);
      for(let c=0;c<m.width;c++) {let sum=0; for(let p=0;p<length;p++) sum+=x[p*m.width+c]; pooled[c]=sum/length;}
      record("pooled",pooled);
      const heads:Logits["heads"]={}; let segmentation=new Float32Array(0);
      for(const [name,labels] of Object.entries({segmentation:m.labels.segmentation,...m.labels.heads})) {
        if(m.rootArchitecture&&name.startsWith('root'))continue;
        const rows=name==="segmentation"?length:1,weight=t[`${name}.weight`],bias=t[`${name}.bias`],input=name==="segmentation"?x:pooled;
        const out=new Float32Array(rows*labels.length);
        for(let p=0;p<rows;p++) for(let o=0;o<labels.length;o++) { let sum=0;
          for(let c=0;c<m.width;c++) sum+=input[p*m.width+c]*weight[o*m.width+c];
          out[p*labels.length+o]=sum+bias[o];
        }
        record(name,out); if(name==="segmentation") segmentation=out; else heads[name]=out;
      }
      if(m.rootArchitecture)Object.assign(heads,rootCPU(word,this.model,record));
      results.push({segmentation,heads});
    }
    return {words:results,...(debug?{trace:Object.fromEntries(Object.entries(traces).map(([k,v])=>[k,Float32Array.from(v)]))}:{})};
  }
  dispose() {}
  metadata() {return {backend:"cpu",gpuAllocatedBytes:0};}
}
