import { LETTERS } from "./normalize";
import { NORMALIZATION_VERSION, SPAN_TYPES, FEATURE_VALUES, POS_TYPES } from "./schema";
export interface TensorSpec {shape:number[];offset:number;length:number;scale:number}
export interface RootTemplate {surface:string;output:(number|string)[];distinctRoots:number}
export interface RootDecoder {format:"learned-transforms-v1";priorWeight:number;unsupportedPenalty:number;nullPenalty:number;minimumDistinctRoots:number;whole:RootTemplate[];stem:RootTemplate[]}
export interface ModelManifest {
  schemaVersion:1; format:"cnn-v1"|"cnn-root-v2"; id:string; status:string; normalizationVersion:string;
  quantization:"symmetric-per-tensor-int8"; vocabulary:string[];
  labels:{segmentation:string[];heads:Record<string,string[]>}; maxLength:32; embedding:number;
  width:number; dilations:number[]; trainingParameters:number; reachableWeights:number;
  packedWeightsBytes:number; sha256:string; floatSha256:string; floatBinarySha256:string; verificationDigest:string;
  tensors:Record<string,TensorSpec>;
  rootArchitecture?:{format:"copy-cnn-v1";embedding:16;width:number;dilations:number[]};
  rootDecoder?:RootDecoder;
}
export interface LoadedModel {manifest:ModelManifest;packed:Uint8Array;tensors:Record<string,Float32Array>}
export function validateManifest(m:ModelManifest,byteLength:number):void {
  if(m.schemaVersion!==1 || !["cnn-v1","cnn-root-v2"].includes(m.format) || m.normalizationVersion!==NORMALIZATION_VERSION || m.quantization!=="symmetric-per-tensor-int8") throw new Error("Unsupported TinySarf model contract");
  const root=m.rootArchitecture;
  if((m.format==="cnn-root-v2")!==!!root || root&&(root.format!=="copy-cnn-v1" || root.embedding!==16 || !Number.isInteger(root.width) || root.width<1 || root.width>256 || JSON.stringify(root.dilations)!=="[1,2,4]")) throw new Error("Invalid root architecture");
  if(m.maxLength!==32 || m.embedding!==32 || !Number.isInteger(m.width) || m.width<1 || m.width>256 || JSON.stringify(m.dilations)!=="[1,2,4]") throw new Error("Invalid model architecture");
  if(JSON.stringify(m.vocabulary)!==JSON.stringify(["__pad__",...LETTERS]) || JSON.stringify(m.labels.segmentation)!==JSON.stringify(SPAN_TYPES)) throw new Error("Model label/vocabulary order mismatch");
  if(m.packedWeightsBytes!==byteLength || byteLength%4!==0 || !/^[a-f0-9]{64}$/.test(m.sha256)) throw new Error("Model weight length/hash mismatch");
  const names=["pos","root0","root1","root2","root3","pattern",...Object.keys(FEATURE_VALUES)];
  if(JSON.stringify(Object.keys(m.labels.heads))!==JSON.stringify(names)) {
    // JSON serialization may sort object keys; tensor order is explicit by name, class order is strict.
    if(Object.keys(m.labels.heads).length!==names.length || names.some(n=>!m.labels.heads[n])) throw new Error("Model head names mismatch");
  }
  for(const [name,classes] of Object.entries(m.labels.heads)) {
    if(classes.length<2 || classes.length>256 || new Set(classes).size!==classes.length) throw new Error("Invalid head class order");
    if(name.startsWith("root") && JSON.stringify(classes)!==JSON.stringify(["__blank__",...LETTERS])) throw new Error("Radical class order mismatch");
    if(name in FEATURE_VALUES && JSON.stringify(classes)!==JSON.stringify(["__missing__","__unknown__","__na__",...FEATURE_VALUES[name as keyof typeof FEATURE_VALUES]])) throw new Error("Feature class order mismatch");
  }
  if(JSON.stringify(m.labels.heads.pos)!==JSON.stringify(["__missing__","__unknown__","__na__",...POS_TYPES.filter(p=>p!=="unknown")])) throw new Error("POS class order mismatch");
  if(JSON.stringify(m.labels.heads.pattern.slice(0,3))!==JSON.stringify(["__missing__","__unknown__","__na__"])) throw new Error("Pattern missing-state order mismatch");
  const shapes:Record<string,number[]>={embedding:[m.vocabulary.length,32],position:[32,32]};
  for(let i=0;i<3;i++) { shapes[`conv${i}.weight`]=[m.width,3,i===0?32:m.width]; shapes[`conv${i}.bias`]=[m.width]; }
  for(const [name,classes] of Object.entries({segmentation:m.labels.segmentation,...m.labels.heads})) { shapes[`${name}.weight`]=[classes.length,root&&name.startsWith('root')?root.width:m.width]; shapes[`${name}.bias`]=[classes.length]; }
  if(root) {
    shapes['root.embedding']=[m.vocabulary.length,root.embedding];shapes['root.position']=[32,root.embedding];
    for(let i=0;i<3;i++){shapes[`root.conv${i}.weight`]=[root.width,3,i===0?root.embedding:root.width];shapes[`root.conv${i}.bias`]=[root.width];}
    shapes['root.pointer.weight']=[4,root.width];shapes['root.gate.weight']=[4,root.width];shapes['root.gate.bias']=[4];
  }
  if(m.rootDecoder) {
    const d=m.rootDecoder;
    if(d.format!=="learned-transforms-v1" || !Number.isInteger(d.minimumDistinctRoots) || d.minimumDistinctRoots<3 || [d.priorWeight,d.unsupportedPenalty,d.nullPenalty].some(x=>!Number.isFinite(x)||x<0||x>1000)) throw new Error("Invalid root decoder");
    for(const rules of [d.whole,d.stem]) {
      if(!Array.isArray(rules)||rules.length>10000)throw new Error("Invalid root transformations");
      const seen=new Set<string>();
      for(const rule of rules) {
        if(!rule || typeof rule.surface!=="string" || !rule.surface.length || rule.surface.length>32 || [...rule.surface].some(c=>c!=="*"&&!LETTERS.includes(c)) || !Array.isArray(rule.output) || ![3,4].includes(rule.output.length) || !Number.isSafeInteger(rule.distinctRoots) || rule.distinctRoots<d.minimumDistinctRoots || rule.distinctRoots>1e8) throw new Error("Invalid root transformation");
        const copied=rule.output.filter((x):x is number=>typeof x==='number');
        if(new Set(copied).size<2 || copied.some(x=>!Number.isInteger(x)||x<0||x>=rule.surface.length||rule.surface[x]!=="*") || rule.output.some(x=>typeof x!=='number'&&(typeof x!=='string'||x.length!==1||!LETTERS.includes(x))) || [...rule.surface].some((c,i)=>c==='*'&&!copied.includes(i))) throw new Error("Invalid root transformation output");
        const key=JSON.stringify([rule.surface,rule.output]);if(seen.has(key))throw new Error("Duplicate root transformation");seen.add(key);
      }
    }
  }
  if(Object.keys(shapes).length!==Object.keys(m.tensors).length) throw new Error("Unexpected tensor names");
  const ranges:[number,number][]=[]; let count=0;
  for(const [name,shape] of Object.entries(shapes)) {
    const t=m.tensors[name];
    if(!t || JSON.stringify(t.shape)!==JSON.stringify(shape) || t.length!==shape.reduce((a,b)=>a*b,1) || !Number.isInteger(t.offset) || t.offset<0 || t.offset%4 || t.offset+t.length>byteLength || !Number.isFinite(t.scale) || t.scale<=0 || Math.fround(t.scale)!==t.scale || !Number.isFinite(Math.fround(t.scale*127))) throw new Error(`Invalid tensor ${name}`);
    ranges.push([t.offset,t.offset+Math.ceil(t.length/4)*4]); count+=t.length;
  }
  ranges.sort((a,b)=>a[0]-b[0]); let offset=0;
  for(const [start,end] of ranges) { if(start!==offset) throw new Error("Overlapping or unreachable packed weights"); offset=end; }
  if(offset!==byteLength || count-m.embedding-(root?.embedding??0)!==m.reachableWeights || count!==m.trainingParameters) throw new Error("Weight count mismatch");
}
export async function loadModel(manifest:ModelManifest,packed:Uint8Array):Promise<LoadedModel> {
  validateManifest(manifest,packed.byteLength);
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",packed.slice().buffer)),b=>b.toString(16).padStart(2,"0")).join("");
  if(hash!==manifest.sha256) throw new Error("TinySarf checkpoint SHA-256 mismatch");
  const signed=new Int8Array(packed.buffer,packed.byteOffset,packed.byteLength);
  const tensors:LoadedModel["tensors"]={};
  for(const [name,t] of Object.entries(manifest.tensors)) {
    const values=new Float32Array(t.length);
    for(let i=0;i<t.length;i++) values[i]=signed[t.offset+i]*t.scale;
    tensors[name]=values;
  }
  return {manifest,packed,tensors};
}
