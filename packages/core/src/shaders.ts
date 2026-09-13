import type {ModelManifest} from "./model";
import {rootShaders} from './root-shaders';
const f=(n:number)=>`${n.toExponential(9)}f`;
export function shaders(m:ModelManifest):Record<string,string> {
  const prelude=`
struct Params { batch: u32, length: u32, unused0: u32, unused1: u32 }
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> source: array<f32>;
@group(0) @binding(3) var<storage, read_write> destination: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Params;
@group(0) @binding(5) var<storage, read> pool: array<f32>;
fn weight(offset: u32, scale: f32) -> f32 {
  let shift = (offset & 3u) * 8u;
  let value = i32((weights[offset >> 2u] >> shift) & 255u);
  return f32(select(value, value - 256, value >= 128)) * scale;
}
`;
  const t=m.tensors,w=m.width,e=m.embedding;
  const code:Record<string,string>={};
  code.embedding=prelude+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index=gid.x; let token=index/${e}u;
  if(token>=cfg.batch*cfg.length) { return; }
  let channel=index%${e}u; let id=packed[token]&255u;
  if(id==0u) { destination[index]=0.0; return; }
  destination[index]=weight(${t.embedding.offset}u+id*${e}u+channel,${f(t.embedding.scale)}) + weight(${t.position.offset}u+(token%cfg.length)*${e}u+channel,${f(t.position.scale)});
}`;
  for(let layer=0;layer<3;layer++) {
    const channels=layer===0?e:w,weight=t[`conv${layer}.weight`],bias=t[`conv${layer}.bias`],dilation=m.dilations[layer];
    code[`conv${layer}`]=prelude+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index=gid.x; let token=index/${w}u;
  if(token>=cfg.batch*cfg.length) { return; }
  if((packed[token]&255u)==0u) { destination[index]=0.0; return; }
  let channel=index%${w}u; let position=token%cfg.length; let word=token/cfg.length;
  var sum=0.0f;
  for(var k=0u;k<3u;k++) {
    let p=i32(position)+(i32(k)-1)*${dilation};
    if(p<0 || p>=i32(cfg.length)) { continue; }
    let a=(word*cfg.length+u32(p))*${channels}u;
    let b=${weight.offset}u+(channel*3u+k)*${channels}u;
    for(var c=0u;c<${channels}u;c++) { sum=sum+source[a+c]*weight(b+c,${f(weight.scale)}); }
  }
  destination[index]=max(0.0f,sum+weight(${bias.offset}u+channel,${f(bias.scale)}));
}`;
  }
  code.pool=prelude+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index=gid.x; let word=index/${w}u; let channel=index%${w}u;
  if(word>=cfg.batch) { return; }
  var sum=0.0f; var count=0u;
  for(var p=0u;p<cfg.length;p++) {
    if((packed[word*cfg.length+p]&255u)!=0u) { count++; }
    sum+=source[(word*cfg.length+p)*${w}u+channel];
  }
  destination[index]=sum/f32(max(count,1u));
}`;
  const heads=Object.entries(m.labels.heads),total=heads.reduce((s,[,v])=>s+v.length,0);let base=0;
  const branches=heads.map(([name,labels])=>{
    const wt=t[`${name}.weight`],bias=t[`${name}.bias`],start=base;base+=labels.length;
    if(m.rootArchitecture&&name.startsWith('root'))return `if(head>=${start}u && head<${base}u) { destination[index]=0.0f; return; }`;
    return `if(head>=${start}u && head<${base}u) { channel=head-${start}u; weightOffset=${wt.offset}u; biasOffset=${bias.offset}u; weightScale=${f(wt.scale)}; biasScale=${f(bias.scale)}; }`;
  }).join("\n");
  const seg=t['segmentation.weight'],segBias=t['segmentation.bias'];
  code.heads=prelude+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride=cfg.length*8u+${total}u;
  let index=gid.x; let word=index/stride; let slot=index%stride;
  if(word>=cfg.batch) { return; }
  var channel=0u;var weightOffset=${seg.offset}u;var biasOffset=${segBias.offset}u;
  var weightScale=${f(seg.scale)};var biasScale=${f(segBias.scale)};
  var sourceOffset=0u;var isWord=false;
  if(slot<cfg.length*8u) {
    channel=slot%8u; sourceOffset=(word*cfg.length+slot/8u)*${w}u;
  } else {
    let head=slot-cfg.length*8u; isWord=true; sourceOffset=word*${w}u;
    ${branches}
  }
  var sum=0.0f;
  for(var c=0u;c<${w}u;c++) {
    var value=0.0f;
    if(isWord) { value=pool[sourceOffset+c]; } else { value=source[sourceOffset+c]; }
    sum+=value*weight(weightOffset+channel*${w}u+c,weightScale);
  }
  destination[index]=sum+weight(biasOffset+channel,biasScale);
}`;
  return {...code,...rootShaders(m,prelude)};
}
