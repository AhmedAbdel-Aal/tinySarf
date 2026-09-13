import type {ModelManifest} from './model';
const f=(n:number)=>`${n.toExponential(9)}f`;
/** Six parallel passes; only root outputs are replaced in the shared output buffer. */
export function rootShaders(m:ModelManifest,prelude:string):Record<string,string> {
  const r=m.rootArchitecture;if(!r)return {};
  const t=m.tensors,w=r.width,e=r.embedding,code:Record<string,string>={};
  code['root.embedding']=prelude+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let index=gid.x;let token=index/${e}u;if(token>=cfg.batch*cfg.length){return;}
  let channel=index%${e}u;let id=packed[token]&255u;
  if(id==0u){destination[index]=0.0;return;}
  destination[index]=weight(${t['root.embedding'].offset}u+id*${e}u+channel,${f(t['root.embedding'].scale)})+weight(${t['root.position'].offset}u+(token%cfg.length)*${e}u+channel,${f(t['root.position'].scale)});
}`;
  for(let layer=0;layer<3;layer++) {
    const channels=layer===0?e:w,wt=t[`root.conv${layer}.weight`],bias=t[`root.conv${layer}.bias`];
    code[`root.conv${layer}`]=prelude+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let index=gid.x;let token=index/${w}u;if(token>=cfg.batch*cfg.length){return;}
  if((packed[token]&255u)==0u){destination[index]=0.0;return;}
  let channel=index%${w}u;let position=token%cfg.length;let word=token/cfg.length;var sum=0.0f;
  for(var k=0u;k<3u;k++){
    let p=i32(position)+(i32(k)-1)*${r.dilations[layer]};if(p<0||p>=i32(cfg.length)){continue;}
    let a=(word*cfg.length+u32(p))*${channels}u;let b=${wt.offset}u+(channel*3u+k)*${channels}u;
    for(var c=0u;c<${channels}u;c++){sum=sum+source[a+c]*weight(b+c,${f(wt.scale)});}
  }
  destination[index]=max(0.0f,sum+weight(${bias.offset}u+channel,${f(bias.scale)}));
}`;
  }
  code['root.pool']=prelude+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let index=gid.x;let word=index/${w}u;let channel=index%${w}u;if(word>=cfg.batch){return;}
  var sum=0.0f;var count=0u;
  for(var p=0u;p<cfg.length;p++){
    if((packed[word*cfg.length+p]&255u)!=0u){count++;}
    sum+=source[(word*cfg.length+p)*${w}u+channel];
  }
  destination[index]=sum/f32(max(count,1u));
}`;
  const offsets:Record<string,number>={};let total=0;
  for(const [name,labels] of Object.entries(m.labels.heads)){offsets[name]=total;total+=labels.length;}
  const classes=m.labels.heads.root0.length,ptr=t['root.pointer.weight'],gate=t['root.gate.weight'],gateBias=t['root.gate.bias'];
  const branches=[0,1,2,3].map(slot=>{
    const wt=t[`root${slot}.weight`],bias=t[`root${slot}.bias`];
    return `if(rootSlot==${slot}u){weightOffset=${wt.offset}u;biasOffset=${bias.offset}u;weightScale=${f(wt.scale)};biasScale=${f(bias.scale)};headOffset=${offsets[`root${slot}`]}u;}`;
  }).join('\n');
  code['root.heads']=prelude+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let word=gid.x/4u;let rootSlot=gid.x%4u;if(word>=cfg.batch){return;}
  var weightOffset=0u;var biasOffset=0u;var weightScale=1.0f;var biasScale=1.0f;var headOffset=0u;
  ${branches}
  var attention:array<f32,32>;var maximum=-1e30f;
  for(var p=0u;p<cfg.length;p++){
    attention[p]=-1e30f;if((packed[word*cfg.length+p]&255u)==0u){continue;}
    var sum=0.0f;
    for(var c=0u;c<${w}u;c++){sum+=source[(word*cfg.length+p)*${w}u+c]*weight(${ptr.offset}u+rootSlot*${w}u+c,${f(ptr.scale)});}
    attention[p]=sum;maximum=max(maximum,sum);
  }
  var attentionSum=0.0f;
  for(var p=0u;p<cfg.length;p++){
    attention[p]=select(exp(attention[p]-maximum),0.0f,(packed[word*cfg.length+p]&255u)==0u);attentionSum+=attention[p];
  }
  var copied:array<f32,${classes}>;
  for(var p=0u;p<cfg.length;p++){copied[packed[word*cfg.length+p]&255u]+=attention[p]/attentionSum;}
  var generated:array<f32,${classes}>;var generatedMaximum=-1e30f;
  for(var o=0u;o<${classes}u;o++){
    var sum=0.0f;
    for(var c=0u;c<${w}u;c++){sum+=pool[word*${w}u+c]*weight(weightOffset+o*${w}u+c,weightScale);}
    generated[o]=sum+weight(biasOffset+o,biasScale);generatedMaximum=max(generatedMaximum,generated[o]);
  }
  var generatedSum=0.0f;
  for(var o=0u;o<${classes}u;o++){generated[o]=exp(generated[o]-generatedMaximum);generatedSum+=generated[o];}
  var gateLogit=0.0f;
  for(var c=0u;c<${w}u;c++){gateLogit+=pool[word*${w}u+c]*weight(${gate.offset}u+rootSlot*${w}u+c,${f(gate.scale)});}
  gateLogit+=weight(${gateBias.offset}u+rootSlot,${f(gateBias.scale)});
  let mixing=1.0f/(1.0f+exp(-clamp(gateLogit,-80.0f,80.0f)));
  let offset=word*(cfg.length*8u+${total}u)+cfg.length*8u+headOffset;
  for(var o=0u;o<${classes}u;o++){
    destination[offset+o]=log(max(1e-12f,mixing*copied[o]+(1.0f-mixing)*generated[o]/generatedSum));
  }
}`;
  return code;
}
