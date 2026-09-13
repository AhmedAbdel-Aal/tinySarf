import type {LoadedModel} from './model';
import type {NormalizedWord} from './normalize';

/** Copy probabilities are accumulated by letter ID, so repeated radicals remain possible. */
export function rootCPU(word:NormalizedWord,model:LoadedModel,record:(name:string,values:Float32Array)=>void):Record<string,Float32Array> {
  const {manifest:m,tensors:t}=model,r=m.rootArchitecture!,length=word.text.length,width=r.width;
  let channels:number=r.embedding,x=new Float32Array(length*channels);
  for(let p=0;p<length;p++)for(let c=0;c<channels;c++)x[p*channels+c]=t['root.embedding'][(word.packed[p]&255)*channels+c]+t['root.position'][p*channels+c];
  record('root.embedding',x);
  for(let layer=0;layer<3;layer++) {
    const y=new Float32Array(length*width),weights=t[`root.conv${layer}.weight`],bias=t[`root.conv${layer}.bias`];
    for(let p=0;p<length;p++)for(let o=0;o<width;o++) {
      let sum=0;
      for(let k=0;k<3;k++) {
        const position=p+(k-1)*r.dilations[layer];if(position<0||position>=length)continue;
        for(let c=0;c<channels;c++)sum+=x[position*channels+c]*weights[(o*3+k)*channels+c];
      }
      y[p*width+o]=Math.max(0,sum+bias[o]);
    }
    x=y;channels=width;record(`root.conv${layer}`,x);
  }
  const pooled=new Float32Array(width);
  for(let c=0;c<width;c++){let sum=0;for(let p=0;p<length;p++)sum+=x[p*width+c];pooled[c]=sum/length;}
  record('root.pooled',pooled);
  const heads:Record<string,Float32Array>={};
  for(let slot=0;slot<4;slot++) {
    const name=`root${slot}`,classes=m.labels.heads[name].length,attention=new Float32Array(length),generated=new Float32Array(classes),copied=new Float32Array(classes);
    for(let p=0;p<length;p++){let sum=0;for(let c=0;c<width;c++)sum+=x[p*width+c]*t['root.pointer.weight'][slot*width+c];attention[p]=sum;}
    const attentionMax=Math.max(...attention);let attentionSum=0;
    for(let p=0;p<length;p++){attention[p]=Math.exp(attention[p]-attentionMax);attentionSum+=attention[p];}
    for(let p=0;p<length;p++)copied[word.packed[p]&255]+=attention[p]/attentionSum;
    for(let o=0;o<classes;o++){let sum=0;for(let c=0;c<width;c++)sum+=pooled[c]*t[`${name}.weight`][o*width+c];generated[o]=sum+t[`${name}.bias`][o];}
    const generatedMax=Math.max(...generated);let generatedSum=0;
    for(let o=0;o<classes;o++){generated[o]=Math.exp(generated[o]-generatedMax);generatedSum+=generated[o];}
    let gateLogit=0;for(let c=0;c<width;c++)gateLogit+=pooled[c]*t['root.gate.weight'][slot*width+c];gateLogit+=t['root.gate.bias'][slot];
    const gate=1/(1+Math.exp(-Math.max(-80,Math.min(80,gateLogit)))),out=new Float32Array(classes);
    for(let o=0;o<classes;o++)out[o]=Math.log(Math.max(1e-12,gate*copied[o]+(1-gate)*generated[o]/generatedSum));
    heads[name]=out;record(name,out);
  }
  return heads;
}
