import type {Logits} from "./cpu";
import type {NormalizedWord} from "./normalize";
import type {ModelManifest} from "./model";
import {SPAN_TYPES, FEATURE_VALUES, type MorphAnalysis, type MorphSpan, type MorphFeatures} from "./schema";
import {constrainRoots} from './root-decode';
function logSoftmax(values:ArrayLike<number>):number[] {
  const v=Array.from(values); if(v.some(x=>!Number.isFinite(x))) throw new Error("Non-finite model output");
  const max=Math.max(...v),z=max+Math.log(v.reduce((s,x)=>s+Math.exp(x-max),0)); return v.map(x=>x-z);
}
function ranked(values:number[],n:number):number[] {return values.map((_,i)=>i).sort((a,b)=>values[b]-values[a]||a-b).slice(0,n);}
interface SegCandidate {tags:number[];score:number}
function segmentationCandidates(word:string,logits:Float32Array,limit=8):SegCandidate[] {
  let beam:SegCandidate[]=[{tags:[],score:0}];
  for(let p=0;p<word.length;p++) {
    const scores=logSoftmax(logits.subarray(p*8,p*8+8)); const next:SegCandidate[]=[];
    for(const b of beam) for(let tag=0;tag<8;tag++) {
      const last=b.tags.at(-1)??-1;
      if(tag<last || (tag>4 && !b.tags.includes(4))) continue;
      if(tag===0 && (p!==0 || !"وف".includes(word[p]))) continue;
      if(tag===1 && !"أسلبوف".includes(word[p])) continue;
      if(tag===2 && (!"بكلو".includes(word[p]) || last===2)) continue;
      if(tag===3 && !(last!==3 && word.slice(p,p+2)==="ال" || last===3 && word[p]==="ل" && word[p-1]==="ا")) continue;
      // An article must include both letters before moving to a stem.
      if(last===3 && tag!==3 && b.tags.filter(t=>t===3).length!==2) continue;
      next.push({tags:[...b.tags,tag],score:b.score+scores[tag]});
    }
    beam=next.sort((a,b)=>b.score-a.score).slice(0,64);
  }
  return beam.filter(b=>b.tags.includes(4)).sort((a,b)=>b.score-a.score).slice(0,limit);
}
function spansFromTags(word:NormalizedWord,tags:number[]):MorphSpan[] {
  const spans:MorphSpan[]=[];
  tags.forEach((tag,i)=>{
    const type=SPAN_TYPES[tag],last=spans.at(-1);
    if(last?.type===type) last.end=word.offsets[i].end;
    else spans.push({type,start:word.offsets[i].start,end:word.offsets[i].end});
  }); return spans;
}
function rootCandidates(heads:Record<string,number[]>,m:ModelManifest):{root:string|null;score:number}[] {
  let beam=[{root:"",score:0}];
  for(let i=0;i<3;i++) {
    const name=`root${i}`,scores=heads[name],choices=ranked(scores.slice(1),3).map(x=>x+1);
    beam=beam.flatMap(b=>choices.map(c=>({root:b.root+m.labels.heads[name][c],score:b.score+scores[c]}))).sort((a,b)=>b.score-a.score).slice(0,8);
  }
  const last=heads.root3,choices=ranked(last,3);
  const candidates:{root:string|null;score:number}[]=beam.flatMap(b=>choices.map(c=>({root:b.root+(c?m.labels.heads.root3[c]:""),score:b.score+last[c]})));
  candidates.push({root:null,score:[0,1,2,3].reduce((s,i)=>s+heads[`root${i}`][0],0)});
  return candidates.sort((a,b)=>b.score-a.score).slice(0,4);
}
function publicValue(s:string):string|null {return s.startsWith("__")?null:s;}
// Legacy teacher exports contain a corrupted mixed-script pattern class. Keep
// its learned ranking score, but never expose that class as an Arabic pattern.
function publicPattern(s:string):string|null {return /^[ء-غف-ي\u064b-\u065f\u0670\u0671+]+$/u.test(s)&&/[ء-غف-ي\u0671]/u.test(s)?s:null;}
function constrainedFeatures(pos:string,features:Record<string,string>,spans:MorphSpan[]):MorphFeatures|null {
  const noun=["noun","proper_noun","adjective","numeral"].includes(pos), verb=pos==="verb",pronoun=pos==="pronoun";
  if(!noun && spans.some(s=>s.type==="article")) return null;
  if(verb && spans.some(s=>s.type==="preposition")) return null;
  const result:MorphFeatures={};
  for(const k of Object.keys(FEATURE_VALUES) as (keyof MorphFeatures)[]) {
    const value=publicValue(features[k]); if(!value) continue;
    if(["aspect","mood","voice"].includes(k) && !verb) continue;
    if(["case","state"].includes(k) && !noun) continue;
    if(k==="person" && !(verb||pronoun)) continue;
    if(["gender","number"].includes(k) && !(noun||verb||pronoun)) continue;
    result[k]=value;
  }
  if(noun && spans.some(s=>s.type==="article")) result.state="definite";
  if(noun && spans.some(s=>s.type==="pronominal_enclitic")) result.state="construct";
  if(result.aspect && result.aspect!=="imperfective") delete result.mood;
  return result;
}
/** Approximate constrained beam decoding. Scores are ranking scores, never calibrated probabilities. */
export function decode(word:NormalizedWord,logits:Logits,m:ModelManifest,topK:number,threshold=-Infinity):MorphAnalysis[] {
  const scores=Object.fromEntries(Object.entries(logits.heads).map(([k,v])=>[k,logSoftmax(v)]));
  const segments=segmentationCandidates(word.text,logits.segmentation);
  let roots=rootCandidates(scores,m);
  let beam:{values:Record<string,string>;score:number}[]=[{values:{},score:0}];
  const names=["pos","pattern",...Object.keys(FEATURE_VALUES)];
  for(const name of names) {
    const choices=ranked(scores[name],3);
    beam=beam.flatMap(b=>choices.map(c=>({values:{...b.values,[name]:m.labels.heads[name][c]},score:b.score+scores[name][c]}))).sort((a,b)=>b.score-a.score).slice(0,96);
  }
  // Best-first traversal of the sorted Cartesian product preserves the original
  // beam ranking without materializing and serializing every combination.
  type Node={s:number;b:number;r:number;score:number};
  const heap:Node[]=[],visited=new Set<string>(),unique=new Set<string>(),result:MorphAnalysis[]=[];
  const better=(a:Node,b:Node)=>a.score>b.score || a.score===b.score&&(a.s<b.s || a.s===b.s&&(a.b<b.b || a.b===b.b&&a.r<b.r));
  const push=(s:number,b:number,r:number)=>{
    if(s>=segments.length || b>=beam.length || r>=roots.length) return;
    const key=`${s}:${b}:${r}`;if(visited.has(key))return;visited.add(key);
    const node={s,b,r,score:segments[s].score+beam[b].score+roots[r].score};heap.push(node);
    let i=heap.length-1;while(i>0){const parent=(i-1)>>1;if(!better(heap[i],heap[parent]))break;[heap[i],heap[parent]]=[heap[parent],heap[i]];i=parent;}
  };
  const pop=()=>{
    const result=heap[0],last=heap.pop()!;
    if(heap.length){heap[0]=last;let i=0;while(true){let best=i;for(const j of [i*2+1,i*2+2])if(j<heap.length&&better(heap[j],heap[best]))best=j;if(best===i)break;[heap[i],heap[best]]=[heap[best],heap[i]];i=best;}}
    return result;
  };
  const spanCache=segments.map(seg=>spansFromTags(word,seg.tags));
  if(m.rootDecoder) {
    // Root constraints use the same best valid segmentation as the public
    // analysis. Roots do not alter the ranking of non-root fields.
    let best=-Infinity,bestSegment=0;
    for(let s=0;s<segments.length;s++) {
      if(segments[s].score+beam[0].score<=best)break;
      for(const b of beam) {
        const score=segments[s].score+b.score;if(score<=best)break;
        if(constrainedFeatures(publicValue(b.values.pos)??"unknown",b.values,spanCache[s])){best=score;bestSegment=s;break;}
      }
    }
    roots=constrainRoots(word.text,segments[bestSegment]?.tags??[],scores,m,roots);
  }
  push(0,0,0);
  while(heap.length && result.length<topK) {
    const node=pop(),score=node.score/(word.text.length+names.length+4);if(score<threshold)break;
    push(node.s+1,node.b,node.r);push(node.s,node.b+1,node.r);push(node.s,node.b,node.r+1);
    const b=beam[node.b],spans=spanCache[node.s],pos=publicValue(b.values.pos)??"unknown";
    const features=constrainedFeatures(pos,b.values,spans);if(!features)continue;
    const a={spans,root:roots[node.r].root,pattern:publicPattern(b.values.pattern),pos,features,score};
    const key=JSON.stringify({...a,score:0});if(unique.has(key))continue;unique.add(key);result.push(a);
  }
  return result;
}
