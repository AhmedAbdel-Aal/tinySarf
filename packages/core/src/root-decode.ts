import type {ModelManifest,RootTemplate} from './model';
export interface RootCandidate {root:string|null;score:number}
type Index=Map<number,RootTemplate[]>;
const indexes=new WeakMap<ModelManifest,{whole:Index;stem:Index;letters:Map<string,number>}>();
function prepare(m:ModelManifest) {
  let cached=indexes.get(m);if(cached)return cached;
  const group=(rules:RootTemplate[])=>{const index:Index=new Map();for(const rule of rules){const list=index.get(rule.surface.length)??[];list.push(rule);index.set(rule.surface.length,list);}return index;};
  cached={whole:group(m.rootDecoder!.whole),stem:group(m.rootDecoder!.stem),letters:new Map(m.vocabulary.slice(1).map((c,i)=>[c,i+1]))};indexes.set(m,cached);return cached;
}
function proposals(text:string,index:Index):Map<string,number> {
  const scores=new Map<string,number>();let total=0;
  for(const rule of index.get(text.length)??[]) {
    let matches=true;for(let i=0;i<text.length;i++)if(rule.surface[i]!=="*"&&rule.surface[i]!==text[i]){matches=false;break;}
    if(!matches)continue;
    const root=rule.output.map(value=>typeof value==='number'?text[value]:value).join('');
    scores.set(root,(scores.get(root)??0)+rule.distinctRoots);total+=rule.distinctRoots;
  }
  for(const [root,mass] of scores)scores.set(root,mass/total);
  return scores;
}
/** Data-derived constraints only: there is no runtime word-to-root dictionary. */
export function constrainRoots(word:string,tags:number[],heads:Record<string,number[]>,m:ModelManifest,unconstrained:RootCandidate[]):RootCandidate[] {
  const settings=m.rootDecoder;if(!settings)return unconstrained;
  const index=prepare(m),mass=proposals(word,index.whole),start=tags.indexOf(4),end=tags.lastIndexOf(4)+1;
  if(start>=0)for(const [root,value] of proposals(word.slice(start,end),index.stem))mass.set(root,(mass.get(root)??0)+value);
  const total=[...mass.values()].reduce((sum,value)=>sum+value,0),ranked=new Map<string,number>();
  const score=(root:string)=>[0,1,2,3].reduce((sum,slot)=>sum+heads[`root${slot}`][slot<root.length?index.letters.get(root[slot])!:0],0);
  for(const [root,value] of mass)ranked.set(root,score(root)+settings.priorWeight*Math.log(value/total));
  ranked.set('',score('')-settings.nullPenalty);
  // Retain the unmodified best neural score when its root is structurally
  // supported. Unsupported neural alternatives remain available at a penalty.
  for(let i=0;i<unconstrained.length;i++) {
    const candidate=unconstrained[i],root=candidate.root??'',supported=root===''||mass.has(root);
    const fallback=candidate.score-(supported?0:settings.unsupportedPenalty)-(root===''?settings.nullPenalty:0);
    if(i===0||!ranked.has(root))ranked.set(root,Math.max(ranked.get(root)??-Infinity,fallback));
  }
  return [...ranked].map(([root,score])=>({root:root||null,score})).sort((a,b)=>b.score-a.score).slice(0,4);
}
