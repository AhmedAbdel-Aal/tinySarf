import type { MorphAnalysis, MorphSpan } from "../../core/src/schema";
import { SPAN_TYPES, FEATURE_VALUES } from "../../core/src/schema";
import { bootstrap } from "./math";
export interface EvaluationWord {word:string; analyses:MorphAnalysis[]; slices?:string[]; labelOrigin?:string}
type Rate = { numerator:number; denominator:number; value:number|null; ci95?:ReturnType<typeof bootstrap> };
function rate(numerator:number,denominator:number):Rate { return {numerator,denominator,value:denominator ? numerator/denominator:null}; }
function canonical(a:unknown):string {
  if(Array.isArray(a)) return `[${a.map(canonical).join(",")}]`;
  if(a!==null && typeof a==="object") return `{${Object.entries(a).sort(([a],[b])=>a.localeCompare(b)).map(([k,v])=>`${k}:${canonical(v)}`).join(",")}}`;
  return JSON.stringify(a);
}
export function analysisKey(a:MorphAnalysis) { return canonical({spans:a.spans,root:a.root,pattern:a.pattern,pos:a.pos,features:a.features,...(a.lemma===undefined ? {} : {lemma:a.lemma})}); }
const same=(a:unknown,b:unknown)=>canonical(a)===canonical(b);
const boundary=(a:MorphAnalysis)=>new Set(a.spans.slice(0,-1).map(s=>s.end));
function intersection<T>(a:Set<T>,b:Set<T>):number { return [...a].filter(x=>b.has(x)).length; }
function charTypes(a:MorphAnalysis, length:number):string[] {
  const labels=Array<string>(length).fill("__missing__");
  for(const s of a.spans) for(let i=s.start;i<s.end;i++) labels[i]=s.type;
  return labels;
}
function confusionReport(pairs:[string,string][], labels?:readonly string[]) {
  const classes=[...new Set([...(labels??[]),...pairs.flat()])].sort();
  const matrix=Object.fromEntries(classes.map(a=>[a,Object.fromEntries(classes.map(b=>[b,0]))]));
  for(const [truth,pred] of pairs) matrix[truth][pred]++;
  const perClass=Object.fromEntries(classes.map(c=>{
    const tp=matrix[c][c], support=pairs.filter(([t])=>t===c).length, predicted=pairs.filter(([,p])=>p===c).length;
    const precision=predicted?tp/predicted:0, recall=support?tp/support:0;
    return [c,{precision,recall,f1:precision+recall?2*precision*recall/(precision+recall):0,support,predicted,truePositive:tp}];
  }));
  // Macro averages over classes supported by either truth or prediction; absent fixed classes retain support zero.
  const supported=Object.values(perClass).filter(c=>c.support+c.predicted>0);
  return {macroF1:supported.length?supported.reduce((s,c)=>s+c.f1,0)/supported.length:null,perClass,confusionMatrix:matrix};
}
export function evaluate(rows:EvaluationWord[], predictions:MorphAnalysis[][], options:{threshold?:number;bootstrap?:boolean}={}) {
  if(rows.length!==predictions.length) throw new Error("Prediction count mismatch");
  const keys=["segmentationExactMatch","rootExactMatch","patternExactMatch","posAccuracy","fullAnalysisTop1","fullAnalysisTop3"] as const;
  const perWord=Object.fromEntries(keys.map(k=>[k,[] as number[]])) as Record<typeof keys[number],number[]>;
  let boundaryTP=0,boundaryPred=0,boundaryTrue=0,covered=0,errors=0,rootChars=0,rootTotal=0;
  const spanPairs:[string,string][]=[],posPairs:[string,string][]=[];
  const featurePairs=Object.fromEntries(Object.keys(FEATURE_VALUES).map(k=>[k,[] as [string,string][]]));
  const featureHits=Object.fromEntries(Object.keys(FEATURE_VALUES).map(k=>[k,0]));
  for(let i=0;i<rows.length;i++) {
    const row=rows[i],pred=predictions[i],top=pred[0],gold=row.analyses;
    if(!gold.length) throw new Error("Every evaluated word needs at least one valid analysis");
    const hit=(fn:(g:MorphAnalysis)=>boolean)=>Number(!!top&&gold.some(fn));
    perWord.segmentationExactMatch.push(hit(g=>same(top.spans,g.spans)));
    perWord.rootExactMatch.push(hit(g=>top.root===g.root));
    perWord.patternExactMatch.push(hit(g=>top.pattern===g.pattern));
    perWord.posAccuracy.push(hit(g=>top.pos===g.pos));
    const exact=hit(g=>analysisKey(top)===analysisKey(g)); perWord.fullAnalysisTop1.push(exact);
    perWord.fullAnalysisTop3.push(Number(pred.slice(0,3).some(p=>gold.some(g=>analysisKey(p)===analysisKey(g)))));
    if(top && top.score>=(options.threshold??-Infinity)) { covered++; errors+=1-exact; }
    // Choose an allowed reference with greatest typed-character agreement for the segmentation confusion diagnostic.
    const pt=top?charTypes(top,row.word.length):Array<string>(row.word.length).fill("__abstain__");
    const best=[...gold].sort((a,b)=>charTypes(b,row.word.length).filter((x,j)=>x===pt[j]).length-charTypes(a,row.word.length).filter((x,j)=>x===pt[j]).length)[0];
    const gt=charTypes(best,row.word.length); gt.forEach((x,j)=>spanPairs.push([x,pt[j]]));
    const pb=top?boundary(top):new Set<number>(), gb=boundary(best);
    boundaryTP+=intersection(pb,gb); boundaryPred+=pb.size; boundaryTrue+=gb.size;
    posPairs.push([gold.find(g=>g.pos===top?.pos)?.pos??gold[0].pos,top?.pos??"__abstain__"]);
    const pr=top?.root??"", roots=gold.map(g=>g.root??"");
    const root=roots.sort((a,b)=>[...b].filter((c,j)=>c===pr[j]).length-[...a].filter((c,j)=>c===pr[j]).length)[0];
    rootTotal+=Math.max(pr.length,root.length); rootChars+=[...root].filter((c,j)=>c===pr[j]).length;
    for(const k of Object.keys(FEATURE_VALUES) as (keyof typeof FEATURE_VALUES)[]) {
      const value=top?.features[k]??"__absent__";
      const correct=!!top&&gold.some(g=>(g.features[k]??"__absent__")===value);
      featureHits[k]+=Number(correct);
      featurePairs[k].push([correct?value:gold[0].features[k]??"__absent__",top?value:"__abstain__"]);
    }
  }
  const rates=Object.fromEntries(keys.map(k=>[k,{...rate(perWord[k].reduce((a,b)=>a+b,0),rows.length),...(options.bootstrap===false?{}:{ci95:bootstrap(perWord[k])})}])) as Record<typeof keys[number],Rate>;
  const precision=boundaryPred?boundaryTP/boundaryPred:boundaryTrue===0?1:0;
  const recall=boundaryTrue?boundaryTP/boundaryTrue:boundaryPred===0?1:0;
  const features=Object.fromEntries(Object.keys(FEATURE_VALUES).map(k=>[k,{accuracy:rate(featureHits[k],rows.length),...confusionReport(featurePairs[k])}]));
  const featureF1=Object.values(features).map(f=>f.macroF1).filter((x):x is number=>x!==null);
  return {count:rows.length,...rates,boundary:{precision:rows.length?precision:null,recall:rows.length?recall:null,f1:rows.length?(precision+recall?2*precision*recall/(precision+recall):0):null,truePositive:boundaryTP,predicted:boundaryPred,support:boundaryTrue},
    spanTypes:confusionReport(spanPairs,SPAN_TYPES),rootCharacterAccuracy:rate(rootChars,rootTotal),pos:confusionReport(posPairs),features,
    featureMacroF1:featureF1.length?featureF1.reduce((a,b)=>a+b,0)/featureF1.length:null,
    coverage:rate(covered,rows.length),risk:rate(errors,covered),calibration:null,perWord};
}
