import type {EvaluationWord} from "./metrics";
import type {MorphAnalysis,MorphSpan} from "../../core/src/schema";
export function frequencyBaseline(train:EvaluationWord[],rows:EvaluationWord[]):MorphAnalysis[][] {
  const counts=new Map<string,{count:number;analysis:MorphAnalysis}>();
  for(const row of train) for(const a of row.analyses) {
    const key=JSON.stringify({root:a.root,pattern:a.pattern,pos:a.pos,features:a.features});
    const entry=counts.get(key)??{count:0,analysis:a};entry.count+=1/row.analyses.length;counts.set(key,entry);
  }
  const best=[...counts.values()].sort((a,b)=>b.count-a.count)[0]?.analysis;
  if(!best) throw new Error("Frequency baseline needs training data");
  return rows.map(r=>[{...best,spans:[{type:"stem",start:0,end:r.word.length}],score:0}]);
}
export function rulesBaseline(rows:EvaluationWord[]):MorphAnalysis[][] {
  return rows.map(({word})=>{
    const spans:MorphSpan[]=[]; let start=0,end=word.length;
    if(word.length>=5 && "وف".includes(word[0])) {spans.push({type:"conjunction",start:0,end:1});start++;}
    if(word.length-start>=5 && word.slice(start+1,start+3)==="ال" && "بكل".includes(word[start])) {spans.push({type:"preposition",start,end:start+1});start++;}
    if(word.slice(start,start+2)==="ال" && word.length-start>=4) {spans.push({type:"article",start,end:start+2});start+=2;}
    const suffix=["هم","هن","ها","نا","ه"].find(s=>word.endsWith(s)&&word.length-start-s.length>=3);
    if(suffix) end-=suffix.length;
    spans.push({type:"stem",start,end}); if(suffix) spans.push({type:"pronominal_enclitic",start:end,end:word.length});
    return [{spans,root:null,pattern:null,pos:"noun",features:{},score:0}];
  });
}
