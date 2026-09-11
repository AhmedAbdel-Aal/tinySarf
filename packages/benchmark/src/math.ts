export function quantile(xs: number[], p: number): number {
  if (!xs.length || p < 0 || p > 1 || xs.some(x => !Number.isFinite(x))) throw new Error("Invalid quantile input");
  const sorted = [...xs].sort((a,b) => a-b), i = (sorted.length-1)*p;
  return sorted[Math.floor(i)] + (sorted[Math.ceil(i)]-sorted[Math.floor(i)])*(i%1);
}
export function stats(xs: number[]) {
  if (!xs.length) return null;
  const mean = xs.reduce((a,b) => a+b,0)/xs.length;
  return {count:xs.length,p50:quantile(xs,.5),p95:quantile(xs,.95),mean,
    standardDeviation:Math.sqrt(xs.reduce((s,x) => s+(x-mean)**2,0)/xs.length),min:Math.min(...xs),max:Math.max(...xs)};
}
export function random(seed: number) { return () => { seed |= 0; seed = seed+0x6d2b79f5|0; let t=Math.imul(seed^seed>>>15,1|seed); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
export function bootstrap(values: number[], seed=42, resamples=2000) {
  if (!values.length) return null;
  const rng=random(seed), means=[];
  for(let r=0;r<resamples;r++) { let sum=0; for(let i=0;i<values.length;i++) sum+=values[Math.floor(rng()*values.length)]; means.push(sum/values.length); }
  return {low:quantile(means,.025),high:quantile(means,.975),seed,resamples,unit:"word"};
}
export function pairedBootstrap(candidate: number[], active: number[], seed=42, resamples=2000) {
  if (candidate.length!==active.length) throw new Error("Paired inputs must have identical length and order");
  const differences=candidate.map((x,i)=>x-active[i]);
  return {difference:differences.length ? differences.reduce((a,b)=>a+b,0)/differences.length : null,ci95:bootstrap(differences,seed,resamples)};
}

/** Predeclared stability rule; never infer stability solely from reaching a call count. */
export function warmupStable(samples:number[]):boolean {
 if(samples.length<20)return false;
 const a=stats(samples.slice(-20,-10))!,b=stats(samples.slice(-10))!;
 return a.mean>0 && b.mean>0 && Math.abs(b.p50/a.p50-1)<=.05 && a.standardDeviation/a.mean<=.10 && b.standardDeviation/b.mean<=.10;
}
