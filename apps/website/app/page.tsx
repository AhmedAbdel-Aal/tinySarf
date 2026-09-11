'use client';

import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import report from '@/lib/results.json';

type Analysis = { spans: {type:string;start:number;end:number}[]; root:string|null; pattern:string|null; pos:string; features:Record<string,string>; score:number };
type Result = {word:string;analyses:Analysis[]};
const examples = ['وبكتابهم', 'المدرسة', 'يكتبون', 'مكتوب'];
const humanize = (value:string) => value.replaceAll('_', ' ');

export default function Home() {
  const [mode,setMode] = useState('word');
  const [input,setInput] = useState(examples[0]);
  const [backend,setBackend] = useState('webgpu');
  const [topK,setTopK] = useState('3');
  const [results,setResults] = useState<Result[]>([]);
  const [rank,setRank] = useState(0);
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState('');
  const [elapsed,setElapsed] = useState<number|null>(null);
  const [gpuAvailable,setGpuAvailable] = useState<boolean|null>(null);
  const [copied,setCopied] = useState(false);
  const callId=useRef(0);
  useEffect(()=>{setGpuAvailable(window.isSecureContext && 'gpu' in navigator);},[]);

  async function run(value=input) {
    const id=++callId.current;
    const words=mode==='batch'?value.trim().split(/\s+/u).filter(Boolean):[value.trim()];
    if(!words.length || !words[0]) {setError('Enter an Arabic word to analyze.');return;}
    setBusy(true);setError('');setCopied(false);setRank(0);
    try {
      const modulePath='/tinysarf/index.js';
      const {analyze}=await import(/* @vite-ignore */ modulePath);
      const start=performance.now();
      const analyses:Analysis[][]=await analyze.batch(words,{topK:Number(topK),backend});
      if(id!==callId.current) return;
      setElapsed(performance.now()-start);setResults(words.map((word,i)=>({word,analyses:analyses[i]})));
    } catch(e) {if(id===callId.current) {setError(e instanceof Error?e.message:'Analysis failed. Please try again.');setResults([]);setElapsed(null);}}
    finally {if(id===callId.current) setBusy(false);}
  }
  async function copy() {
    try {await navigator.clipboard.writeText(JSON.stringify(results.map(r=>({word:r.word,analyses:r.analyses})),null,2));setCopied(true);}
    catch {setError('Clipboard access is unavailable. You can select the output in the JSON view.');}
  }

  return <div className="app-shell">
    <header className="topbar">
      <a href="/" className="wordmark" aria-label="TinySarf home"><span lang="ar" className="mark">صرف</span><span>TinySarf<span className="wordmark-dot">.</span></span></a>
      <div className="header-right"><span className="local-indicator"><i /> Inference stays in your browser</span><a href="/docs/MODEL_CARD.md">Model card ↗</a></div>
    </header>

    <main>
      <div className="page-intro"><div><p className="eyebrow">ARABIC MORPHOLOGY LAB</p><h1>Look inside a word.</h1><p className="intro-copy">Explore morphemes, roots, patterns, and the ambiguity between them.</p></div><span className="experiment-label">Experimental · unpromoted</span></div>

      <div className="workbench">
        <section className="input-panel" aria-label="Arabic input">
          <div className="panel-heading"><span className="section-number">01</span><h2>Input</h2><span className="msa-label">Modern Standard Arabic</span></div>
          <Tabs value={mode} onValueChange={(value)=>{setMode(String(value));setInput(value==='batch'?examples.join('\n'):examples[0]);setResults([]);setError('');setElapsed(null);}}>
            <TabsList className="mode-tabs"><TabsTrigger disabled={busy} value="word">Single word</TabsTrigger><TabsTrigger disabled={busy} value="batch">Batch</TabsTrigger></TabsList>
          </Tabs>
          <label htmlFor="arabic-input" className="input-label">{mode==='batch'?'Words separated by spaces or new lines':'Your Arabic word'}</label>
          <Textarea id="arabic-input" className={`arabic-input ${mode==='batch'?'batch-input':''}`} dir="rtl" lang="ar" value={input} onChange={e=>setInput(e.target.value)} placeholder="اكتب كلمة" maxLength={mode==='batch'?70000:512} spellCheck={false} onKeyDown={e=>{if(e.key==='Enter' && (mode==='word'||e.metaKey||e.ctrlKey)){e.preventDefault();if(!busy) void run();}}} />
          <div className="input-footnote"><span>{mode==='batch'?'Up to 2,048 words':'Up to 32 letters'}</span><span>Diacritics accepted</span></div>
          {mode==='word'?<div className="examples"><span>Try</span>{examples.map(word=><Button key={word} variant="ghost" disabled={busy} onClick={()=>{setInput(word);void run(word);}} className="example-word" lang="ar" dir="rtl">{word}</Button>)}</div>:null}
          <div className="settings-row">
            <div><label id="backend-label">Run with</label><Select value={backend} onValueChange={value=>setBackend(String(value))} disabled={busy}><SelectTrigger aria-labelledby="backend-label"><SelectValue>{backend==='webgpu'?'WebGPU':'CPU reference'}</SelectValue></SelectTrigger><SelectContent><SelectItem value="webgpu">WebGPU</SelectItem><SelectItem value="cpu">CPU reference</SelectItem></SelectContent></Select></div>
            <div><label id="topk-label">Analyses</label><Select value={topK} onValueChange={value=>setTopK(String(value))} disabled={busy}><SelectTrigger aria-labelledby="topk-label"><SelectValue>{topK==='1'?'Top 1':'Top 3'}</SelectValue></SelectTrigger><SelectContent><SelectItem value="1">Top 1</SelectItem><SelectItem value="3">Top 3</SelectItem></SelectContent></Select></div>
          </div>
          <Button className="analyze-button" size="lg" disabled={busy} onClick={()=>void run()}>{busy?<><span className="loading-dot"/>Analyzing locally…</>:<>Analyze {mode==='batch'?'batch':'word'}<span aria-hidden="true">↗</span></>}</Button>
          {gpuAvailable===false && backend==='webgpu'?<p className="support-note">WebGPU is unavailable in this browser. Select CPU reference, or use a browser with WebGPU support in a secure context.</p>:null}
          {error?<p className="error-message" role="alert">{error}</p>:null}
          <p className="input-disclaimer">A small learned model, still under evaluation. It can be wrong, including on these examples.</p>
        </section>

        <section className="output-panel" aria-label="Morphological analyses" aria-busy={busy}>
          <div className="panel-heading"><span className="section-number">02</span><h2>Analysis</h2><span className="result-meta" aria-live="polite">{elapsed===null?'Awaiting input':`${results.length} word${results.length===1?'':'s'} · ${elapsed.toFixed(1)} ms this call`}</span></div>
          {!results.length?<div className="empty-result"><span className="empty-arabic" lang="ar" dir="rtl">كلمة</span><h3>Every word has a structure.</h3><p>Analyze a word to inspect its predicted morphemes and alternative readings.</p><span className="empty-caption">Local model · no inference server</span></div>:<Tabs defaultValue="visual" className="result-tabs">
            <div className="output-toolbar"><TabsList><TabsTrigger value="visual">Visual</TabsTrigger><TabsTrigger value="json">JSON</TabsTrigger></TabsList><Button variant="ghost" onClick={()=>void copy()} className="copy-button">{copied?'Copied ✓':'Copy JSON'}</Button></div>
            <TabsContent value="visual"><div className="result-list">{results.map((result,index)=>{
              const selected=result.analyses[Math.min(rank,result.analyses.length-1)];
              return <article key={`${index}:${result.word}`} className="word-result">
                {results.length>1?<p className="batch-word" lang="ar" dir="rtl">{result.word}</p>:null}
                {!selected?<p className="abstained">No candidate passed the decoder constraints.</p>:<>
                  <div className="morphemes" dir="rtl" lang="ar">{selected.spans.map((span,i)=><div className={`morpheme morph-${span.type}`} key={i}><span>{result.word.slice(span.start,span.end)}</span><small lang="en" dir="ltr">{humanize(span.type)}</small></div>)}</div>
                  <div className="analysis-facts"><div><span>Root</span><strong className={selected.root?undefined:"unknown-value"} lang={selected.root?"ar":"en"} dir={selected.root?"rtl":"ltr"}>{selected.root??'Unknown'}</strong></div><div><span>Pattern</span><strong className={selected.pattern?undefined:"unknown-value"} lang={selected.pattern?"ar":"en"} dir={selected.pattern?"rtl":"ltr"}>{selected.pattern??'Unknown'}</strong></div><div><span>Part of speech</span><strong className="pos-value">{humanize(selected.pos)}</strong></div></div>
                  <div className="feature-chips">{Object.entries(selected.features).map(([key,value])=><span key={key}><small>{key}</small>{humanize(value)}</span>)}{!Object.keys(selected.features).length?<p>No applicable features predicted.</p>:null}</div>
                  {result.analyses.length>1 && results.length===1?<div className="alternatives"><p>Alternative readings <span>Ranking scores, not probabilities</span></p>{result.analyses.map((analysis,i)=><Button variant="ghost" key={i} onClick={()=>setRank(i)} className={`alternative ${rank===i?'selected':''}`} aria-pressed={rank===i}><span className="rank-index">0{i+1}</span><span lang="ar" dir="rtl">{analysis.root??'—'}</span><span>{humanize(analysis.pos)}</span><code>{analysis.score.toFixed(3)}</code></Button>)}</div>:null}
                </>}
              </article>;
            })}</div></TabsContent>
            <TabsContent value="json"><pre className="json-output" tabIndex={0}>{JSON.stringify(results,null,2)}</pre></TabsContent>
          </Tabs>}
          <div className="output-note">Spans use UTF-16 offsets into your original input. Unvowelled words can have several valid readings.</div>
        </section>
      </div>

      <section className="evidence-section" aria-labelledby="evidence-title"><div className="evidence-heading"><div><p className="eyebrow">THE EVIDENCE</p><h2 id="evidence-title">Measured, with the gaps visible.</h2></div><a href="/docs/MODEL_CARD.md">Read the full model card ↗</a></div><div className="evidence-grid">{report.entries.map(entry=><div className="evidence-item" key={entry.label}><p>{entry.label}</p><strong>{entry.value}</strong><span>{entry.note}</span></div>)}</div><p className="evidence-caption">{report.note} {report.artifact?<a href={report.artifact}>View result artifact ↗</a>:null}</p></section>
      <section className="limitations"><h2>Know its limits.</h2><p>TinySarf analyzes isolated MSA words. Dialects, names, rare roots, and spelling variants may fail. The current experiment has no independently human-labelled gold evaluation. Do not use it for grading or religious, legal, medical, or security decisions.</p></section>
    </main>
    <footer><span>TinySarf <span lang="ar">· صرف صغير</span></span><span>Small model. Open methods. Local inference.</span><a href="/docs/architecture.md">Architecture ↗</a></footer>
  </div>;
}
