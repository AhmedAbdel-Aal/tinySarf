/** All public numeric tables are derived from immutable, hash-linked evidence. */
import {readFile,writeFile,readdir,mkdir,copyFile,stat} from 'node:fs/promises';
import path from 'node:path';
import {parseArgs} from 'node:util';
import {ROOT,json,hash,validateArtifact} from '../packages/benchmark/src/artifact';
import {selectedModel} from '../packages/benchmark/src/selection';
const {values}=parseArgs({options:{refresh:{type:'boolean',default:false}}});
const selected=await selectedModel(),m=await json(path.join(selected.directory,'manifest.json'));
const results=path.join(ROOT,'packages/benchmark/results'),indexFile=path.join(ROOT,'packages/benchmark/reports/current.json');
let index:any;
if(values.refresh){
 const runtimeSha256=hash(await readFile(path.join(ROOT,'packages/core/dist/index.js')));
 const files=(await readdir(results)).sort().reverse(),artifacts:any={};
 for(const kind of ['correctness','size','browser','parity']) {
  for(const file of files.filter(f=>f.startsWith(kind+'-')&&f.endsWith('.json'))){const candidate=await json(path.join(results,file));if(candidate.model.sha256===m.sha256&&candidate.runtime?.sha256===runtimeSha256){artifacts[kind]={file:`packages/benchmark/results/${file}`,sha256:hash(await readFile(path.join(results,file)))};break;}}
  if(!artifacts[kind])throw new Error(`No measured ${kind} artifact for this checkpoint`);
 }
 const supplemental:any={};
 for(const kind of ['reproduction','ai-review'])for(const file of files.filter(f=>f.startsWith(kind+'-')&&f.endsWith('.json'))){const record=await json(path.join(results,file));if(record.model.sha256===m.sha256&&record.runtime?.sha256===runtimeSha256&&(kind!=='reproduction'||record.passed)){supplemental[kind]={file:`packages/benchmark/results/${file}`,sha256:hash(await readFile(path.join(results,file)))};break;}}
 for(const file of files.filter(f=>f.startsWith('corrected-comparison-')&&f.endsWith('.json'))){const record=await json(path.join(results,file));if(record.baselineModel.sha256===m.sha256&&record.runtime?.sha256===runtimeSha256){supplemental['corrected-comparison']={file:`packages/benchmark/results/${file}`,sha256:hash(await readFile(path.join(results,file)))};break;}}
 const promotions=path.join(ROOT,'packages/training/promotions');for(const file of (await readdir(promotions)).filter(f=>/^\d.*Z\.json$/.test(f)).sort().reverse()){supplemental.eligibility={file:`packages/training/promotions/${file}`,sha256:hash(await readFile(path.join(promotions,file)))};break;}
 index={schemaVersion:1,status:'experimental-unpromoted',modelId:m.id,modelSha256:m.sha256,artifacts,supplemental};await mkdir(path.dirname(indexFile),{recursive:true});await writeFile(indexFile,JSON.stringify(index,null,2)+'\n');
}else index=await json(indexFile);
const loaded:any={};for(const [kind,entry] of Object.entries(index.artifacts) as [string,any][]){const bytes=await readFile(path.join(ROOT,entry.file));if(hash(bytes)!==entry.sha256)throw new Error(`Frozen ${kind} artifact changed`);loaded[kind]=JSON.parse(bytes.toString());validateArtifact(loaded[kind]);if(loaded[kind].model.sha256!==m.sha256)throw new Error('Report/checkpoint mismatch');}
const {correctness:c,size:s,browser:b,parity:p}=loaded,q=c.correctness.teacherAgreement,sz=s.size;
const supplemental:any={};for(const [kind,entry] of Object.entries(index.supplemental??{}) as [string,any][]){const bytes=await readFile(path.join(ROOT,entry.file));if(hash(bytes)!==entry.sha256)throw new Error(`Frozen ${kind} evidence changed`);supplemental[kind]=JSON.parse(bytes.toString());}
const split=await json(path.join(ROOT,'packages/training/data/split-manifest.json')),config=await json(path.join(selected.directory,'config.json')),training=await json(path.join(selected.directory,'result.json'));
const pct=(x:number|null|undefined)=>x==null?'TBD — not measured':`${(x*100).toFixed(2)}%`;
const num=(x:number|null|undefined)=>x==null?'TBD — not measured':x.toLocaleString('en-US',{maximumFractionDigits:2});
const rate=(x:any)=>!x||x.value==null?'TBD — not measured':`${num(x.numerator)} / ${num(x.denominator)} · ${pct(x.value)}${x.ci95?` (95% CI ${pct(x.ci95.low)}–${pct(x.ci95.high)})`:''}`;
const link=(kind:string)=>`[${path.basename(index.artifacts[kind].file)}](${index.artifacts[kind].file})`;
const table=(headers:string[],rows:any[][])=>`| ${headers.join(' | ')} |\n| ${headers.map(()=>'---').join(' | ')} |\n${rows.map(row=>`| ${row.join(' | ')} |`).join('\n')}`;
const batch=b.warm.find((w:any)=>w.id==='batch-128'),single=b.warm.find((w:any)=>w.id==='single-short');
const roleStats=[];
for(const role of ['train','verification']) {const rows=(await json(path.join(ROOT,'packages/training/data',split.directory,`${role}.json`))).records;roleStats.push([role,rows.length,new Set(rows.map((r:any)=>r.word)).size,rows.reduce((n:number,r:any)=>n+r.analyses.length,0),new Set(rows.flatMap((r:any)=>r.analyses.map((a:any)=>a.root)).filter(Boolean)).size,new Set(rows.flatMap((r:any)=>r.analyses.map((a:any)=>a.lemmaFamily))).size,rows.reduce((n:number,r:any)=>n+r.word.length,0)]);}
const ambiguity=(await json(path.join(ROOT,'packages/training/data',split.directory,'train.json'))).records.reduce((h:any,r:any)=>{const key=r.analyses.length===1?'1':r.analyses.length<=3?'2–3':r.analyses.length<=10?'4–10':'11+';h[key]=(h[key]??0)+1;return h;},{});
const comparisons=[],candidateDirs=(await readdir(path.join(ROOT,'packages/training/candidates'))).sort(),measuredResults=[];
for(const file of (await readdir(results)).filter(f=>f.startsWith('correctness-')).sort().reverse())measuredResults.push({file,result:await json(path.join(results,file))});
for(const directory of candidateDirs.filter(d=>!candidateDirs.includes(d+'-export-v2'))){
 const dir=path.join(ROOT,'packages/training/candidates',directory),r=await json(path.join(dir,'result.json')),manifest=await json(path.join(dir,'manifest.json'));
 const evidence=measuredResults.find(e=>e.result.model.id===manifest.id),evaluation=evidence?.result.correctness.teacherAgreement;
 comparisons.push([evidence?`[${directory}](packages/benchmark/results/${evidence.file})`:directory,num(manifest.trainingParameters),num(r.training.bestEpoch),r.training.verificationLoss.toFixed(6),pct(evaluation?.segmentationExactMatch.value),pct(evaluation?.rootExactMatch.value),pct(evaluation?.fullAnalysisTop3.value)]);
}
const metrics=[['Segmentation exact match',q.segmentationExactMatch],['Root exact match',q.rootExactMatch],['Pattern exact match',q.patternExactMatch],['POS accuracy',q.posAccuracy],['Full analysis / top-1',q.fullAnalysisTop1],['Full analysis / top-3',q.fullAnalysisTop3],['Root character accuracy',q.rootCharacterAccuracy],['Coverage (no abstention)',q.coverage],['Error among covered predictions',q.risk]];
let text=`# TinySarf model card

Generated by \`pnpm docs:generate\` from [the frozen report index](packages/benchmark/reports/current.json). **Experimental, unpromoted.** The scientific and public-release criteria remain unmet. Teacher agreement is measured; independent human gold accuracy is unavailable. No active checkpoint or npm release exists.

## Model

- Public package version: ${(await json(path.join(ROOT,'packages/core/package.json'))).version} ([experimental GitHub prerelease](https://github.com/AhmedAbdel-Aal/tinySarf/releases/tag/v0.1.0-experimental.0); no npm registry release).
- Promoted checkpoint and timestamp: **none**. Experimental candidate: \`${m.id}\`.
- Architecture: ${m.format}; embedding width ${m.embedding}, convolution width ${m.width}, dilations ${m.dilations.join('/')}. Three full convolutions; no affine scan in this experiment.
- Training parameters: ${num(m.trainingParameters)}. Reachable deployed weights: ${num(m.reachableWeights)}. The zero padding embedding row is excluded from the reachable count.
- Quantization: ${m.quantization}, float32 scales, four-byte aligned tensors; ${num(m.packedWeightsBytes)} packed bytes. No QAT.
- Input: one MSA word, up to 32 normalized Arabic letters; logical batches up to 2,048. UTF-16 original-input offsets, attached combining marks and tatweel, no spelling folding. [Exact contract](docs/CONTRACT.md).
- Output: ${m.labels.segmentation.join(', ')} spans; root, pattern, POS, and person/gender/number/aspect/mood/voice/case/state. Fixed class order is in the checkpoint manifest. Lemma output is deferred. Missing/unknown/not-applicable are distinct internally; public features omit these values. Ranking scores are not probabilities.

${table(['Component','Raw bytes','Gzip bytes','Brotli bytes'],[['Runtime, minified',num(sz.wrapper.raw),num(sz.wrapper.gzip),num(sz.wrapper.brotli)],['Checkpoint JS module',num(sz.modelModule.raw),num(sz.modelModule.gzip),num(sz.modelModule.brotli)],['Packed model binary',num(sz.modelBinary.raw),num(sz.modelBinary.gzip),num(sz.modelBinary.brotli)],['Manifest',num(sz.modelMetadata.raw),num(sz.modelMetadata.gzip),num(sz.modelMetadata.brotli)],['Complete npm tar',num(sz.completePackage.raw),num(sz.completePackage.gzip),num(sz.completePackage.brotli)],['First-load modules',num(sz.firstLoad.webgpu.raw),num(sz.firstLoad.webgpu.gzip),num(sz.firstLoad.webgpu.brotli)]])}

Non-minified runtime source bundle: ${num(sz.javascriptRawBytes)} bytes; included WGSL source: ${num(sz.wgslSourceBytes)} bytes. Actual npm gzip tarball: ${num(sz.npmTarballBytes)} bytes; unpacked files: ${num(sz.npmUnpackedBytes)} bytes. The complete-package Brotli figure compresses the actual npm tar. No WASM binary is included. ${link('size')}.

## Intended use

Experimental educational, research and search-enrichment tools for isolated Modern Standard Arabic words where errors are acceptable. Inference runs locally after loading static assets. It is unsuitable for religious, legal, medical, security or access-control decisions, automatic grading without review, dialect analysis, syntactic parsing or exhaustive lexicon replacement.

## Training data

Pinned [corpus manifest](packages/training/data/corpus.json): CAMELMORPH MSA database at commit \`15f5aede4b609db54abd6f87aded1f5ec5d930af\` (CC BY 4.0), with the compatible CAMeL Tools fork at \`1a41f9a17e1fa21183537d8b757f2883b1d68060\` (MIT). Source checksums are mandatory before compilation; downloaded databases and generated corpora are reconstructed, not distributed in the package. [Attribution](THIRD_PARTY_NOTICES.md).

This is **teacher-sampled lexical data, not a natural-text distribution**. An initial proper-name-dominated sample was rejected before training; its [audit](packages/training/data/audits/v1-ineligible.json) is retained. No independent human-labelled source is available. The 50 author-curated contract fixtures still require independent human review; the blueprint's pre-training human audit was not completed, so these runs remain exploratory.

${table(['Role','Words','Unique surfaces','Analyses','Distinct non-null roots','Lemma families','Supervised characters'],roleStats.map(row=>row.map((v,i)=>i?num(v as number):v)))}

Mining: ${num(split.roles.mining.words)} words / ${num(split.roles.mining.analyses)} analyses. Sealed final test: ${num(split.roles['final-test'].words)} words / ${num(split.roles['final-test'].analyses)} analyses; **not evaluated**. Train ambiguity buckets: ${Object.entries(ambiguity).map(([k,v])=>`${k} accepted analyses: ${num(v as number)} words`).join('; ')}.

Split related forms by connected lemma families across ambiguous surfaces. Example-connected families are reserved for verification. This ensures lemma-family separation but not natural-distribution representativeness. Root-disjoint behavior is reported as an unseen-root diagnostic subset. [Split manifest](packages/training/data/split-manifest.json), [label mapping and dropped fields](packages/training/data/label-policy.json). Strict morpheme concatenation filters out unalignable teacher analyses and can bias coverage.

## Training procedure

Character and positional embeddings feed three masked ReLU convolutions, masked mean pooling, per-character segmentation and fixed classification heads. Uniformly sample an accepted mappable analysis per word each epoch. Segmentation cross-entropy receives weight 2; the loss averages this and all head losses. AdamW, learning rate ${config.learningRate}, weight decay ${config.weightDecay}, batch ${config.batchSize}, seed ${config.seed}, gradient clipping at 1. Deterministic CPU execution with ${config.threads} threads. No curriculum or replay was used in this candidate.

${table(['Run / export','Parameters','Selected epoch','Verification loss','Segmentation','Root','Full top-3'],comparisons)}

The smaller models were re-exported to correct legacy metadata without changing trained tensors; the [export audit](packages/training/data/audits/legacy-exports.json) retains the reason and original manifest hashes. All comparison scores use the same frozen verification words.

The selected candidate continued the initial 250K run for ${config.epochs} additional epochs and selected continuation epoch ${training.training.bestEpoch} by lowest deterministic sampled verification loss. Optimizer state resets for continuation. All later, worse epochs remain in the saved history. Selection never used the sealed final test. This small data sample has not established useful root or full-analysis quality.

Environment: PyTorch ${config.torch}, NumPy ${config.numpy}; training device ${config.device}; initial machine ${b.environment.cpu}, ${num(b.environment.ramBytes)} bytes RAM. Pinned Python dependencies are in [requirements.lock](packages/training/requirements.lock). The separate CPU/NumPy reference defines exported semantics; int8 uses independent symmetric per-tensor rounding.

## Evaluation

Contract \`${c.data.normalizationVersion}\`; all valid mappable analyses are accepted. Field agreement can use different accepted analyses; full-analysis agreement requires one complete accepted analysis. Top-3 is an approximate beam, not exhaustive ambiguity recovery. Confusion diagnostics choose an allowed reference as documented in [metrics.ts](packages/benchmark/src/metrics.ts). Macro-F1 averages classes with truth or prediction support; feature diagnostics include an explicit absent class. Bootstrap: ${c.protocol.bootstrap.resamples} resamples, seed ${c.protocol.bootstrap.seed}, word-level percentile 95% intervals. Per-class confusion/support and all predictions are retained in ${link('correctness')}.

### Teacher agreement

Frozen verification: ${num(q.count)} words, SHA-256 \`${c.data.verificationDigest}\`. This is repeated model-selection data. It is not an untouched final-test accuracy claim.

${table(['Metric','Agreement'],metrics.map(([name,r])=>[name,rate(r)]))}

Boundary precision / recall / F1: ${pct(q.boundary.precision)} / ${pct(q.boundary.recall)} / ${pct(q.boundary.f1)}. Span-type macro-F1: ${pct(q.spanTypes.macroF1)}. POS macro-F1: ${pct(q.pos.macroF1)}. Feature macro-F1: ${pct(q.featureMacroF1)}. Calibration / ECE / Brier: TBD — not measured; score semantics remain ranking.

${table(['Feature','Accuracy','Macro-F1'],Object.entries(q.features).map(([k,v]:[string,any])=>[k,rate(v.accuracy),pct(v.macroF1)]))}

${table(['Method, same words','Segmentation','Root','Full top-1','Full top-3'],[['Int8 candidate',pct(q.segmentationExactMatch.value),pct(q.rootExactMatch.value),pct(q.fullAnalysisTop1.value),pct(q.fullAnalysisTop3.value)],...Object.entries(c.correctness.baselines).map(([name,v]:[string,any])=>[name,pct(v.segmentationExactMatch.value),pct(v.rootExactMatch.value),pct(v.fullAnalysisTop1.value),pct(v.fullAnalysisTop3.value)])])}

${table(['Diagnostic slice','Words','Segmentation','Root','Full top-3'],Object.entries(c.correctness.slices).map(([name,v]:[string,any])=>[name,num(v.count),pct(v.metrics.segmentationExactMatch.value),pct(v.metrics.rootExactMatch.value),pct(v.metrics.fullAnalysisTop3.value)]))}

### Gold accuracy

${table(['Dataset','Words','Segmentation','Root','Full top-3'],[...(['goldNatural','goldBalanced'] as const).map(key=>{const g=c.correctness[key]?.metrics;return [key,g?num(g.count):'TBD — not measured',g?rate(g.segmentationExactMatch):'TBD — not measured',g?rate(g.rootExactMatch):'TBD — not measured',g?rate(g.fullAnalysisTop3):'TBD — not measured'];})])}

Independent annotations are absent. Teacher agreement and author-curated fixtures cannot substitute for these results. The [gold template](packages/training/data/gold-template.json) and loader enforce independent review provenance and explicit release-candidate access. The teacher final-test set is also unopened for evaluation.

Float-to-int8 paired changes (percentage points):

${table(['Metric','Int8 minus float','95% paired interval'],Object.entries(c.correctness.floatToQuantizedDeltas).map(([name,d]:[string,any])=>[name,d.difference==null?'TBD':(d.difference*100).toFixed(3),d.ci95?`${(d.ci95.low*100).toFixed(3)} to ${(d.ci95.high*100).toFixed(3)}`:'TBD']))}

Quantized CPU/WebGPU parity: ${num(p.parity.headArgmaxMatches)} / ${num(p.parity.headArgmaxCount)} matching head argmax labels; ${num(p.parity.completeAnalysisMatches)} / ${num(p.parity.completeAnalysisCount)} complete analysis sets match. ${p.deterministic.runs} repeated warm calls deterministic: ${p.deterministic.deterministic}. Browser input/maximum-shape/device-loss checks: ${p.contracts.passed?'pass':'fail'}. Tensor error maxima, Python/CPU comparison and disagreements: ${link('parity')}. PyTorch/exported float max absolute error: ${Math.max(...Object.values((await json(path.join(selected.directory,'reference-parity.json'))).tensors).map((t:any)=>t.floatMaxAbsoluteError))}; tolerance 1e-4. Quantization can change labels and complete analyses.

## Browser performance

Device: ${b.environment.deviceModel}, ${b.environment.cpu}, macOS ${b.environment.osVersion}; ${b.environment.browser} Chromium via Brave, headless=${b.environment.headless}; adapter ${JSON.stringify(b.environment.adapter)}. Plugged in: ${b.environment.pluggedIn}. One integrated GPU only; discrete GPU, Android, Safari and Firefox are **TBD — not measured**. CPU comparison is the straightforward JavaScript reference; WASM crossover is **TBD — not measured**.

Protocol: ${b.protocol.mode}; ${b.protocol.coldRuns} fresh-context cold runs per backend; minimum ${b.protocol.warmups} warmups, ${b.protocol.iterations} recorded calls per shape, ${b.protocol.repeats} repetition(s). Release qualifying: **${b.protocol.releaseQualifying}**. Warmup stability assessed: ${b.protocol.warmupStabilityAssessed}; ${b.protocol.workloadSubset?'explicit workload subset only':'all frozen workloads'}. Canonical times surround the public API including normalization, scheduling, GPU work, readback and CPU decoding. Hashing and test validation occur outside timing. Static modules load from loopback with no-store; driver shader-cache state is uncontrolled. These are local experimental measurements, not internet cold-load claims.

${table(['Backend cold','API p50 / p95 ms','Import+API p50 / p95 ms'],Object.entries(b.cold.summary).map(([name,v]:[string,any])=>[name,`${num(v.apiMs.p50)} / ${num(v.apiMs.p95)}`,`${num(v.totalMs.p50)} / ${num(v.totalMs.p95)}`]))}

${table(['Workload','Batch','GPU p50 / p95 ms','JS CPU p50 / p95 ms','GPU speedup','GPU words/sec'],b.warm.map((w:any)=>[w.id,w.batch,`${num(w.webgpu.p50)} / ${num(w.webgpu.p95)}`,`${num(w.cpu.p50)} / ${num(w.cpu.p95)}`,`${w.speedup.toFixed(2)}×`,num(w.webgpuWordsPerSecond)]))}

Cases where WebGPU is slower: ${b.warm.filter((w:any)=>w.speedup<1).map((w:any)=>w.id).join(', ')||'none in this measured artifact; this does not establish universal superiority'}. GPU peak allocated buffers: ${num(b.diagnostic.peakAllocatedGPUBytes)} bytes. Browser heap readings are implementation-specific; total process/driver memory is not measured. Phase timings are diagnostic and are not claimed as GPU timestamp-query measurements. Raw samples, distributions, hashes, padding waste and power state: ${link('browser')}.

Exact recorded command: \`${b.commands.join('; ')}\`.

## Limitations

Root and full-analysis agreement remain too low for promotion. Many missing challenge slices have no evidence, especially independently reviewed diacritized, name, foreign and non-MSA data. Isolated unvowelled words have multiple readings; top-3 misses many. The teacher can be wrong and filtering can remove valid analyses. Lexical sampling and absent human auditing limit generalization claims. Rare/unseen roots, broken plurals, spelling variants and dialects are unreliable. The first encoder ignores diacritic distinctions. Scores have no calibrated probability semantics or validated abstention threshold. Int8 changes some outputs. Browser support, GPU limits, background load and driver caches affect latency; cold and tiny-call overhead are separate from batch gains. The JavaScript reference is not a competitive optimized WASM baseline.

## Reproducibility

Experimental checkpoint: [${path.relative(ROOT,selected.directory)}](${path.relative(ROOT,selected.directory)}). Active location: [packages/training/active](packages/training/active) currently contains no promoted checkpoint. Every run remains available; no bad run was silently deleted.

\`pnpm setup\` installs locked dependencies. \`pnpm data:prepare\` reconstructs pinned teacher data and checks frozen split digests. \`pnpm train --all\` reproduces the three initial target sizes; \`pnpm train --target 250000 --epochs 24 --resume <initial-run>/checkpoint.pt\` reproduces the continuation. CPU timing and exact training run IDs differ by machine; deterministic labels, dataset hashes and reference correctness are the reproducibility targets.

\`pnpm build:core\`, \`pnpm test\`, \`pnpm check:package\`, \`pnpm benchmark:correctness\`, \`pnpm benchmark:size\`, and \`pnpm benchmark:browser --smoke\` exercise the pipeline. For full timings omit \`--smoke\`; the synthetic maximum is expensive on JavaScript. Set \`BROWSER_PATH\` to a WebGPU-capable Chromium executable. \`--workloads single-short,batch-8,batch-32,batch-128\` selects a practical subset and explicitly prevents release-qualifying claims.

\`pnpm reproduce\` validates a separate clean checkout, reconstructs data, evaluates and records hashes. \`pnpm fine-tune --add <reviewed-failure.json> --run\` admits only human-approved mining/external examples disjoint from held-out word and lemma identities. \`pnpm model:promote --check\` produces a hashed eligibility report and cannot activate a checkpoint. Actual promotion reruns both candidate and active on identical verification data, enforces [the policy](packages/training/promotion-policy.json), and atomically changes the active pointer only when every gate passes. Independent gold and clean reproduction are mandatory; current quality floors fail.

${table(['Artifact','Source commit','Dirty source','SHA-256'],Object.entries(index.artifacts).map(([kind,e]:[string,any])=>[link(kind),loaded[kind].git.commit??'Uncommitted initial experiment',String(loaded[kind].git.dirty),e.sha256]))}

Checkpoint SHA-256: \`${m.sha256}\`. Source manifest and per-split hashes: [split manifest](packages/training/data/split-manifest.json). An uncommitted training origin cannot be retroactively made clean; later clean-clone runtime reproduction is separate evidence. Generated tables never rewrite historical provenance.

## Ethical and licensing notes

Code and authored fixtures: MIT. Teacher data and derived checkpoint attribution: CC BY 4.0; preserve CAMeL Lab attribution and the license notices when redistributing the model. Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for pinned sources and model/data terms. Local inference keeps submitted words in the browser; hosting still serves static assets. Incorrect morphology can mislead learners or downstream systems. Human review remains necessary where errors matter.
`;
const evidenceLink=(kind:string)=>`[${kind} evidence](${index.supplemental[kind].file})`;
const ai=supplemental['ai-review'];
const aiSection=ai?`### AI-reviewed diagnostic agreement

${ai.counts.inputs} curated inputs were annotated independently of model and teacher outputs by two AI agents and adjudicated by a third. This is **AI-reviewed, not human gold**, and does not satisfy human-review promotion requirements. Labels cover roots, POS and span boundaries only; accepted values are scored separately, so these numbers are not full-analysis accuracy. AI reviewers can share correlated errors. ${evidenceLink('ai-review')}.

${table(['Field','Top-1 agreement','Top-3 field recall'],Object.entries(ai.metrics.all).map(([field,v]:[string,any])=>[field,rate(v.top1),rate(v.top3)]))}

${table(['Same-input method','Segmentation top-1','Canonical root top-1','POS top-1'],[['Int8 candidate',pct(ai.metrics.all.segmentation.top1.value),pct(ai.metrics.all.rootCanonical.top1.value),pct(ai.metrics.all.pos.top1.value)],...Object.entries(ai.baselines).map(([name,v]:[string,any])=>[name,pct(v.all.segmentation.top1.value),pct(v.all.rootCanonical.top1.value),pct(v.all.pos.top1.value)])])}

Valid morphology inputs: ${ai.counts.valid}; input-rejection cases: ${ai.counts.rejections} (${rate(ai.metrics.inputRejection)}). Normalized surface overlap: ${Object.entries(ai.counts.overlap).map(([role,count])=>`${role}: ${count}`).join('; ')}. The raw report separates high-confidence and unseen-training-surface results. Surface separation does not establish lemma-family separation. Bare-hamza canonical root agreement is a separate secondary diagnostic; exact radical spelling remains reported.

The [annotation audit](packages/training/data/ai-review/README.md) preserves source labels, disagreements, adjudication decisions and input hashes. These labels were not used to retrain or select the current model.

`:'### AI-reviewed diagnostic agreement\n\nTBD — not measured. Agent review is separate from independent human gold.\n\n';
text=text.replace('## Browser performance',aiSection+'## Browser performance');
const corrected=supplemental['corrected-comparison'];
if(corrected){
 const experiment=await json(path.join(ROOT,'packages/training/experiments/corrected-pilot-v1/result.json'));
 const dataset=await json(path.join(ROOT,'packages/training/data/audits/corrected-pilot-v1-manifest.json'));
 const section=`### Corrected teacher pilot, retained unselected

An opt-in mapping fixes corrupted non-Arabic pattern labels and teacher connective-particle tags. It preserves the legacy compiler and frozen verification surface order. The release decoder also rejects the malformed legacy pattern and admits attested particle and oath-preposition prefixes. The repaired pilot uses ${num(dataset.roles.train.words)} training words, adds ${num(dataset.supplementWords)} POS-stratified teacher words, and excludes ambiguity-connected held-out and AI challenge families. AI labels were never used for training or selection. [Dataset audit](packages/training/data/audits/corrected-pilot-v1-manifest.json).

The pilot ran ${num((await json(path.join(ROOT,'packages/training/experiments/corrected-pilot-v1/config.json'))).epochs)} epochs and selected epoch ${num(experiment.training.bestEpoch)}. Its bounded sampling retains ordinary ambiguity sampling and adds rare-POS coverage; actual per-epoch counts and losses are preserved in the [history](packages/training/experiments/corrected-pilot-v1/history.json). [Checkpoint and reference-parity artifacts](packages/training/experiments/corrected-pilot-v1), [preparation and training CLI](packages/training/src/tinysarf_training/corrected.py).

Both models below use the same ${num(dataset.roles.verification.words)} words and **corrected** labels. These figures are not directly comparable to the legacy-label headline. Mapping, training coverage and sampling changed together, so this is not a single-factor ablation. ${evidenceLink('corrected-comparison')}.

${table(['Metric','Current checkpoint','Corrected pilot','Paired change / 95% CI, percentage points'],Object.entries(corrected.metrics).map(([name,value]:[string,any])=>{const d=corrected.deltas[name];return [name,pct(value.baseline.value),pct(value.candidate.value),`${(d.difference*100).toFixed(2)} (${(d.ci95.low*100).toFixed(2)} to ${(d.ci95.high*100).toFixed(2)})`];}))}

The pilot remains **unselected** because root and full-analysis agreement regressed. Float reference parity passed; separate browser hardware qualification of this rejected pilot was not performed. The complete run is retained for investigation. Reconstruct with \`.venv/bin/python -m tinysarf_training.corrected prepare\`, then run \`.venv/bin/python -m tinysarf_training.corrected pilot --dataset packages/training/data/generated/teacher-corrected-pilot-v1\`. Neither command changes the selected model.

`;
 text=text.replace('## Evaluation',section+'## Evaluation');
}
text=text.replace('## Ethical and licensing notes',`### Validation evidence\n\n${supplemental.reproduction?`Clean-clone reproduction passed for source commit \`${supplemental.reproduction.git.commit}\`. It reconstructed teacher data and repeated correctness, package size and real WebGPU parity. ${evidenceLink('reproduction')}.`:'Clean reproduction: TBD — not measured.'}\n\n${supplemental.eligibility?`The guarded promotion check rejected the experimental candidate. The report records each failed quality, human-gold and hardware/protocol gate. ${evidenceLink('eligibility')}.`:''}\n\n## Ethical and licensing notes`);
await writeFile(path.join(ROOT,'MODEL_CARD.md'),text);
const readme=await readFile(path.join(ROOT,'README.md'),'utf8');
const compact=`Experimental checkpoint \`${m.id}\` — **unpromoted**. [Model card](MODEL_CARD.md), [result index](packages/benchmark/reports/current.json).

${table(['Measured item','Result','Evidence'],[['Reachable deployed weights',num(m.reachableWeights),link('size')],['Complete package Brotli',`${num(sz.completePackage.brotli)} bytes`,link('size')],['Teacher segmentation agreement',`${pct(q.segmentationExactMatch.value)} (${q.count} verification words)`,link('correctness')],['Gold segmentation / root / top-3','TBD — not measured; no gold samples',link('correctness')],['Reference batch throughput',batch?`${num(batch.webgpuWordsPerSecond)} words/s at batch 128 (${b.protocol.mode}; JS-reference comparison)`:'TBD — not measured',link('browser')]])}`;
await writeFile(path.join(ROOT,'README.md'),readme.replace(/<!-- GENERATED:RESULTS:START -->[\s\S]*?<!-- GENERATED:RESULTS:END -->/,`<!-- GENERATED:RESULTS:START -->\n${compact}\n<!-- GENERATED:RESULTS:END -->`));
const website=path.join(ROOT,'apps/website');await writeFile(path.join(website,'lib/results.json'),JSON.stringify({entries:[{label:'Reachable weights',value:num(m.reachableWeights),note:`Int8 · ${num(m.packedWeightsBytes)} packed bytes`},{label:'Teacher segmentation agreement',value:pct(q.segmentationExactMatch.value),note:`${num(q.count)} teacher-sampled verification words`},{label:'Independent gold accuracy',value:'Not measured',note:'Independent human annotation is required'},{label:'WebGPU batch advantage',value:batch?`${batch.speedup.toFixed(2)}×`:'Not measured',note:`Batch 128 vs JavaScript reference · ${b.protocol.mode}`}],note:'Experimental, unpromoted. Root and full-analysis quality remain below release targets. One headless Chromium / Apple M2 measurement; not a cross-browser claim.'+(ai?` An additional ${ai.counts.inputs}-input challenge was independently annotated by two AI agents and adjudicated by a third; see the model card for AI agreement and its limitations.`:''),artifact:'/results/current.json'},null,2)+'\n');
await mkdir(path.join(website,'public/docs'),{recursive:true});await mkdir(path.join(website,'public/results'),{recursive:true});
const publicEntry=(e:any)=>({...e,file:`/results/${path.basename(e.file)}`});
const webIndex={...index,artifacts:Object.fromEntries(Object.entries(index.artifacts).map(([k,e])=>[k,publicEntry(e)])),supplemental:Object.fromEntries(Object.entries(index.supplemental??{}).map(([k,e])=>[k,publicEntry(e)]))};await writeFile(path.join(website,'public/results/current.json'),JSON.stringify(webIndex,null,2)+'\n');
for(const e of [...Object.values(index.artifacts),...Object.values(index.supplemental??{})] as any[])await copyFile(path.join(ROOT,e.file),path.join(website,'public/results',path.basename(e.file)));
// Documentation links must remain usable when served outside the repository.
const published=new Set<string>();
async function publishDocument(file:string,destination:string):Promise<void>{
 if(published.has(destination))return;published.add(destination);await mkdir(path.dirname(destination),{recursive:true});
 if(!file.endsWith('.md')){await copyFile(file,destination);return;}
 let markdown=await readFile(file,'utf8');
 for(const match of [...markdown.matchAll(/(?<!!)\[[^\]]*\]\(([^)]+)\)/g)]){
  const target=match[1];if(/^(https?:|mailto:|#|\/)/.test(target))continue;
  let source=path.resolve(path.dirname(file),target);if(!source.startsWith(ROOT+path.sep))throw new Error('Documentation link escapes repository');
  if((await stat(source)).isDirectory()){const manifest=path.join(source,'manifest.json');try{await stat(manifest);source=manifest;}catch{source=path.join(source,'README.md');}}
  const relative=path.relative(ROOT,source),url=relative==='packages/benchmark/reports/current.json'?'/results/current.json':`/source/${relative}`;
  if(url!=='/results/current.json')await publishDocument(source,path.join(website,'public',url));
  markdown=markdown.replaceAll(`](${target})`,`](${url})`);
 }
 await writeFile(destination,markdown);
}
for(const file of ['MODEL_CARD.md','architecture.md','THIRD_PARTY_NOTICES.md'])await publishDocument(path.join(ROOT,file),path.join(website,'public/docs',file));
console.log(`Generated model card, README and website evidence for ${m.id}`);
