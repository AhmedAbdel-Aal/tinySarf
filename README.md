# TinySarf

Experimental Modern Standard Arabic morphology, executed locally through a small learned WebGPU model. One function returns morpheme spans, roots, patterns, POS and features, with explicit top-k ambiguity.

**Status: experimental, unpromoted.** No independent human gold evaluation is available. This repository implements the research and deployment pipeline; it does not claim that the blueprint's scientific or public-release success criteria have been met.

```ts
import { analyze } from 'tinysarf';
const alternatives = await analyze('وبكتابهم', { topK: 3 });
const batch = await analyze.batch(['كتاب', 'يكتبون'], { topK: 3 });
```

WebGPU requires a secure browser context. `{ backend: 'cpu' }` explicitly chooses the local reference. There is no remote inference or automatic backend fallback. The package is built locally; it has not been published to npm.

See [the contract](docs/CONTRACT.md), [architecture](architecture.md), [model card](MODEL_CARD.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

<!-- GENERATED:RESULTS:START -->
Experimental checkpoint `20260911T191243.077466Z-250000` — **unpromoted**. [Model card](MODEL_CARD.md), [result index](packages/benchmark/reports/current.json).

| Measured item | Result | Evidence |
| --- | --- | --- |
| Reachable deployed weights | 245,667 | [size-2026-09-11T195058746Z.json](packages/benchmark/results/size-2026-09-11T195058746Z.json) |
| Complete package Brotli | 227,706 bytes | [size-2026-09-11T195058746Z.json](packages/benchmark/results/size-2026-09-11T195058746Z.json) |
| Teacher segmentation agreement | 87.11% (1094 verification words) | [correctness-2026-09-11T191845656Z.json](packages/benchmark/results/correctness-2026-09-11T191845656Z.json) |
| Gold segmentation / root / top-3 | TBD — not measured; no gold samples | [correctness-2026-09-11T191845656Z.json](packages/benchmark/results/correctness-2026-09-11T191845656Z.json) |
| Reference batch throughput | 1,515.78 words/s at batch 128 (smoke; JS-reference comparison) | [browser-2026-09-11T193240105Z.json](packages/benchmark/results/browser-2026-09-11T193240105Z.json) |
<!-- GENERATED:RESULTS:END -->

Development requires Node 22+, pnpm 11, and Python 3.11. Use `pnpm install`, then the setup and reproduction instructions in the model card. The website is a separate package under `apps/website`.
