# TinySarf

Experimental Modern Standard Arabic morphology, executed locally through a small learned WebGPU model. One function returns morpheme spans, roots, patterns, POS and features, with explicit top-k ambiguity.

**Status: public experimental release, unpromoted.** No independent human gold evaluation is available. The blueprint's scientific release criteria have not been met.

[Try TinySarf in your browser](https://tinysarf-lab.ahmedabdouuu.chatgpt.site). Enter an Arabic word and select **Analyze word**. The demo defaults to CPU, with optional WebGPU acceleration. No installation or account is needed.

For your own JavaScript or TypeScript project, install the self-contained experimental package. It includes the checkpoint; no Python, separate model download, account, or API key is needed for inference.

```sh
npm install https://github.com/AhmedAbdel-Aal/tinySarf/releases/download/v0.1.0-experimental.1/tinysarf-0.1.0-experimental.1.tgz
```

In a browser application:

```ts
import { analyze } from 'tinysarf';
const alternatives = await analyze('وبكتابهم', { topK: 3 });
const batch = await analyze.batch(['كتاب', 'يكتبون'], { topK: 3 });
```

WebGPU requires a secure browser context. `{ backend: 'cpu' }` explicitly chooses the local reference. There is no remote inference or automatic backend fallback. The [GitHub release](https://github.com/AhmedAbdel-Aal/tinySarf/releases/tag/v0.1.0-experimental.1) includes checksums and validation evidence. Use the full installation URL above; this package is not on the npm registry.

Try the CPU backend in Node 22+ immediately after installing:

```sh
node --input-type=module -e "import {analyze} from 'tinysarf'; console.log(JSON.stringify(await analyze('وَبِكِتَابِهِمْ',{backend:'cpu',topK:3}),null,2))"
```

For a browser demo from source, run `pnpm install --frozen-lockfile`, `pnpm build:core`, then `node examples/serve.mjs` and open `http://localhost:4173`. [The example](examples/index.html) uses the real package and works with either backend. The complete research workbench is under `apps/website`; `pnpm setup && pnpm dev` starts it locally.

See [the contract](docs/CONTRACT.md), [architecture](architecture.md), [model card](MODEL_CARD.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

[Report a morphology error](https://github.com/AhmedAbdel-Aal/tinySarf/issues/new?template=morphology.yml) with the word, returned JSON and a proposed correction. Reviewed cases can enter the guarded replay bank described in the model card.

<!-- GENERATED:RESULTS:START -->
Experimental checkpoint `root-rethink-20260913` — **unpromoted**. [Model card](MODEL_CARD.md), [result index](packages/benchmark/reports/current.json).

| Measured item | Result | Evidence |
| --- | --- | --- |
| Reachable deployed weights | 245,063 | [size-2026-09-13T185806696Z.json](packages/benchmark/results/size-2026-09-13T185806696Z.json) |
| Complete package Brotli | 239,894 bytes | [size-2026-09-13T185806696Z.json](packages/benchmark/results/size-2026-09-13T185806696Z.json) |
| Teacher segmentation agreement | 87.02% (1094 verification words) | [correctness-2026-09-13T185802812Z.json](packages/benchmark/results/correctness-2026-09-13T185802812Z.json) |
| Teacher root agreement | 78.24% (1094 verification words) | [correctness-2026-09-13T185802812Z.json](packages/benchmark/results/correctness-2026-09-13T185802812Z.json) |
| Gold segmentation / root / top-3 | TBD — not measured; no gold samples | [correctness-2026-09-13T185802812Z.json](packages/benchmark/results/correctness-2026-09-13T185802812Z.json) |
| Reference batch throughput | 1,314.47 words/s at batch 128 (full; JS-reference comparison) | [browser-2026-09-13T184740542Z.json](packages/benchmark/results/browser-2026-09-13T184740542Z.json) |
<!-- GENERATED:RESULTS:END -->

Development requires Node 22+, pnpm 11, and Python 3.11. Use `pnpm install`, then the setup and reproduction instructions in the model card. The website is a separate package under `apps/website`.
