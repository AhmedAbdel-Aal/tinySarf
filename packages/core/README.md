# tinysarf

Experimental Modern Standard Arabic morphology, running locally in the browser with WebGPU. This build contains an **unpromoted research checkpoint**. It has not passed independent human gold evaluation or all release gates. Do not infer linguistic correctness from a well-formed output.

Source, installable experimental releases, the model card and raw measurements are available in [the TinySarf repository](https://github.com/AhmedAbdel-Aal/tinySarf).

```ts
import { analyze } from 'tinysarf';
const alternatives = await analyze('وبكتابهم', { topK: 3 });
const batch = await analyze.batch(['كتاب', 'يكتبون']);
```

WebGPU is the default and needs a secure browser context. Use `{ backend: 'cpu' }` to explicitly select the local reference. There is no server inference or automatic fallback. The model and GPU resources remain resident across calls.

Spans index the original input in UTF-16 code units. Diacritics and tatweel are attached to their preceding letter. The maximum input is 32 Arabic letters per word and 2,048 words per batch; `topK` is 1–3. A finite `threshold` applies to an uncalibrated log ranking score. Unknown roots and patterns are `null`; non-applicable or unknown features are omitted. See [the full contract](docs/CONTRACT.md).

The TinySarf workspace includes the model card, exact teacher/version manifests, training/checkpoint artifacts, correctness and real-browser benchmark runners, and fail-closed promotion commands. This package README contains no changing benchmark figures so the measured npm tarball can be reproduced independently of generated report text.

Source code: MIT. Experimental teacher-derived weights: CC BY 4.0 with attribution in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
