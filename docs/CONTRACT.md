# TinySarf contract v1

The sole package export is `analyze(word, options?)`, with `analyze.batch(words, options?)`. A call returns an array of up to `topK` candidates (1–3; default 1). Batch preserves input order. Empty batch returns `[]`; an invalid word rejects the entire batch. Maximum: 32 normalized letters per word and 2,048 words per call.

`arabic-v1` accepts the literal letters in `packages/core/src/normalize.ts`. Arabic combining marks and tatweel following a letter are removed for inference and attached to that letter's original span. Leading marks, whitespace, foreign characters, Unicode presentation ligatures, and decomposed hamza spellings are rejected explicitly. Spelling variants are preserved. There is no implicit trimming, NFKC expansion, or alef/ya folding. Mark-presence and boundaries are packed with letter IDs; the first model uses letter IDs and positions only. Diacritic distinctions are therefore not learned in this architecture.

Public offsets always index the **original input**, in JavaScript UTF-16 code units. Spans cover it exactly, including attached marks. Internally the model uses normalized letter positions and maps boundaries back using the retained offset table. This resolves the blueprint's mixed normalized/original-offset wording in favor of safe `input.slice(start, end)`.

Internally `__missing__`, `__unknown__`, and `__na__` are distinct classes. Public features omit all three; unknown root/pattern become `null`; unknown POS becomes `unknown`. Lemma is omitted in v1. Scores are average log ranking scores, not probabilities. An explicit threshold can return no candidates. Top-k is approximate constrained beam search, and cannot enumerate all valid Arabic analyses.

WebGPU is the default and requires a secure browser context. `{backend: 'cpu'}` explicitly selects the reference. There is no remote inference or implicit fallback. Both backends share a serialized resident runtime and decoder.

The initial 50 contract fixtures are author-curated implementation examples, pending independent human review. They are **not gold evaluation data** and are excluded from optimization, including related lemma families.
