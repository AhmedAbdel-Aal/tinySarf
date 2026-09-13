# Root identification redesign

The selected local experimental candidate is `root-rethink-20260913`. Exact root agreement rises from **45.34% to 78.24%** on the same 1,094 verification words. Segmentation stays at **87.02%**. Every top-1 segmentation, POS, pattern and feature output is unchanged, and every non-root tensor retains the original packed bytes and scale. This is teacher agreement on repeatedly used development data, not independent human accuracy.

## What was wrong

The old model predicted four radicals from a shared mean-pooled representation. It had no explicit route for copying the word's consonants, and its encoder also served segmentation and the other morphology heads. The root training signal was limited to teacher analyses whose segmentation mapped cleanly to the surface. Those requirements are different: useful root supervision can survive even when the teacher's segment strings do not concatenate correctly.

The high segmentation score did not establish that the encoder understood root formation. A whole-word stem is an accepted segmentation for 632 of the 1,094 development words. Root recovery still requires distinguishing derivational letters and restoring weak radicals. The data also allow multiple distinct roots for 405 words, so the loss must handle complete alternatives consistently.

## Implemented change

- Replace the old root classifiers with a separate 16-dimensional character embedding and three width-48 convolutions. Four attention heads copy letters from the input; four small generators can restore absent radicals. Repeated copying supports doubled roots. The non-root model stays frozen.
- Train on 27,780 teacher-labelled root words, including usable root analyses rejected by segmentation alignment. Deduplicate accepted complete roots and optimize their summed probability. Duplicate analyses no longer give a root extra weight, and the objective does not treat arbitrary combinations of different accepted roots as targets.
- Induce 1,784 whole-word and 674 stem transformations from training. Each rule copies at least two distinct input positions and is supported by at least three distinct roots; one absent radical may be restored. The decoder uses the same predicted stem as the public analysis, with a learned neural fallback. There is no runtime word-to-root dictionary.
- Implement the new operations in both JavaScript CPU and WGSL. The GPU adds six passes and reuses existing buffers. Total trainable parameters fall from 245,699 to 245,111; 245,063 weights are reachable at inference.

The copying/generation idea follows the general [pointer-generator construction](https://aclanthology.org/P17-1099/); the TinySarf architecture and measurements here are specific to this implementation.

## Same-input results

| Slice | Original exact root | New exact root |
| --- | --- | --- |
| all | 496/1094 · 45.34% | 856/1094 · 78.24% |
| root-required | 209/666 · 31.38% | 504/666 · 75.68% |
| weak-root | 181/427 · 42.39% | 280/427 · 65.57% |
| hamzated | 14/53 · 26.42% | 37/53 · 69.81% |
| doubled | 111/191 · 58.12% | 158/191 · 82.72% |
| quadriliteral | 11/59 · 18.64% | 42/59 · 71.19% |
| clitic-stack | 160/347 · 46.11% | 269/347 · 77.52% |

There are **402 fixes and 42 regressions**, reducing root errors by **60.20%**. The paired improvement is **32.91 percentage points**, with a word-bootstrap 95% interval of **29.71 to 36.20 points**. Exact radical spelling and null handling are unchanged. “root-required” includes only words whose every accepted analysis has a non-null root; the improvement is not just a gain on absent-root labels.

[Paired report and changed words](../packages/benchmark/results/root-comparison-2026-09-13T183909338Z.json), [complete float/int8 results and predictions](../packages/benchmark/results/correctness-2026-09-13T183855624Z.json), [selected configuration and frozen-tensor audit](../packages/training/candidates/root-rethink-20260913/config.json).

The comparison's unseen-root slices use the original training-root set for both models. Ordinary correctness slices use the selected root training pool. These different definitions are explicit and must not be compared as if they were the same population. The unseen-root word slices also include null-only cases because no non-null training root matches; null-only and root-required results are reported separately.

## Experiments retained

The following are development experiments, not independent estimates or a complete factorial study. Changes to data, architecture, objective and decoding are partly combined. Neural checkpoints use minimum verification marginal loss; decoder settings use verification root agreement. All later, worse epochs and rejected runs are retained.

| Experiment | Float exact root agreement |
| --- | --- |
| Original pooled heads | 45.34% |
| Copying on frozen encoder, original corrected pool | 54.94% |
| Copying on frozen encoder, expanded root pool | 59.05% |
| Small root CNN with inherited generators, expanded pool | 67.00% |
| Independent root CNN and generators, expanded pool | 74.04% |
| Original neural heads with whole-word and stem transformations | 72.12% |
| Independent root CNN plus transformations, selected | 78.15% |
| Larger teacher-compatible inflection pool, independent root CNN | 64.81% |
| Larger inflection pool, independent root CNN plus transformations | 75.50% |
| Selected neural network with larger-pool transformations | 77.79% |

Int8 changes the selected combined result from 855 to 856 correct words. The larger sampler produced 72,710 retained words with teacher-compatible verb prefixes and suffixes. It improved some individual slices but reduced overall agreement, so it remains unselected. This does not show that inflection coverage is unhelpful; it shows this sampling/training recipe did not improve the selection metric.

[All experiment artifacts](../packages/training/experiments/root-rethink/README.md), [selected data audit](../packages/training/data/audits/teacher-roots-v1-60000-manifest.json), [larger-data audit](../packages/training/data/audits/teacher-roots-v2-120000-manifest.json).

## Isolation and reproducibility

Held-out surfaces and complete raw teacher lemma identities are excluded from new root training, including connected ambiguity aliases. This also excludes the curated AI challenge identities and contract fixture identities. Only protected identities are used for exclusion; final-test labels and AI annotation labels are not used as training targets. The inherited non-root encoder retains its original data provenance.

A fresh training run reproduced the selected epoch-12 tensors **exactly**, and all losses and root metrics through epoch 12 match. The original 40-epoch history remains available. [Training reproduction audit](../packages/training/data/audits/root-training-reproduction.json). A separate clean checkout of source commit `50e47c25cb6d6e505d22c3b722d82d5696417a16` reconstructed identical training data, CPU predictions and tarball bytes, and passed real WebGPU parity. [Clean-checkout evidence](../packages/benchmark/results/reproduction-2026-09-13T185523137Z.json).

From the repository root:

```sh
pnpm setup
pnpm data:prepare
.venv/bin/python -m tinysarf_training.root_data --selected
# Reproduce all epochs through the selected weights (40 reproduces the full exploratory run):
.venv/bin/python -m tinysarf_training.root_experiment --epochs 12 --mode standalone --dataset packages/training/data/generated/teacher-roots-v1-60000
.venv/bin/python -m tinysarf_training.root_stems --dataset packages/training/data/generated/teacher-roots-v1-60000 --output packages/training/runs/reproduced-root-stems
.venv/bin/python -m tinysarf_training.root_templates --dataset packages/training/data/generated/teacher-roots-v1-60000 --root-run <new-root-run> --stem-templates packages/training/runs/reproduced-root-stems --output packages/training/runs/reproduced-root-decoder
.venv/bin/python -m tinysarf_training.root_export --root-run <new-root-run> --decoder-run packages/training/runs/reproduced-root-decoder --output packages/training/runs/reproduced-root-model
```

Each output directory must be new. Training and export do not change the selected pointer. `tinysarf_training.select <exported-run>` stages an experimental candidate; it does not promote an active model. Generic multi-task continuation rejects root checkpoints because it would discard the root-only supervision protocol; use the dedicated root commands.

`pnpm test`, `pnpm check:package`, `pnpm benchmark:correctness`, `pnpm benchmark:size`, and `pnpm benchmark:browser --workloads single-short,batch-8,batch-32,batch-128` exercise the final runtime. The workload subset is explicitly not release-qualifying. `pnpm reproduce` verifies a separate clean checkout and reconstructs frozen data. [Current model card and hardware evidence](../MODEL_CARD.md).

## Runtime cost

The local `0.1.0-experimental.1` tarball is 258,057 bytes; complete-package Brotli is 239,894 bytes. Float/NumPy, JavaScript CPU and real WebGPU checks pass, including irregular and maximum input sizes, device recovery and repeated-call determinism. All 55 tests and TypeScript checks pass. [Package measurements](../packages/benchmark/results/size-2026-09-13T183853714Z.json), [hardware parity](../packages/benchmark/results/parity-2026-09-13T184417380Z.json).

Fresh original and candidate timing runs were executed sequentially with model training stopped, on the same Apple M2 / headless Chromium device. Both runs reported battery power. Each uses 20 cold calls per backend and 100 warm calls per workload across three repetitions. The root redesign adds latency; these local measurements are not a broad hardware or release qualification.

| Workload | WebGPU p50 ms, original / new | JS CPU p50 ms, original / new |
| --- | --- | --- |
| single-short | 1.09 / 1.23 | 0.98 / 1.10 |
| batch-8 | 5.96 / 6.72 | 11.09 / 12.70 |
| batch-32 | 23.25 / 25.30 | 48.66 / 56.08 |
| batch-128 | 92.71 / 97.38 | 201.86 / 227.49 |

[Original timing samples](../packages/benchmark/results/browser-2026-09-13T184324423Z.json), [candidate timing samples](../packages/benchmark/results/browser-2026-09-13T184740542Z.json). CPU means the JavaScript reference, not an optimized WASM implementation. Background operating-system activity and driver cache state are not fully controlled.

## Remaining limitations

The curated AI diagnostic improves from 27/76 to 44/76 exact roots (**35.53% to 57.89%**), or 29/76 to 53/76 after its separately reported hamza-only canonicalization (**38.16% to 69.74%**). This challenge was inspected during development; it is not an untouched holdout or human gold. New root training has zero normalized surface overlap with it. The inherited multi-task pool has one overlapping surface. [AI diagnostic](../packages/benchmark/results/ai-review-2026-09-13T183856172Z.json).

Common weak forms still fail, including `يقول` and `يبيع` in the current package. Root spelling follows the pinned teacher and can differ from a pedagogical convention (for example final `ى` versus `ي`). No evaluation spelling rule was changed to raise the headline score. Ambiguous isolated words, rare roots and out-of-distribution inputs remain difficult. POS and full-analysis quality still need work. Independent human annotation and broader hardware validation remain absent, and the candidate remains experimental and unpromoted.
