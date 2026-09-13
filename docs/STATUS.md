# Implementation status

The experimental implementation is runnable end to end: pinned data compilation, three model sizes and one continuation, float export, int8 quantization, CPU and WebGPU execution, real-browser parity/timing, local package, website, report generation, reviewed failure intake and guarded promotion.

The blueprint's release definition is not met. These external or research requirements remain open:

- Independent human review of the 50 contract fixtures and the teacher mapping audit.
- Licensed natural-text sampling and independent gold-natural / gold-balanced annotation, with the required challenge slices.
- Better POS and full-analysis quality, plus independent validation of the redesigned root branch. Root teacher agreement rose from 45.34% to 78.24% on the same verification words. These are development results, not gold accuracy; see [the root experiment](ROOT_EXPERIMENT.md).
- A complete controlled reference benchmark plus discrete-GPU, Android and additional browser evidence. The implemented full protocol is available; recorded runs are marked according to their actual scope.
- Promotion of an eligible checkpoint and npm publication. Public historical source and prereleases are separate from the current local experimental candidate. No npm registry package or active model is claimed.

The sealed teacher final-test labels have not been used for model selection or scored. Missing measurements are null / “TBD — not measured”. A clean-clone reproduction validates the frozen checkpoint and runtime; it cannot retroactively establish clean provenance for the original exploratory training session.
