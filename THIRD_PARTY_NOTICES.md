# Third-party notices

TinySarf's original source code is MIT licensed. The experimental teacher-derived weights are distributed under CC BY 4.0 with the attribution below. Downloaded corpora and teacher code retain their original licenses and are not included in the browser runtime.

- **CAMELMORPH MSA v1.0** — CAMeL Lab, New York University Abu Dhabi. Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); code: MIT. Source: [CAMeL-Lab/camel_morph](https://github.com/CAMeL-Lab/camel_morph), pinned commit `15f5aede4b609db54abd6f87aded1f5ec5d930af`. Cite Christo Khairallah et al., [Camel Morph MSA: A Large-Scale Open-Source Morphological Analyzer for Modern Standard Arabic](https://aclanthology.org/2024.lrec-main.240/), LREC-COLING 2024. TinySarf transforms teacher analyses into a reduced label schema, filters alignment failures, samples and groups surfaces, and distills a new CNN. The authors of the teacher have not endorsed TinySarf.
- **CAMeL Tools compatible fork** — CAMeL Lab and contributors; Christo Khairallah. MIT. [christios/camel_tools](https://github.com/christios/camel_tools), pinned commit `1a41f9a17e1fa21183537d8b757f2883b1d68060`. This fork is used because CAMELMORPH's release documentation calls for its compatible analyzer.
- **gpu-lexer** — Shu Ding / Vercel Labs, MIT. [Repository](https://github.com/vercel-labs/gpu-lexer). Engineering inspiration; TinySarf does not reuse its model, datasets, labels, benchmark numbers or shader implementation.
- **PyTorch**, **NumPy**, and Python teacher dependencies are offline development dependencies. Exact installed versions are recorded in each training run and the training lock file.
- The separate website uses React, the Sites/Vinext starter, Base UI/Shadcn, Tailwind, and their dependencies under their respective package licenses. These are not part of the `tinysarf` inference runtime.

Pinned source URLs, checksums, build instructions, sampling policy, and split roles are in `packages/training/data/corpus.json`. This notice must accompany redistributed experimental weights.
