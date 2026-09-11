# AI-reviewed challenge v1 — NOT HUMAN GOLD

This is an AI-reviewed diagnostic artifact. Two separate AI agents produced blind annotations from the frozen input list, followed by a separate AI adjudicator that compared both files and checked selected primary dictionary entries. No human annotated or adjudicated these labels. This data cannot satisfy human-gold gates and is **not eligible for model promotion**.

The source annotations declare independence from model predictions, teacher labels, generated training data and contract fixture labels. The adjudicator used only the frozen list, the two annotations, the public contract/schema, the AIReview interface/validation and targeted external dictionary evidence. No model inference or evaluation was run during adjudication. Procedural blindness does not establish independent human evidence or independence from unknown model pretraining data.

Files: [frozen words](words.json), [blind annotator A](annotator-a.json), [blind annotator B](annotator-b.json), [adjudicated labels](adjudicated.json), and [complete disagreement audit](disagreements.json). Original files remain unchanged. The pending status in the frozen input metadata records its freeze-time state; the completed review is the adjudicated file.

Completed at `2026-09-11T20:38:17Z` by `ai-adjudicator`, after `ai-annotator-a` and `ai-annotator-b`.

## Counts generated from the files

| Measure | Count |
| --- | ---: |
| Frozen inputs / adjudicated records | 82 / 82 |
| Accepted Arabic inputs | 76 |
| Expected input rejections | 6 |
| Words with substantive original disagreements | 27 |
| Root / POS / segmentation / rejection disagreements | 7 / 15 / 13 / 0 |
| Accepted records: high / medium / low confidence | 38 / 34 / 4 |
| Open follow-up cases | 6 |

Disagreement counts compare accepted sets, ignoring their order. Each segmentation preserves its ordered spans and exact type/start/end; omitted expectedError equals false. The field counts overlap across words. Confidence/prose/order-only differences are not counted as substantive disagreements. Every substantive disagreement retains the complete original pair, original field sets, final decision and individual rationale in the audit.

## Scope and conventions

Labels cover only expected rejection, root, coarse POS and segmentation. Pattern and features are intentionally omitted. Accepted values are scored **separately by field**; their Cartesian product is not a set of compatible complete analyses. A noun reading can motivate one root while a verb reading motivates another. This artifact does not establish full-analysis accuracy, exhaustive ambiguity coverage or performance on naturally distributed Arabic.

- Roots use bare `ء` for radical hamza, repeated geminate radicals and underlying `و`/`ي`. Arabic-derived names retain lexical roots; borrowed names and rootless grammatical readings use null. The lexical quantifier `كل` retains `كلل`. `طمأنينة` uses provisional synchronic `طمءن`; historical/metathesized/triliteral alternatives are preserved in the audit but excluded from its accepted root set.
- POS follows the lexical host. Participles with credible substantive and descriptive uses may accept noun/adjective. Generic ellipsis does not automatically add noun to every adjective. Supplied diacritics constrain readings; unvocalized forms may retain independently supported homographs.
- A stem keeps derivational prefixes, imperfective person prefixes, internal broken-plural material and weak-root allomorphs. There is no invented inflectional-prefix category. Transparent attached articles, prepositions, conjunctions and other particles are split where the reading supports them.
- Productive gender, number, case, mood and verbal subject agreement use `inflectional_suffix`. Subject agreement is not relabeled `pronominal_enclitic`, even though traditional grammatical descriptions call subject endings attached pronouns. Here `pronominal_enclitic` identifies object, possessor or complement pronouns. Bundled versus separate agreement/mood endings are retained only where explicitly listed.
- Lexical or verbal-noun `ة` may remain in the stem or be a `derivational_suffix`. Productive feminine agreement may be `inflectional_suffix`. Lexical feminine gender alone does not license all three labels. Construct `ت` inherits the corresponding interpretation. Cairo retains lexical `ة`, or separates it as feminine agreement for the participial reading.
- Spans are end-exclusive offsets in the original JavaScript UTF-16 string, cover it exactly once, and never detach a combining mark from its preceding letter. A silent support alif stays with the plural ending; accusative-support alif may be an inflectional span. Standalone grammatical words still need a `stem` span to meet the schema.
- `في` and `أنا` contain possible fused/shared morphology; the accepted spans are declared surface-allocation conventions. `هذا` may remain whole or split the attention particle. No claim of unique, lossless morpheme recovery is made for these cases.

Adjudication did not simply union the sources: it excluded historical root variants for طمأنينة, generic noun conversion for شديد, null for أكل, blanket feminine-inflection labels on verbal nouns, and subject-agreement enclitic variants. It retained dictionary-supported additional homographs and explicitly identified regular conjugational inferences.

## Open follow-up cases

- **طمأنينة** (roots): Synchronic طمءن is selected; historical/triliteral lexicographic alternatives are excluded pending human agreement on root policy.
- **في** (segmentations): Fused ي can represent base/case/pronoun material. The accepted boundary is an explicit allocation convention, not unique morpheme recovery.
- **هذا** (segmentations): The demonstrative may be lexical whole or attention ه + ذا; a human segmentation policy is needed.
- **أنا** (pos, segmentations): Particle-headed contracted أَنّا is included, but shared ن prevents a unique nonoverlapping particle/pronoun segmentation.
- **وبكتابهم** (pos): Intensive adjective كَتّاب is retained by productive morphological inference; exact lemma/POS attestation remains to be independently confirmed.
- **كتابان** (pos): Dual intensive adjective is retained by productive morphological inference; exact lemma/POS attestation remains to be independently confirmed.

The four low-confidence records are طمأنينة, في, هذا, أنا. The two intensive adjectival readings are medium-confidence provisional inferences. These cases have explicit working decisions for diagnostic use; they are not resolved human gold.

## Sources

The linked dictionary pages below were checked selectively by the adjudicator. They support specified lemmas/senses; regular imperatives, duals and other inflected readings are sometimes AI grammatical inferences from those lemmas, not directly quoted token attestations. The Academy كتب page supports ordinary book/writer readings but did not independently establish the provisional intensive adjective. Per-record notes and per-field rationales distinguish that gap. Original annotator citations are preserved and are not all claimed as separately rechecked.

- [Cairo Arabic Academy — قارئ](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D9%82%D8%B1%D8%A3)
- [Cairo Arabic Academy — سائل](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D8%B3%D8%A3%D9%84)
- [Cairo Arabic Academy — داع](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D8%AF%D8%B9%D9%88)
- [Cairo Arabic Academy — رام](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D8%B1%D9%8A%D9%85)
- [Cairo Arabic Academy — طمأنينة](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D8%B7%D9%85%D8%A3%D9%86)
- [Cairo Arabic Academy — بيوت](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D8%A8%D9%8A%D8%AA)
- [Cairo Arabic Academy — علم](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D8%B9%D9%84%D9%85)
- [Cairo Arabic Academy — قال](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D9%82%D9%88%D9%84)
- [Cairo Arabic Academy — دعا](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D8%AF%D8%B9%D8%B9)
- [Cairo Arabic Academy — أكل](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D9%83%D9%84%D9%84)
- [Cairo Arabic Academy — وبكتابهم, كتابان](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D9%83%D8%AA%D8%A8)
- [Cairo Arabic Academy — بالسيارتين](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D8%B3%D9%8A%D8%B1)
- [Lisan al-Arab, هذذ — هذا](https://www.islamweb.net/ar/library/content/122/8577/%D9%87%D8%B0%D8%B0)
- [Cairo Arabic Academy — نحن](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D8%AD%D9%86%D9%86)
- [Cairo Arabic Academy — أنا](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D8%A3%D9%86%D9%86)
- [Cairo Arabic Academy — أنا](https://www.arabicacademy.gov.eg/ar/%D9%85%D8%AD%D8%B1%D9%83-%D8%A7%D9%84%D8%A8%D8%AD%D8%AB/%D8%A3%D9%86)
- [Cairo Arabic Academy — من](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D9%85%D9%8A%D9%86)
- [Cairo Arabic Academy — من](https://www.arabicacademy.gov.eg/ar/search_engine/roots/%D9%85%D9%88%D9%86)

## Frozen SHA-256 provenance

Hashes are computed over exact file bytes. Changing an input or blind annotation invalidates this review provenance. Review/audit hashes below identify the delivered derived artifacts; the audit retains the original frozen hashes as well.

| File | SHA-256 |
| --- | --- |
| [words.json](words.json) | `b2da0fa9a51ee112ee3a9be9d3164f6855a7429eeb147f76fa95327851bd0539` |
| [annotator-a.json](annotator-a.json) | `35478ac3d0d33ef2af8b28cdf3b535359290d94bfaabc45a4116eee9bf519b17` |
| [annotator-b.json](annotator-b.json) | `b960f72493ed0479dcf32d6c0e1bd00780e2a59ba86d4f20459686e4a0cc1132` |
| [adjudicated.json](adjudicated.json) | `6798a2216e8dbb98a8f0d35b4b613a04c0f289f91a9bfd78caa535c693deafff` |
| [disagreements.json](disagreements.json) | `f48ca1b13720209a107a7eff2f2266ff991d02a6cba8bc5327dbf490817d644f` |

## Validation and limits

Validation passed using `node --import tsx` to import and call `validateAIReview`, without invoking `evaluateAI` or generating model predictions. Checks covered the frozen order, provenance fields, accepted root/POS/span schema, original UTF-16 coverage and mark attachment, expected rejections, frozen hashes, and completeness/fidelity of the disagreement audit.

The two blind AI annotators and AI adjudicator have correlated model knowledge and possible shared errors. Confidence is qualitative and uncalibrated. The curated isolated-word distribution, incomplete alternative sets, unresolved morphological granularity and absence of pattern/features limit interpretation. Independent human annotation and adjudication plus proper split/provenance controls are required for any future gold benchmark. This artifact remains diagnostic-only and cannot be used to claim human-gold accuracy or promotion eligibility.
