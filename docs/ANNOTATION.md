# Independent annotation and release evaluation

The 50 author-curated fixtures are implementation examples, not independently reviewed gold. Their `reviewStatus` must remain `pending-human-review` until a qualified person checks them. Teacher analyses are not ground truth.

To create gold evaluation, obtain a licensed natural MSA word sample independently of the teacher-sampled verification set. Have qualified Arabic annotators work independently of model/teacher predictions, include every acceptable analysis, record ambiguity and notes, adjudicate disagreements, and retain reviewer identities or stable IDs and review dates. Build a separate balanced challenge set with weak, hamzated, doubled and quadriliteral roots, broken plurals, stacked clitics, diacritics, names, foreign forms and non-MSA inputs. Do not present the teacher-sampled distribution as natural text.

Start from `packages/training/data/gold-template.json`. Empty metadata, zero records or an unreviewed declaration intentionally fail validation. Public span offsets must follow `docs/CONTRACT.md`. Invalid strings may use `expectedError: true` and `analyses: []`; their appropriate metric is input-rejection accuracy, not morphological exact match.

Gold files are final-test data. The loader requires an explicit release-candidate flag and logs access. Do not open these files during architecture tuning. Use train for optimization, verification for repeated selection, and mining for reviewed failure discovery. The teacher final-test split has not been opened by this implementation run.

Human review is an external requirement. The software validates the recorded provenance but cannot certify an annotator's independence or linguistic expertise.

Commands: `pnpm evaluate --gold path/to/gold-natural.json --gold path/to/gold-balanced.json --release-candidate` evaluates independent gold; `pnpm evaluate --dataset-role final-test --release-candidate` evaluates the sealed teacher test in a separate artifact. Neither command has been run on final-test data for this experiment.
