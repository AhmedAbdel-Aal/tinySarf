# Reviewed failure replay

Save submissions outside `reviewed/` until an independent human has checked every accepted analysis. A submission is not gold by virtue of entering this bank. Do not use verification/final-test examples for replay. The command checks both normalized surfaces and all lemma-family identities against the frozen held-out identity manifest.

```json
{
  "schemaVersion": 1,
  "source": { "role": "mining", "name": "source name", "license": "source license" },
  "review": { "status": "pending", "independentHuman": false, "reviewer": null, "reviewedAt": null },
  "record": { "word": "كتب", "analyses": [] }
}
```

Fill `analyses` with the internal training schema (including `lemmaFamily`) and have the reviewer supply their review metadata. Use `pnpm fine-tune --add path/to/reviewed.json` to validate and retain it. `pnpm fine-tune --run --epochs 3` appends up to 10% reviewed replay words to the frozen original training data and creates a new candidate. It never activates weights. Run evaluation and the guarded promotion command separately.

No human-reviewed failure is currently available; the reviewed bank is intentionally empty.
