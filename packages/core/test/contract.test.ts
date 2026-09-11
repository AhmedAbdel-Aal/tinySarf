import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { normalize, MAX_LENGTH } from "../src/normalize";
import { validateAnalysis } from "../src/schema";
test("frozen 50 author-curated fixtures satisfy the span contract", () => {
 const bytes = readFileSync("packages/training/data/contract-fixtures.json");
 assert.equal(createHash("sha256").update(bytes).digest("hex"), readFileSync("packages/training/data/contract-fixtures.sha256", "utf8").trim());
 const fixtures = JSON.parse(bytes.toString()); assert.equal(fixtures.length, 50);
 for (const f of fixtures) { normalize(f.word); for (const a of f.analyses) validateAnalysis(f.word, a); }
});
test("normalization maps marks and tatweel back to original offsets", () => {
 const n = normalize("وَبِكِتَابِهِمْ"); assert.equal(n.text, "وبكتابهم");
 assert.equal(n.original.slice(n.offsets[2].start,n.offsets[5].end), "كِتَابِ");
 assert.equal(normalize("كـتاب").text, "كتاب");
 assert.equal(normalize("أإآىة").text, "أإآىة");
 assert.ok(n.packed[0] & (1 << 16));
});
test("reject malformed, foreign, overlength, and ambiguous whitespace inputs", () => {
 for (const input of ["", " ", "book", "كتاب جيد", "😀", "َكتب", "ﻻ", "ك".repeat(MAX_LENGTH + 1)]) assert.throws(() => normalize(input));
 assert.equal(normalize("ك".repeat(MAX_LENGTH)).text.length, MAX_LENGTH);
 assert.throws(() => normalize(null as unknown as string));
});
test("validator rejects gaps, overlaps, invalid scores and features", () => {
 const a = { spans: [{type: "stem" as const, start:0,end:3}], root:"كتب",pattern:null,pos:"verb",features:{},score:0 };
 validateAnalysis("كتب", a);
 assert.throws(() => validateAnalysis("كتب", {...a, score:NaN}));
 assert.throws(() => validateAnalysis("كتب", {...a, spans:[{type:"stem",start:1,end:3}]}));
 assert.throws(() => validateAnalysis("كتب", {...a, features:{number:"seven"}}));
 assert.throws(() => validateAnalysis("كتب", {...a, pos:"invented"}));
 assert.throws(() => validateAnalysis("كَتَبَ", {...a, spans:[{type:"stem",start:0,end:1},{type:"inflectional_suffix",start:1,end:6}]}));
});
