import assert from "node:assert/strict";
import {test} from "node:test";
import {rulesBaseline} from "../src/baselines";
import {validateAnalysis} from "../../core/src/schema";
test("rules baseline is explicit, simple and span-valid",()=>{
 const rows=["وبالكتابهم","كتب","البيت"].map(word=>({word,analyses:[]}));
 const predictions=rulesBaseline(rows); rows.forEach((row,i)=>validateAnalysis(row.word,predictions[i][0]));
 assert.equal(predictions[1][0].root,null);
});
