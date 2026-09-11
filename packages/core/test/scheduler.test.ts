import assert from "node:assert/strict";
import {test} from "node:test";
import {createAnalyze} from "../src/scheduler";
test("empty batch does not acquire a device; invalid input rejects atomically",async()=>{
 let calls=0; const analyze=createAnalyze(async()=>{calls++;throw new Error("No device");});
 assert.deepEqual(await analyze.batch([]),[]);
 await assert.rejects(analyze.batch(["كتب","foreign"]),/Arabic/);
 await assert.rejects(analyze("كتب",{topK:4}),/topK/);
 await assert.rejects(analyze.batch(Array(2049).fill("كتب")),/batch size/);
 assert.equal(calls,0);
});
test("scheduled calls serialize and recover after failed initialization",async()=>{
 let active=0,peak=0,calls=0;
 const analyze=createAnalyze(async()=>{
  active++;peak=Math.max(peak,active);calls++;
  await new Promise(r=>setTimeout(r,5));active--;throw new Error("No device");
 });
 const results=await Promise.allSettled([analyze("كتب"),analyze("دخل"),analyze.batch(["ذهب","خرج"])]);
 assert.equal(calls,3);assert.equal(peak,1);assert.ok(results.every(r=>r.status==="rejected"));
});
