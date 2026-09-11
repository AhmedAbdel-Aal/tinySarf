import test from 'node:test';
import assert from 'node:assert/strict';
import {warmupStable} from '../src/math';
test('warmup stability rejects drift, jitter and too few samples',()=>{
 assert.equal(warmupStable(Array(19).fill(1)),false);
 assert.equal(warmupStable(Array(20).fill(1)),true);
 assert.equal(warmupStable([...Array(10).fill(1),...Array(10).fill(1.2)]),false);
 assert.equal(warmupStable(Array.from({length:20},(_,i)=>i%2?2:1)),false);
});
