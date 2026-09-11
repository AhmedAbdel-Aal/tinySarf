import assert from 'node:assert/strict';
import {test} from 'node:test';
import {loadGold,validateGold,type GoldDataset} from '../src/gold';
test('gold cannot be opened without explicit release-candidate intent',async()=>{
 await assert.rejects(loadGold('/does-not-exist',false),/release-candidate/);
});
test('unreviewed or teacher-generated data cannot masquerade as independent gold',()=>{
 assert.throws(()=>validateGold({schemaVersion:1,labelOrigin:'teacher-generated'} as unknown as GoldDataset),/provenance/);
 assert.throws(()=>validateGold({schemaVersion:1,labelOrigin:'human-independent',role:'final-test',name:'gold-natural',annotation:{independentOfTeacher:false}} as unknown as GoldDataset),/provenance/);
});

import {correctness} from '../src/correctness';
test('teacher final-test access is rejected before loading data without release intent',async()=>{await assert.rejects(()=>correctness(undefined,true,[],false,'final-test'),/release-candidate/);});
