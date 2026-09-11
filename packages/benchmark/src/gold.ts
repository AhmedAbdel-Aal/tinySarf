import {readFile,appendFile,mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {hash,ROOT} from './artifact';
import {normalize} from '../../core/src/normalize';
import {validateAnalysis} from '../../core/src/schema';
import type {EvaluationWord} from './metrics';
export interface GoldDataset {schemaVersion:1;labelOrigin:'human-independent';role:'final-test';name:'gold-natural'|'gold-balanced';source:{name:string;version:string;license:string};annotation:{independentOfTeacher:true;reviewedBy:string[];completedAt:string};records:(EvaluationWord&{expectedError?:boolean;lemmaFamilies?:string[]})[]}
export function validateGold(d:GoldDataset):void {
 if(d.schemaVersion!==1 || d.labelOrigin!=='human-independent' || d.role!=='final-test' || !['gold-natural','gold-balanced'].includes(d.name) || d.annotation?.independentOfTeacher!==true || !d.annotation.reviewedBy?.length || !d.annotation.completedAt || !d.source?.license || !d.source.version || !Array.isArray(d.records) || !d.records.length) throw new Error('Independent gold provenance/review metadata is incomplete');
 const words=new Set<string>();
 for(const row of d.records) {
  if(words.has(row.word))throw new Error('Duplicate gold surface');words.add(row.word);
  if(row.expectedError) {let invalid=false;try{normalize(row.word);}catch{invalid=true;}if(!invalid)throw new Error('Expected-error gold record must violate the input contract');continue;}
  if(!row.lemmaFamilies?.length || row.lemmaFamilies.some(l=>!l))throw new Error('Gold lemma families are required to check leakage');
  normalize(row.word);if(!row.analyses?.length)throw new Error('Gold word needs every accepted analysis');row.analyses.forEach(a=>validateAnalysis(row.word,a));
 }
}
export async function loadGold(file:string,releaseCandidate:boolean):Promise<{dataset:GoldDataset;digest:string}> {
 if(!releaseCandidate) throw new Error('Gold/final-test access requires explicit --release-candidate; do not use it for architecture tuning');
 const bytes=await readFile(file),d=JSON.parse(bytes.toString()) as GoldDataset;
 validateGold(d);
 const directory=path.join(ROOT,'packages/training/data/generated/audit');await mkdir(directory,{recursive:true});
 const identities={schemaVersion:1,digest:hash(bytes),words:d.records.map(r=>r.word),lemmaFamilies:[...new Set(d.records.flatMap(r=>r.lemmaFamilies??[]))]};
 const identityFile=path.join(directory,`gold-identities-${hash(bytes)}.json`);try{await writeFile(identityFile,JSON.stringify(identities)+'\n',{flag:'wx'});}catch(error){if((error as NodeJS.ErrnoException).code!=='EEXIST')throw error;}
 await appendFile(path.join(directory,'final-test-access.jsonl'),JSON.stringify({createdAt:new Date().toISOString(),file:path.resolve(file),digest:hash(bytes),purpose:'explicit release-candidate evaluation'})+'\n');
 return {dataset:d,digest:hash(bytes)};
}
