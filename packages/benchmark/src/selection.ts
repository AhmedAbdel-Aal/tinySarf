import {access,readFile} from 'node:fs/promises';
import path from 'node:path';
import {ROOT,json,hash} from './artifact';
/** Promotion uses an explicit environment override; normal builds prefer the active pointer. */
export async function selectedModel(override=process.env.TINYSARF_MODEL_DIR) {
 if(override) {const directory=path.resolve(ROOT,override);return {directory,manifestSha256:hash(await readFile(path.join(directory,'manifest.json')))};}
 const active=path.join(ROOT,'packages/training/active/current.json');
 let file=active;try {await access(file);}catch {file=path.join(ROOT,'packages/training/candidate.json');}
 const selected=await json(file);const directory=path.resolve(ROOT,selected.directory);
 if(hash(await readFile(path.join(directory,'manifest.json')))!==selected.manifestSha256)throw new Error('Selected manifest hash mismatch');
 return {...selected,directory};
}
