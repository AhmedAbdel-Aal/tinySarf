import {createHash} from "node:crypto";
import {execFileSync} from "node:child_process";
import {readFile,writeFile,mkdir} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
export const ROOT=path.resolve(import.meta.dirname,"../../..");
export const hash=(bytes:string|Uint8Array)=>createHash("sha256").update(bytes).digest("hex");
export const json=async(file:string)=>JSON.parse(await readFile(file,"utf8"));
export function provenance(command=process.argv.join(" ")) {
  function git(args:string[]) {try{return execFileSync("git",args,{cwd:ROOT,encoding:"utf8",stdio:["ignore","pipe","ignore"]}).trim();}catch{return null;}}
  const dirty=git(["status","--porcelain"]);
  return {schemaVersion:1,runId:new Date().toISOString().replace(/[:.]/g,""),git:{commit:git(["rev-parse","HEAD"]),dirty:dirty===null?null:!!dirty},createdAt:new Date().toISOString(),commands:[command],environment:{os:`${os.platform()} ${os.release()}`,architecture:os.arch(),cpu:os.cpus()[0]?.model??null,ramBytes:os.totalmem(),node:process.version,browser:null,webgpuAdapter:null}};
}
export async function writeArtifact(kind:string,value:Record<string,unknown>,directory="packages/benchmark/results") {
  validateArtifact(value);
  const dir=path.join(ROOT,directory);await mkdir(dir,{recursive:true});
  const filename=path.join(dir,`${kind}-${value.runId}.json`);
  await writeFile(filename,JSON.stringify(value,null,2)+"\n",{flag:"wx"});
  return filename;
}
export function validateArtifact(value:Record<string,unknown>):void {
  if(value.schemaVersion!==1 || typeof value.runId!=="string" || typeof value.createdAt!=="string" || !Array.isArray(value.commands) || !value.git || !value.environment || !value.model || !value.data) throw new Error("Missing result provenance");
  const visit=(x:unknown):void=>{
    if(typeof x==="number" && !Number.isFinite(x)) throw new Error("Nonfinite result number");
    if(x && typeof x==="object") Object.values(x).forEach(visit);
  };visit(value);
}
