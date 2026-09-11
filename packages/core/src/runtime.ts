import {loadModel,type LoadedModel} from "./model";
import {CPUBackend,type Backend} from "./cpu";
import {GPUBackend,acquireDevice} from "./gpu";
let modelPromise:Promise<LoadedModel>|undefined;
const runtimes:Partial<Record<"cpu"|"webgpu",Promise<{model:LoadedModel;backend:Backend}>>>={};
export function builtinModel():Promise<LoadedModel> {
  if(!modelPromise) modelPromise=(async()=>{
    const {manifest,weightsBase64}=await import("./checkpoint.js");
    const bytes=Uint8Array.from(atob(weightsBase64),c=>c.charCodeAt(0));return loadModel(manifest,bytes);
  })().catch(error=>{modelPromise=undefined;throw error;});
  return modelPromise;
}
export async function getRuntime(kind:"cpu"|"webgpu"):Promise<{model:LoadedModel;backend:Backend}> {
  return runtime(kind);
}
async function runtime(kind:"cpu"|"webgpu"):Promise<{model:LoadedModel;backend:Backend}> {
  const existing=runtimes[kind];
  if(existing) {
    const value=await existing;
    if(!(value.backend instanceof GPUBackend) || !value.backend.lostReason) return value;
    value.backend.dispose();runtimes[kind]=undefined;
  }
  if(!runtimes[kind]) runtimes[kind]=(async()=>{
    if(kind==="cpu") {const model=await builtinModel();return {model,backend:new CPUBackend(model)};}
    const [model,acquired]=await Promise.all([builtinModel(),acquireDevice()]);return {model,backend:await GPUBackend.create(model,acquired)};
  })().catch(error=>{runtimes[kind]=undefined;throw error;});
  return runtimes[kind]!;
}
