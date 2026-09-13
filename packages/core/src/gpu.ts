/// <reference types="@webgpu/types" />
import type {LoadedModel} from "./model";
import {sizeClass,type NormalizedWord} from "./normalize";
import type {Backend,InferenceResult,Logits} from "./cpu";
import {shaders} from "./shaders";
const CHUNK=256;
export class WebGPUError extends Error {constructor(message:string){super(message);this.name="TinySarfWebGPUError";}}
export async function acquireDevice():Promise<{device:GPUDevice;adapter:Record<string,unknown>;acquisitionMs:number}> {
  const start=performance.now();
  if(typeof navigator==="undefined" || !navigator.gpu || !globalThis.isSecureContext) throw new WebGPUError("WebGPU requires a supported browser in a secure context. Choose { backend: 'cpu' } explicitly to use the local reference.");
  const adapter=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});
  if(!adapter) throw new WebGPUError("No WebGPU adapter is available. Select the CPU reference explicitly or try a supported browser.");
  const device=await adapter.requestDevice();const info=adapter.info;
  return {device,acquisitionMs:performance.now()-start,adapter:{vendor:info.vendor||null,architecture:info.architecture||null,device:info.device||null,description:info.description||null,isFallbackAdapter:info.isFallbackAdapter??null}};
}
interface BufferEntry {buffer:GPUBuffer;size:number}
export class GPUBackend implements Backend {
  readonly buffers=new Map<string,BufferEntry>();
  readonly pipelines=new Map<string,GPUComputePipeline>();
  readonly groups=new Map<string,GPUBindGroup>();
  readonly code:Record<string,string>;
  readonly layout:GPUBindGroupLayout;
  lostReason:string|null=null;
  peakBytes=0;
  initialization:Record<string,number>={};
  lastTimings:Record<string,number>={};
  /** Internal hook; capture and readback implementation lives only in the diagnostic entry. */
  onStage?: (encoder:GPUCommandEncoder,name:string,buffer:GPUBuffer,elements:number,batch:number,length:number)=>void;
  private constructor(readonly model:LoadedModel,readonly device:GPUDevice,readonly adapter:Record<string,unknown>) {
    this.code=shaders(model.manifest);
    this.layout=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
    ]});
    void device.lost.then(info=>{this.lostReason=info.message||info.reason;});
  }
  static async create(model:LoadedModel,acquired?:Awaited<ReturnType<typeof acquireDevice>>):Promise<GPUBackend> {
    const acquisition=acquired??await acquireDevice();const backend=new GPUBackend(model,acquisition.device,acquisition.adapter);
    backend.initialization.deviceAcquisitionMs=acquisition.acquisitionMs;
    const start=performance.now(),device=backend.device;
    device.pushErrorScope("validation");
    try {
      const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[backend.layout]});
      await Promise.all(Object.entries(backend.code).map(async([name,code])=>{
        const module=device.createShaderModule({code,label:`TinySarf ${name}`});
        const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
        if(messages.length) throw new WebGPUError(`${name} shader compilation failed: ${messages.map(m=>m.message).join('; ')}`);
        backend.pipelines.set(name,await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:"main"},label:`TinySarf ${name}`}));
      }));
      backend.initialization.shaderCompilationMs=performance.now()-start;
      const uploadStart=performance.now();
      backend.ensure("weights",model.packed.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(backend.buffer("weights"),0,model.packed.slice().buffer);
      backend.ensure("uniform",16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
      backend.ensure("dummy",4,GPUBufferUsage.STORAGE);
      backend.initialization.weightUploadMs=performance.now()-uploadStart;
      const error=await device.popErrorScope();if(error) throw new WebGPUError(error.message);
      return backend;
    } catch(error) {backend.dispose();throw error;}
  }
  buffer(name:string):GPUBuffer {return this.buffers.get(name)!.buffer;}
  private ensure(name:string,size:number,usage:GPUBufferUsageFlags):void {
    size=Math.max(4,Math.ceil(size/4)*4);const entry=this.buffers.get(name);if(entry && entry.size>=size) return;
    const total=[...this.buffers.values()].reduce((s,b)=>s+b.size,0)+size;
    this.peakBytes=Math.max(this.peakBytes,total);
    const buffer=this.device.createBuffer({size,usage,label:`TinySarf ${name}`});
    entry?.buffer.destroy();this.buffers.set(name,{buffer,size});this.groups.clear();
  }
  private group(name:string,source:string,destination:string,pool="dummy"):GPUBindGroup {
    let group=this.groups.get(name);if(group) return group;
    group=this.device.createBindGroup({layout:this.layout,entries:[
      {binding:0,resource:{buffer:this.buffer("input")}},{binding:1,resource:{buffer:this.buffer("weights")}},
      {binding:2,resource:{buffer:this.buffer(source)}},{binding:3,resource:{buffer:this.buffer(destination)}},
      {binding:4,resource:{buffer:this.buffer("uniform")}},{binding:5,resource:{buffer:this.buffer(pool)}},
    ]});this.groups.set(name,group);return group;
  }
  async run(words:NormalizedWord[]):Promise<InferenceResult> {
    const results:Logits[]=[];const totalStart=performance.now();this.lastTimings={packingMs:0,inputUploadMs:0,encodingMs:0,readbackAndExecutionMs:0};
    for(let start=0;start<words.length;start+=CHUNK) results.push(...await this.chunk(words.slice(start,start+CHUNK)));
    this.lastTimings.backendTotalMs=performance.now()-totalStart;return {words:results};
  }
  private async chunk(words:NormalizedWord[]):Promise<Logits[]> {
    if(this.lostReason) throw new WebGPUError(`WebGPU device was lost: ${this.lostReason}. Retry to acquire a new device.`);
    const m=this.model.manifest,device=this.device,batch=words.length,capacity=2**Math.ceil(Math.log2(batch));
    const length=sizeClass(Math.max(...words.map(w=>w.text.length))),heads=Object.entries(m.labels.heads);
    const globalSize=heads.reduce((s,[,labels])=>s+labels.length,0),stride=length*8+globalSize;
    const start=performance.now(),packed=new Uint32Array(capacity*length);
    words.forEach((word,i)=>packed.set(word.packed,i*length));
    this.lastTimings.packingMs+=performance.now()-start;
    const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC;
    this.ensure("input",packed.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
    const width=Math.max(m.width,m.rootArchitecture?.width??0);
    this.ensure("a",capacity*length*Math.max(width,m.embedding)*4,storage);
    this.ensure("b",capacity*length*width*4,storage);
    this.ensure("pool",capacity*width*4,storage);
    this.ensure("output",capacity*stride*4,storage);
    this.ensure("readback",capacity*stride*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const upload=performance.now();device.queue.writeBuffer(this.buffer("input"),0,packed);
    device.queue.writeBuffer(this.buffer("uniform"),0,new Uint32Array([batch,length,0,0]));
    this.lastTimings.inputUploadMs+=performance.now()-upload;
    const encoding=performance.now(),encoder=device.createCommandEncoder({label:"TinySarf inference"});
    const dispatch=(name:string,source:string,destination:string,elements:number,pool?:string)=>{
      const pass=encoder.beginComputePass({label:name});pass.setPipeline(this.pipelines.get(name)!);pass.setBindGroup(0,this.group(name,source,destination,pool));pass.dispatchWorkgroups(Math.ceil(elements/64));pass.end();
      this.onStage?.(encoder,name,this.buffer(destination),elements,batch,length);
    };
    dispatch("embedding","dummy","a",batch*length*m.embedding);
    dispatch("conv0","a","b",batch*length*m.width);
    dispatch("conv1","b","a",batch*length*m.width);
    dispatch("conv2","a","b",batch*length*m.width);
    dispatch("pool","b","pool",batch*m.width);
    dispatch("heads","b","output",batch*stride,"pool");
    if(m.rootArchitecture) {
      const root=m.rootArchitecture;
      dispatch("root.embedding","dummy","a",batch*length*root.embedding);
      dispatch("root.conv0","a","b",batch*length*root.width);
      dispatch("root.conv1","b","a",batch*length*root.width);
      dispatch("root.conv2","a","b",batch*length*root.width);
      dispatch("root.pool","b","pool",batch*root.width);
      dispatch("root.heads","b","output",batch*4,"pool");
    }
    encoder.copyBufferToBuffer(this.buffer("output"),0,this.buffer("readback"),0,batch*stride*4);
    device.queue.submit([encoder.finish()]);this.lastTimings.encodingMs+=performance.now()-encoding;
    const wait=performance.now(),readback=this.buffer("readback");
    try {
      await readback.mapAsync(GPUMapMode.READ,0,batch*stride*4);
      const data=new Float32Array(readback.getMappedRange(0,batch*stride*4));const results:Logits[]=[];
      for(let i=0;i<batch;i++) {
        const segmentation=data.slice(i*stride,i*stride+words[i].text.length*8),headValues:Logits["heads"]={};
        let offset=i*stride+length*8;
        for(const [name,labels] of heads) {headValues[name]=data.slice(offset,offset+labels.length);offset+=labels.length;}
        results.push({segmentation,heads:headValues});
      }
      readback.unmap();this.lastTimings.readbackAndExecutionMs+=performance.now()-wait;return results;
    } catch(error) {
      if(readback.mapState==="mapped") readback.unmap();
      throw new WebGPUError(`WebGPU execution/readback failed${this.lostReason?`: ${this.lostReason}`:''}: ${String(error)}`);
    }
  }
  metadata():Record<string,unknown> {return {backend:"webgpu",adapter:this.adapter,pipelineCount:this.pipelines.size,internalBatchLimit:CHUNK,residentGPUBytes:[...this.buffers.values()].reduce((s,b)=>s+b.size,0),peakAllocatedGPUBytes:this.peakBytes,initialization:this.initialization,lastTimings:this.lastTimings};}
  dispose():void {for(const {buffer} of this.buffers.values()) buffer.destroy();this.buffers.clear();this.groups.clear();this.device.destroy();}
}
