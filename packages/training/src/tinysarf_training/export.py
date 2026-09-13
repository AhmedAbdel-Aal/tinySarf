import argparse, hashlib, json
import numpy as np
import torch
from .artifacts import read_json, write_json, sha
from .model import Student, export_float, numpy_reference, encode_words
from .contract import LETTERS

def export_model(model,path,model_id,data_digest):
    path.mkdir(parents=True,exist_ok=True); arrays=export_float(model)
    np.savez(path/'float.npz',**arrays)
    packed=bytearray(); tensors={}; quant_arrays={}
    for name,array in arrays.items():
        if not np.all(np.isfinite(array)): raise ValueError(f'Non-finite tensor: {name}')
        scale=float(np.max(np.abs(array))/127) if np.max(np.abs(array))>0 else 1.0
        scale=float(np.float32(scale))
        values=np.clip(np.rint(array/scale),-127,127).astype(np.int8)
        offset=len(packed); packed.extend(values.tobytes()); length=array.size
        packed.extend(b'\0'*((-len(packed))%4))
        tensors[name]={'shape':list(array.shape),'offset':offset,'length':length,'scale':scale}
        quant_arrays[name]=values.astype(np.float32)*np.float32(scale)
    (path/'weights.bin').write_bytes(packed)
    float_packed=np.zeros(len(packed),dtype='<f4')
    for name,array in arrays.items():
        spec=tensors[name];float_packed[spec['offset']:spec['offset']+spec['length']]=array.reshape(-1)
    (path/'float.weights.bin').write_bytes(float_packed.tobytes())
    parameters=sum(a.size for a in arrays.values())
    root=model.root_model
    manifest={'schemaVersion':1,'format':'cnn-root-v2' if root is not None else 'cnn-v1','id':model_id,'status':'experimental-unpromoted',
      'normalizationVersion':'arabic-v1','quantization':'symmetric-per-tensor-int8','labels':model.labels,
      'vocabulary':['__pad__']+list(LETTERS),'maxLength':32,'embedding':32,'width':model.width,
      'dilations':[1,2,4],'trainingParameters':parameters,'reachableWeights':parameters-model.embedding_width-(root.embedding_width if root is not None else 0),
      'packedWeightsBytes':len(packed),'sha256':sha(path/'weights.bin'),'floatSha256':sha(path/'float.npz'),
      'floatBinarySha256':sha(path/'float.weights.bin'),'verificationDigest':data_digest,'tensors':tensors}
    if root is not None: manifest['rootArchitecture']={'format':'copy-cnn-v1','embedding':root.embedding_width,'width':root.width,'dilations':[1,2,4]}
    if model.root_decoder is not None: manifest['rootDecoder']=model.root_decoder
    write_json(path/'manifest.json',manifest)
    return manifest,arrays,quant_arrays

def parity(model,arrays,quant_arrays,words):
    ids=encode_words(words); model.eval()
    with torch.no_grad(): pt=model(ids,trace=True)
    ref=numpy_reference(ids.numpy(),arrays,trace=True); quant=numpy_reference(ids.numpy(),quant_arrays,trace=True)
    reports={}
    for name,tensor in pt.items():
        expected=tensor.numpy(); actual=ref[name]; q=quant[name]
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(q)): raise ValueError('Non-finite parity tensor')
        diff=np.abs(expected-actual)
        reports[name]={'floatMaxAbsoluteError':float(diff.max()),'floatMaxRelativeError':float((diff/np.maximum(np.abs(expected),1e-6)).max()),
          'floatArgmaxAgreement':float((expected.argmax(-1)==actual.argmax(-1)).mean()),
          'quantMaxAbsoluteError':float(np.abs(actual-q).max()),'quantArgmaxAgreement':float((actual.argmax(-1)==q.argmax(-1)).mean())}
    return {'tensors':reports,'floatPassed':all(t['floatMaxAbsoluteError']<=1e-4 for t in reports.values()),'webgpu':None}
