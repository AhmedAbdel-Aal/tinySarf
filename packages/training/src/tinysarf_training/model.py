"""Small non-autoregressive CNN with mask-correct local context and fixed heads."""
import collections
import numpy as np
import torch
from torch import nn
from .contract import LETTERS, SPECIAL, SPAN_TYPES, FEATURE_MAP

def make_labels(rows):
    counts=collections.Counter(a['pattern'] for r in rows for a in r['analyses'] if a['pattern'])
    patterns=sorted(counts,key=lambda p:(-counts[p],p))[:64]
    heads={'pos':SPECIAL+['noun','proper_noun','numeral','adjective','verb','adverb','preposition','conjunction','interjection','pronoun','particle']}
    for i in range(4): heads[f'root{i}']=['__blank__']+list(LETTERS)
    heads['pattern']=SPECIAL+patterns
    for _,(target,values) in FEATURE_MAP.items(): heads[target]=SPECIAL+list(dict.fromkeys(values.values()))
    return {'segmentation':SPAN_TYPES,'heads':heads}

class Student(nn.Module):
    def __init__(self,width,labels,embedding=32,root_width=None):
        super().__init__(); self.width=width; self.labels=labels; self.embedding_width=embedding
        self.embedding=nn.Embedding(len(LETTERS)+1,embedding,padding_idx=0)
        self.position=nn.Embedding(32,embedding)
        self.convs=nn.ModuleList([nn.Conv1d(embedding,width,3,padding=1),nn.Conv1d(width,width,3,padding=2,dilation=2),nn.Conv1d(width,width,3,padding=4,dilation=4)])
        self.segmentation=nn.Linear(width,len(SPAN_TYPES))
        self.heads=nn.ModuleDict({k:nn.Linear(width,len(v)) for k,v in labels['heads'].items() if root_width is None or not k.startswith('root')})
        self.root_model=None
        self.root_decoder=None
        if root_width is not None:
            from .root_model import RootModel
            self.root_model=RootModel(root_width)
    def forward(self,ids,trace=False):
        mask=(ids!=0).float().unsqueeze(-1)
        embedded=(self.embedding(ids)+self.position(torch.arange(ids.shape[1],device=ids.device)))*mask
        x=embedded.transpose(1,2); states={}
        for i,conv in enumerate(self.convs):
            x=torch.relu(conv(x))*mask.transpose(1,2); states[f'conv{i}']=x.transpose(1,2)
        h=x.transpose(1,2); pooled=h.sum(1)/mask.sum(1).clamp(min=1)
        outputs={'segmentation':self.segmentation(h),**{k:head(pooled) for k,head in self.heads.items()}}
        if self.root_model is not None:
            roots,root_states=self.root_model(ids,trace=True)
            outputs.update({f'root{i}':roots[:,i] for i in range(4)})
            states.update(root_states)
        if trace: return {'embedding':embedded,**states,'pooled':pooled,**outputs}
        return outputs

def parameter_count(model): return sum(p.numel() for p in model.parameters())
def choose_width(target,labels):
    # Exact parameter counts are generated, never inferred from the candidate's nominal name.
    return min(range(16,225,4),key=lambda w:abs(parameter_count(Student(w,labels))-target))

def encode_words(words,length=32):
    ids=np.zeros((len(words),length),dtype=np.int64)
    for i,word in enumerate(words):
        ids[i,:len(word)]=[LETTERS.index(c)+1 for c in word]
    return torch.from_numpy(ids)

def encode_analysis(row,analysis,labels,length=32):
    seg=np.full(length,-100,dtype=np.int64)
    for s in analysis['spans']: seg[s['start']:s['end']]=SPAN_TYPES.index(s['type'])
    target={'segmentation':seg}
    for name,classes in labels['heads'].items():
        if name.startswith('root'):
            root=analysis['root'] or ''; slot=int(name[-1]); value=root[slot] if slot<len(root) else '__blank__'
        elif name in ('pos','pattern'): value=analysis[name] or '__unknown__'
        else: value=analysis['features'].get(name,'__missing__')
        target[name]=classes.index(value) if value in classes else classes.index('__unknown__')
    return target

def export_float(model):
    arrays={'embedding':model.embedding.weight.detach().cpu().numpy(), 'position':model.position.weight.detach().cpu().numpy()}
    for i,c in enumerate(model.convs):
        arrays[f'conv{i}.weight']=c.weight.detach().cpu().numpy().transpose(0,2,1).copy()
        arrays[f'conv{i}.bias']=c.bias.detach().cpu().numpy()
    for name,head in [('segmentation',model.segmentation),*model.heads.items()]:
        arrays[f'{name}.weight']=head.weight.detach().cpu().numpy()
        arrays[f'{name}.bias']=head.bias.detach().cpu().numpy()
    if model.root_model is not None:
        from .root_model import export_root
        arrays.update(export_root(model.root_model))
    return arrays

def numpy_reference(ids,arrays,trace=False):
    """Independent float CPU semantics, no PyTorch operations."""
    mask=(ids!=0)[...,None].astype(np.float32)
    x=(arrays['embedding'][ids]+arrays['position'][np.arange(ids.shape[1])])*mask
    tensors={'embedding':x.copy()}
    for layer,dilation in enumerate([1,2,4]):
        weight=arrays[f'conv{layer}.weight']; y=np.zeros((*ids.shape,weight.shape[0]),dtype=np.float32)
        for k in range(3):
            delta=(k-1)*dilation
            dst=slice(max(0,-delta),min(ids.shape[1],ids.shape[1]-delta))
            src=slice(max(0,delta),min(ids.shape[1],ids.shape[1]+delta))
            y[:,dst] += x[:,src] @ weight[:,k,:].T
        x=np.maximum(y+arrays[f'conv{layer}.bias'],0)*mask
        tensors[f'conv{layer}']=x.copy()
    pooled=x.sum(1)/np.maximum(mask.sum(1),1); tensors['pooled']=pooled
    root_model='root.embedding' in arrays
    for name in [n[:-7] for n in arrays if n.endswith('.weight') and not n.startswith(('conv','root.')) and not (root_model and n.startswith('root'))]:
        h=x if name=='segmentation' else pooled
        tensors[name]=h@arrays[f'{name}.weight'].T+arrays[f'{name}.bias']
    if root_model:
        from .root_model import numpy_root
        tensors.update(numpy_root(ids,arrays))
    return tensors if trace else {k:v for k,v in tensors.items() if k not in ('embedding','conv0','conv1','conv2','pooled') and not k.startswith('root.')}
