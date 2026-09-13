"""Parallel character copying with learned restoration for absent radicals."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .contract import LETTERS


class RootModel(nn.Module):
    def __init__(self, width=48, embedding=16):
        super().__init__()
        self.width, self.embedding_width = width, embedding
        self.root_embedding = nn.Embedding(len(LETTERS) + 1, embedding, padding_idx=0)
        self.root_position = nn.Embedding(32, embedding)
        self.root_convs = nn.ModuleList([nn.Conv1d(embedding, width, 3, padding=1),
            nn.Conv1d(width, width, 3, padding=2, dilation=2),
            nn.Conv1d(width, width, 3, padding=4, dilation=4)])
        self.pointer = nn.Linear(width, 4, bias=False)
        self.gate = nn.Linear(width, 4)
        self.generators = nn.ModuleList([nn.Linear(width, len(LETTERS) + 1) for _ in range(4)])
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad(): self.gate.bias.copy_(torch.tensor([1., 1., 1., -2.]))

    def forward(self, ids, trace=False):
        mask = (ids != 0).unsqueeze(1)
        embedded = (self.root_embedding(ids) + self.root_position(torch.arange(ids.shape[1], device=ids.device))) * mask.transpose(1, 2)
        x, states = embedded.transpose(1, 2), {'root.embedding': embedded}
        for layer, conv in enumerate(self.root_convs):
            x = F.relu(conv(x)) * mask
            states[f'root.conv{layer}'] = x.transpose(1, 2)
        attention = self.pointer(x.transpose(1, 2)).transpose(1, 2).masked_fill(~mask, -1e9).softmax(-1)
        copied = torch.zeros((len(ids), 4, len(LETTERS) + 1), device=ids.device)
        copied.scatter_add_(2, ids[:, None, :].expand(-1, 4, -1), attention)
        pooled = x.sum(-1) / mask.sum(-1).clamp_min(1)
        states['root.pooled'] = pooled
        generated = torch.stack([head(pooled).softmax(-1) for head in self.generators], 1)
        gate = self.gate(pooled).sigmoid().unsqueeze(-1)
        logits = (gate * copied + (1 - gate) * generated).clamp_min(1e-12).log()
        return (logits, states) if trace else logits


def export_root(model):
    arrays = {'root.embedding': model.root_embedding.weight.detach().cpu().numpy(),
              'root.position': model.root_position.weight.detach().cpu().numpy()}
    for layer, conv in enumerate(model.root_convs):
        arrays[f'root.conv{layer}.weight'] = conv.weight.detach().cpu().numpy().transpose(0, 2, 1).copy()
        arrays[f'root.conv{layer}.bias'] = conv.bias.detach().cpu().numpy()
    arrays['root.pointer.weight'] = model.pointer.weight.detach().cpu().numpy()
    arrays['root.gate.weight'] = model.gate.weight.detach().cpu().numpy()
    arrays['root.gate.bias'] = model.gate.bias.detach().cpu().numpy()
    for i, head in enumerate(model.generators):
        arrays[f'root{i}.weight'] = head.weight.detach().cpu().numpy()
        arrays[f'root{i}.bias'] = head.bias.detach().cpu().numpy()
    return arrays


def numpy_root(ids, arrays):
    """Independent float32 reference for every root operation, including copying."""
    mask = (ids != 0)[..., None].astype(np.float32)
    x = (arrays['root.embedding'][ids] + arrays['root.position'][np.arange(ids.shape[1])]) * mask
    states = {'root.embedding': x.copy()}
    for layer, dilation in enumerate([1, 2, 4]):
        weight = arrays[f'root.conv{layer}.weight']
        y = np.zeros((*ids.shape, weight.shape[0]), dtype=np.float32)
        for k in range(3):
            delta = (k - 1) * dilation
            dst = slice(max(0, -delta), min(ids.shape[1], ids.shape[1] - delta))
            src = slice(max(0, delta), min(ids.shape[1], ids.shape[1] + delta))
            y[:, dst] += x[:, src] @ weight[:, k, :].T
        x = np.maximum(y + arrays[f'root.conv{layer}.bias'], 0) * mask
        states[f'root.conv{layer}'] = x.copy()
    pooled = x.sum(1) / np.maximum(mask.sum(1), 1)
    states['root.pooled'] = pooled
    def softmax(values):
        exp = np.exp(values - values.max(-1, keepdims=True))
        return exp / exp.sum(-1, keepdims=True)
    attention = softmax(np.where(ids[:, None, :] != 0,
        (x @ arrays['root.pointer.weight'].T).transpose(0, 2, 1), np.float32(-1e9)))
    generated = softmax(np.stack([pooled @ arrays[f'root{i}.weight'].T + arrays[f'root{i}.bias'] for i in range(4)], 1))
    copied = np.zeros_like(generated)
    for batch in range(len(ids)):
        for slot in range(4): np.add.at(copied[batch, slot], ids[batch], attention[batch, slot])
    gate_logit = pooled @ arrays['root.gate.weight'].T + arrays['root.gate.bias']
    gate = (1 / (1 + np.exp(-np.clip(gate_logit, -80, 80))))[..., None]
    logits = np.log(np.maximum(gate * copied + (1 - gate) * generated, np.float32(1e-12)))
    states.update({f'root{i}': logits[:, i] for i in range(4)})
    return states
