"""Train a root pointer/generator on frozen CNN features; no sealed labels are read."""
import argparse
import copy
import json
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .artifacts import ROOT, DATA, RUNS, read_json, write_json, sha, provenance
from .contract import LETTERS
from .corrected import load_corrected_dataset
from .model import Student, encode_words
from .root_model import RootModel

BASE_DIRECTORY = 'packages/training/candidates/20260911T191243.077466Z-250000'


class RootPointer(nn.Module):
    def __init__(self, base):
        super().__init__()
        width = base.width
        self.pointer = nn.Linear(width, 4, bias=False)
        self.gate = nn.Linear(width, 4)
        self.generators = nn.ModuleList([copy.deepcopy(base.heads[f'root{i}']) for i in range(4)])
        nn.init.zeros_(self.pointer.weight)
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad(): self.gate.bias.copy_(torch.tensor([1., 1., 1., -2.]))

    def forward(self, h, pooled, ids):
        attention = self.pointer(h).transpose(1, 2).masked_fill(ids[:, None, :] == 0, -1e9).softmax(-1)
        copied = torch.zeros((len(ids), 4, len(LETTERS) + 1), device=h.device)
        copied.scatter_add_(2, ids[:, None, :].expand(-1, 4, -1), attention)
        generated = torch.stack([head(pooled).softmax(-1) for head in self.generators], 1)
        gate = self.gate(pooled).sigmoid().unsqueeze(-1)
        return (gate * copied + (1 - gate) * generated).clamp_min(1e-12).log()


class CNNRootPointer(RootPointer):
    """A small task-specific feature extractor keeps the existing segmentation untouched."""
    def __init__(self, base):
        super().__init__(base)
        self.root_embedding = nn.Embedding(len(LETTERS) + 1, 16, padding_idx=0)
        self.root_position = nn.Embedding(32, 16)
        self.root_convs = nn.ModuleList([nn.Conv1d(16, 24, 3, padding=1),
            nn.Conv1d(24, 24, 3, padding=2, dilation=2),
            nn.Conv1d(24, 24, 3, padding=4, dilation=4)])
        self.pointer = nn.Linear(24, 4, bias=False)
        self.root_residual = nn.Linear(24, 4 * (len(LETTERS) + 1))
        nn.init.zeros_(self.root_residual.weight)
        nn.init.zeros_(self.root_residual.bias)

    def forward(self, h, pooled, ids):
        mask = (ids != 0).unsqueeze(1)
        x = (self.root_embedding(ids) + self.root_position(torch.arange(ids.shape[1]))).transpose(1, 2) * mask
        for conv in self.root_convs: x = F.relu(conv(x)) * mask
        features = x.transpose(1, 2)
        attention = self.pointer(features).transpose(1, 2).masked_fill(~mask, -1e9).softmax(-1)
        copied = torch.zeros((len(ids), 4, len(LETTERS) + 1), device=h.device)
        copied.scatter_add_(2, ids[:, None, :].expand(-1, 4, -1), attention)
        root_pool = x.sum(-1) / mask.sum(-1).clamp_min(1)
        residual = self.root_residual(root_pool).view(-1, 4, len(LETTERS) + 1)
        generated = (torch.stack([head(pooled) for head in self.generators], 1) + residual).softmax(-1)
        gate = self.gate(pooled).sigmoid().unsqueeze(-1)
        return (gate * copied + (1 - gate) * generated).clamp_min(1e-12).log()


class StandaloneRootPointer(RootModel):
    """Root features learned from scratch, with no pretrained lexical features."""
    def __init__(self, base, width=48):
        super().__init__(width)

    def forward(self, h, pooled, ids):
        return super().forward(ids)


def roots_for(rows):
    return [sorted({a['root'] or '' for a in row['analyses']}) for row in rows]


def targets_for(rows):
    roots = roots_for(rows)
    targets = torch.zeros((len(rows), max(map(len, roots)), 4), dtype=torch.long)
    valid = torch.zeros(targets.shape[:2], dtype=torch.bool)
    for i, choices in enumerate(roots):
        for j, root in enumerate(choices):
            targets[i, j, :len(root)] = torch.tensor([LETTERS.index(c) + 1 for c in root])
            valid[i, j] = True
    return targets, valid


def root_loss(logp, targets, valid):
    expanded = logp[:, None].expand(-1, targets.shape[1], -1, -1)
    joint = expanded.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(-1)
    # Marginalize unique complete roots, never combine radicals from incompatible references.
    return -joint.masked_fill(~valid, -torch.inf).logsumexp(-1).mean()


def decode_roots(logp):
    indices = logp.argmax(-1)
    indices[:, :3] = logp[:, :3, 1:].argmax(-1) + 1
    best = logp.gather(-1, indices.unsqueeze(-1)).squeeze(-1).sum(-1)
    blank = logp[:, :, 0].sum(-1)
    return ['' if blank[i] > best[i] else ''.join(LETTERS[c - 1] for c in row if c) for i, row in enumerate(indices.tolist())]


def report(rows, predictions, training_roots):
    allowed = roots_for(rows)
    hits = [p in roots for p, roots in zip(predictions, allowed)]
    groups = {'all': list(range(len(rows))), 'nonNullOnly': [], 'unseenRoot': [], 'seenRoot': [],
              'allRadicalsVisible': [], 'requiresRestoration': [], 'nullAllowed': []}
    for i, (row, roots) in enumerate(zip(rows, allowed)):
        groups['nullAllowed' if '' in roots else 'nonNullOnly'].append(i)
        groups['seenRoot' if set(roots) & training_roots else 'unseenRoot'].append(i)
        visible = any(root and all(c in row['word'] for c in root) for root in roots)
        groups['allRadicalsVisible' if visible else 'requiresRestoration'].append(i)
        for label in row.get('slices', []): groups.setdefault(label, []).append(i)
    return {label: {'correct': sum(hits[i] for i in indices), 'count': len(indices),
                    'exactMatch': sum(hits[i] for i in indices) / len(indices) if indices else None}
            for label, indices in groups.items()}


def load_base():
    path = ROOT / BASE_DIRECTORY
    manifest = read_json(path / 'manifest.json')
    selected = {'id': manifest['id'], 'directory': BASE_DIRECTORY,
                'manifestSha256': sha(path / 'manifest.json'), 'status': 'experimental-unpromoted'}
    if sha(path / 'weights.bin') != manifest['sha256']: raise ValueError('Base checkpoint hash mismatch')
    base = Student(manifest['width'], manifest['labels'])
    signed = np.frombuffer((path / 'weights.bin').read_bytes(), dtype=np.int8)
    state = {}
    for name, spec in manifest['tensors'].items():
        values = (signed[spec['offset']:spec['offset'] + spec['length']].astype(np.float32) * np.float32(spec['scale'])).reshape(spec['shape'])
        if name in ('embedding', 'position'): key = name + '.weight'
        elif name.startswith('conv'):
            layer, suffix = name.split('.')
            key = f'convs.{layer[-1]}.{suffix}'
            if suffix == 'weight': values = values.transpose(0, 2, 1)
        else: key = name if name.startswith('segmentation') else 'heads.' + name
        state[key] = torch.from_numpy(values.copy())
    base.load_state_dict(state)
    base.requires_grad_(False).eval()
    return base, manifest, selected


def run(epochs=80, seed=42, mode='pointer', dataset_directory=None):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    dataset_dir = __import__('pathlib').Path(dataset_directory) if dataset_directory else DATA / 'generated/teacher-corrected-pilot-v1'
    if dataset_directory:
        from .root_data import load
        dataset, roles = load(dataset_dir)
    else: dataset, roles = load_corrected_dataset(dataset_dir)
    base, manifest, selected = load_base()
    training_roots = set(root for choices in roots_for(roles['train']) for root in choices)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-root-' + mode
    path = RUNS / run_id
    path.mkdir(parents=True)
    config = {'experiment': 'frozen-cnn-root-pointer-v1', 'mode': mode, 'seed': seed, 'epochs': epochs,
              'base': selected, 'baseWeightsSha256': manifest['sha256'], 'dataset': dataset['roles'],
              'datasetManifestSha256': sha(dataset_dir / 'manifest.json'), 'datasetDirectory': str(dataset_dir),
              'selection': 'lowest verification marginal loss over complete allowed roots',
              'rootSupervision': 'unique complete roots, marginalized without analysis-count weighting',
              'frozen': 'embedding, positions, convolution layers and every non-root head',
              'optimizer': 'AdamW', 'learningRate': .003, 'weightDecay': .001, 'batchSize': 128,
              'gradientClip': 5, 'threads': 4, 'device': 'cpu', 'torch': torch.__version__, 'numpy': np.__version__,
              'sourceHashes': {name: sha(__import__('pathlib').Path(__file__).parent / name)
                               for name in ['root_experiment.py', 'root_model.py', 'root_data.py', 'model.py']},
              'finalTestOpened': False, 'aiAnnotationsUsed': False, **provenance()}
    write_json(path / 'config.json', config)
    cache = {}
    for role, rows in roles.items():
        ids = encode_words([r['word'] for r in rows])
        standalone = mode == 'standalone'
        hs = torch.empty((len(ids), 1 if standalone else 32, 1 if standalone else base.width))
        pools = torch.empty((len(ids), 1 if standalone else base.width))
        old_roots = []
        with torch.no_grad():
            for begin in range(0, len(ids), 128):
                if standalone and role == 'train': break
                batch = ids[begin:begin + 128]
                state = base(batch, trace=True)
                if not standalone:
                    hs[begin:begin + len(batch)] = state['conv2']
                    pools[begin:begin + len(batch)] = state['pooled']
                if role == 'verification': old_roots.append(torch.stack([state[f'root{i}'].log_softmax(-1) for i in range(4)], 1))
        cache[role] = (hs, pools, ids, *targets_for(rows))
        if role == 'verification':
            old = torch.cat(old_roots)
            baseline = report(rows, decode_roots(old), training_roots)
            write_json(path / 'baseline.json', baseline)
            print(json.dumps({'baseline': baseline}), flush=True)
    audit = {'verification': {'words': len(roles['verification']),
             'wholeWordStemAllowed': sum(any(len(a['spans']) == 1 and a['spans'][0]['type'] == 'stem' for a in r['analyses']) for r in roles['verification']),
             'multipleRoots': sum(len(x) > 1 for x in roots_for(roles['verification']))}}
    write_json(path / 'audit.json', audit)
    model = StandaloneRootPointer(base) if mode == 'standalone' else CNNRootPointer(base) if mode == 'cnn-pointer' else RootPointer(base)
    for parameter in model.generators.parameters(): parameter.requires_grad_(True)
    if mode == 'generator':
        model.pointer.requires_grad_(False)
        model.gate.requires_grad_(False)
        with torch.no_grad(): model.gate.bias.fill_(-30)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003, weight_decay=.001)
    start = time.perf_counter()
    history, best, best_state, best_epoch = [], float('inf'), None, None
    h, pooled, ids, targets, valid = cache['train']
    vh, vp, vi, vt, vv = cache['verification']
    for epoch in range(epochs):
        model.train()
        losses = []
        for indices in torch.randperm(len(ids)).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = root_loss(model(h[indices], pooled[indices], ids[indices]), targets[indices], valid[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            output = model(vh, vp, vi)
            val = float(root_loss(output, vt, vv))
            metrics = report(roles['verification'], decode_roots(output), training_roots)
        entry = {'epoch': epoch + 1, 'trainLoss': float(np.mean(losses)), 'verificationLoss': val,
                 'verificationRootExact': metrics['all']['exactMatch'], 'elapsedSeconds': time.perf_counter() - start}
        history.append(entry)
        if val < best:
            best, best_epoch, best_state = val, epoch + 1, copy.deepcopy(model.state_dict())
            torch.save({'state': best_state, 'width': base.width, 'baseDirectory': selected['directory']}, path / 'root-head.pt')
            write_json(path / 'metrics.json', metrics)
            write_json(path / 'predictions.json', [{'word': r['word'], 'root': p or None} for r, p in zip(roles['verification'], decode_roots(output))])
        write_json(path / 'history.json', history)
        if epoch % 5 == 0 or epoch == epochs - 1: print(json.dumps(entry), flush=True)
    write_json(path / 'result.json', {'bestEpoch': best_epoch, 'verificationLoss': best,
               'metrics': read_json(path / 'metrics.json'), 'baseline': baseline,
               'elapsedSeconds': time.perf_counter() - start, 'addedParameters': sum(p.numel() for p in model.parameters()) - sum(p.numel() for head in model.generators for p in head.parameters()),
               'finalTestOpened': False, 'aiAnnotationsUsed': False, **provenance()})
    print('ROOT_RUN=' + str(path), flush=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['pointer', 'generator', 'cnn-pointer', 'standalone'], default='pointer')
    parser.add_argument('--dataset')
    args = parser.parse_args()
    run(args.epochs, args.seed, args.mode, args.dataset)
