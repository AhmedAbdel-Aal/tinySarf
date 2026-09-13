"""Export a root candidate, preserving the existing non-root tensors exactly."""
import argparse
from pathlib import Path
import torch
from .artifacts import ROOT, DATA, read_json, write_json, sha, provenance
from .model import Student, encode_words, numpy_reference
from .root_experiment import load_base, roots_for
from .export import export_model, parity


def export(root_run, decoder_run, output):
    root_run, decoder_run, output = map(Path, (root_run, decoder_run, output))
    if output.exists(): raise ValueError('Refusing to overwrite an exported candidate')
    root_config = read_json(root_run / 'config.json')
    root_result = read_json(root_run / 'result.json')
    decoder_result = read_json(decoder_run / 'result.json')
    if root_config['mode'] != 'standalone' or Path(decoder_result['rootRun']).resolve() != root_run.resolve():
        raise ValueError('Decoder/model experiment mismatch')
    base, original, selected = load_base()
    if original['sha256'] != root_config['baseWeightsSha256']: raise ValueError('Base checkpoint changed')
    corrected = read_json(DATA / 'generated/teacher-corrected-pilot-v1/verification.json')['records']
    legacy = read_json(DATA / 'generated/teacher-sampled-v2/verification.json')['records']
    if [r['word'] for r in corrected] != [r['word'] for r in legacy] or roots_for(corrected) != roots_for(legacy):
        raise ValueError('Root evaluation targets changed across the comparison')
    checkpoint = torch.load(root_run / 'root-head.pt', weights_only=True)
    root_width = checkpoint['state']['pointer.weight'].shape[1]
    model = Student(base.width, base.labels, root_width=root_width)
    state = {k: v for k, v in base.state_dict().items() if not k.startswith('heads.root')}
    state.update({'root_model.' + k: v for k, v in checkpoint['state'].items()})
    model.load_state_dict(state)
    model.eval()
    output.mkdir(parents=True)
    manifest, arrays, quant = export_model(model, output, output.name, original['verificationDigest'])
    stem_directory = Path(decoder_result['stemTemplates'])
    settings = decoder_result['best']
    manifest['rootDecoder'] = {'format': 'learned-transforms-v1',
        'priorWeight': settings['priorWeight'], 'unsupportedPenalty': settings['unsupportedPenalty'],
        'nullPenalty': settings['nullPenalty'], 'minimumDistinctRoots': decoder_result['minimumDistinctRoots'],
        'whole': read_json(decoder_run / 'templates.json'), 'stem': read_json(stem_directory / 'templates.json')}
    write_json(output / 'manifest.json', manifest)
    # The int8 non-root arrays have identical values and scales to the old package.
    original_bytes = (ROOT / selected['directory'] / 'weights.bin').read_bytes()
    packed = (output / 'weights.bin').read_bytes()
    unchanged = {}
    for name, before in original['tensors'].items():
        if name.startswith('root'): continue
        after = manifest['tensors'][name]
        unchanged[name] = (before['scale'] == after['scale'] and before['shape'] == after['shape'] and
            original_bytes[before['offset']:before['offset'] + before['length']] == packed[after['offset']:after['offset'] + after['length']])
    if not all(unchanged.values()): raise ValueError('A supposedly frozen non-root tensor changed')
    words = [r['word'] for r in legacy[:128]] + ['ك', 'ك' * 32, 'وبكتابهم']
    reference = parity(model, arrays, quant, words)
    write_json(output / 'reference-parity.json', {**reference, 'model': manifest['id'], **provenance()})
    ids = encode_words(words).numpy()
    write_json(output / 'parity-inputs.json', {'words': words,
        'float': {k: v.tolist() for k, v in numpy_reference(ids, arrays).items()},
        'quantized': {k: v.tolist() for k, v in numpy_reference(ids, quant).items()}})
    torch.save({'state': model.state_dict(), 'width': model.width, 'rootWidth': root_width, 'labels': model.labels, 'rootDecoder': manifest['rootDecoder']}, output / 'checkpoint.pt')
    config = {'experiment': 'root-rethink-v1', 'actualParameters': manifest['trainingParameters'],
        'targetParameters': 250000, 'width': model.width, 'rootWidth': root_width, 'embedding': 32,
        'baseModel': original['id'], 'baseWeightsSha256': original['sha256'],
        'rootTraining': root_config,
        'rootTrainingPools': list({pool['directory']: pool for pool in [
            {'directory': root_config['datasetDirectory'], 'sha256': root_config['dataset']['train']['sha256']},
            {'directory': decoder_result.get('sourceDatasetDirectory', root_config['datasetDirectory']), 'sha256': decoder_result['data']['train']['sha256']}
        ]}.values()),
        'rootDecoderTraining': {'datasetDirectory': decoder_result.get('sourceDatasetDirectory', root_config['datasetDirectory']), 'dataset': decoder_result['data'], 'stemTemplatesDirectory': str(stem_directory)},
        'rootRun': str(root_run), 'decoderRun': str(decoder_run),
        'rootTargetsIdenticalAcrossVerificationMappings': True, 'nonRootTensorsUnchanged': unchanged,
        'verificationSelection': 'root marginal loss for checkpoint; root exact agreement for transformation settings',
        'finalTestOpened': False, 'aiAnnotationsUsedForRoot': False, **provenance()}
    write_json(output / 'config.json', config)
    write_json(output / 'history.json', read_json(root_run / 'history.json'))
    write_json(output / 'result.json', {'schemaVersion': 1, 'runId': manifest['id'], 'model': manifest,
        'training': {'bestEpoch': root_result['bestEpoch'], 'verificationLoss': root_result['verificationLoss'],
                     'elapsedSeconds': root_result['elapsedSeconds']},
        'data': root_config['dataset'], 'rootDevelopmentMetrics': settings['metrics'],
        'eligibility': {'eligible': False, 'reasons': ['Experimental candidate; independent human gold is absent',
            'Public runtime quantization, browser parity and performance gates still required']}, **provenance()})
    print(__import__('json').dumps({'output': str(output), 'parameters': manifest['trainingParameters'],
          'reachableWeights': manifest['reachableWeights'], 'packedWeightsBytes': manifest['packedWeightsBytes'],
          'floatParityPassed': reference['floatPassed'], 'nonRootTensorsUnchanged': all(unchanged.values())}), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-run', required=True)
    parser.add_argument('--decoder-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export(args.root_run, args.decoder_run, args.output)
