#!/usr/bin/env python3
"""Freeze a BF16 structured MLP-channel pruning pilot. Does not load or prune a model."""
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/root/autodl-tmp/qwen-inference-lab')
MODEL = ORIGINAL / 'models/Qwen3-4B-Instruct-2507'
CALIBRATION = ORIGINAL / 'data/calibration/seed_42'
KEEP = 9216
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def freeze(path, content):
    if path.exists():
        require(path.read_bytes() == content, 'Existing study file differs; preserve it: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(content)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def tensor_inventory(model, config):
    """Read safetensors headers and validate layout; never load weight tensors."""
    tensors, mapping, files = {}, {}, sorted(model.glob('*.safetensors'))
    require(files, 'Missing source weight shards')
    for path in files:
        with path.open('rb') as f:
            prefix = f.read(8)
            require(len(prefix) == 8, 'Invalid safetensors prefix')
            header_size = struct.unpack('<Q', prefix)[0]
            require(0 < header_size <= 32 * 1024**2, 'Unexpected safetensors header size')
            header = json.loads(f.read(header_size))
        ranges = []
        for name, value in header.items():
            if name == '__metadata__':
                continue
            require(name not in tensors, 'Duplicate source tensor: ' + name)
            require(value['dtype'] == 'BF16', 'Expected an all-BF16 source checkpoint: ' + name)
            shape = value['shape']
            require(isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape), 'Invalid tensor shape')
            a, b = value['data_offsets']
            require(type(a) is int and type(b) is int and 0 <= a < b
                    and b - a == math.prod(shape) * 2, 'Tensor byte range differs from shape')
            tensors[name] = {'shape': shape, 'dtype': value['dtype']}
            mapping[name] = path.name
            ranges.append((a, b))
        ranges.sort()
        require(ranges and ranges[0][0] == 0 and all(ranges[i-1][1] == ranges[i][0] for i in range(1, len(ranges)))
                and 8 + header_size + ranges[-1][1] == path.stat().st_size, 'Invalid source tensor data layout')
    layers, hidden, intermediate = (config[k] for k in ('num_hidden_layers', 'hidden_size', 'intermediate_size'))
    expected = {}
    for layer in range(layers):
        prefix = f'model.layers.{layer}.mlp.'
        for projection, shape in (('gate_proj', [intermediate, hidden]),
                                  ('up_proj', [intermediate, hidden]),
                                  ('down_proj', [hidden, intermediate])):
            name = prefix + projection + '.weight'
            require(tensors.get(name, {}).get('shape') == shape, 'Source projection differs: ' + name)
            expected[name] = shape
    require({n for n in tensors if '.mlp.' in n} == set(expected), 'Unexpected MLP parameters or biases')
    index = model / 'model.safetensors.index.json'
    if index.exists():
        require(read(index)['weight_map'] == mapping, 'Source weight index differs from shards')
    count = sum(math.prod(x['shape']) for x in tensors.values())
    return tensors, files, count


def validate_calibration(pool_bytes, subset_bytes, manifest, audit, subset_manifest, source, splits, vocab_size):
    require(digest(pool_bytes) == manifest['pool_sha256'] == audit['pool_sha256']
            == subset_manifest['pool_sha256'], 'Calibration pool hash differs')
    require(audit['verified_sequences'] == 512 and audit['source_tokens_match']
            and audit['math_reserved_rows_excluded'], 'Missing successful original token/split audit')
    require(manifest['model_source'] == source, 'Calibration model revision differs')
    pool = [json.loads(line) for line in pool_bytes.splitlines()]
    require(len(pool) == 512 and len({x['id'] for x in pool}) == 512, 'Pool coverage differs')
    by_id = {x['id']: x for x in pool}
    entry = subset_manifest['subsets']['Random128']
    require(digest(subset_bytes) == entry['sha256'], 'Random128 subset hash differs')
    rows = [json.loads(line) for line in subset_bytes.splitlines()]
    ids = [x['id'] for x in rows]
    require(ids == entry['sequence_ids'] and len(ids) == len(set(ids)) == 128,
            'Expected original ordered Random128 subset')
    require(Counter(x['domain'] for x in rows) == {'general': 64, 'math': 64}, 'Domain balance differs')
    allowed_math = {x['row'] for x in splits['partitions']['calibration']}
    reserved = {x['row'] for name in ('development', 'fewshot_reserve') for x in splits['partitions'][name]}
    require(not allowed_math.intersection(reserved), 'Original math partitions overlap')
    math_rows = set()
    for row in rows:
        require(row == by_id.get(row['id']), 'Subset sequence differs from audited pool')
        require(len(row['input_ids']) == 512 and all(type(x) is int and 0 <= x < vocab_size for x in row['input_ids']),
                'Calibration tokens invalid')
        require(row['sources'], 'Missing source spans')
        if row['domain'] == 'math':
            used = {span['row'] for span in row['sources']}
            require(used.issubset(allowed_math) and not used.intersection(reserved), 'Math development/fewshot leakage')
            math_rows.update(used)
    return rows, entry, len(math_rows)


def main():
    require(ROOT.resolve() != ORIGINAL.resolve(), 'Install in the separate qwen-pruning-lab project')
    for key in ('CUDA_LAUNCH_BLOCKING', 'TRITON_INTERPRET'):
        require(os.environ.get(key, '') in ('', '0'), 'Debug mode is active: ' + key)
    packages = {name: version(name) for name in ('torch', 'transformers', 'safetensors')}
    require(packages == {'torch': '2.13.0+cu132', 'transformers': '5.19.0', 'safetensors': '0.8.0'},
            'Environment differs from the inspected setup; preserve the working environment')
    config = read(MODEL / 'config.json')
    require((config['model_type'], config['num_hidden_layers'], config['hidden_size'], config['intermediate_size'],
             config['hidden_act'], config['tie_word_embeddings']) == ('qwen3', 36, 2560, 9728, 'silu', True),
            'Source architecture differs from inspection')
    require(not config.get('quantization_config'), 'Start from the BF16 checkpoint')
    source_path = ORIGINAL / 'logs/model_revision.json'
    source = read(source_path)
    require(source['repo_id'] == 'Qwen/Qwen3-4B-Instruct-2507' and source['revision'] == REVISION
            and Path(source['local_dir']).resolve() == MODEL.resolve(), 'Source identity differs')
    pool_path, pool_manifest_path = CALIBRATION / 'pool.jsonl', CALIBRATION / 'manifest.json'
    audit_path = CALIBRATION / 'pool_audit.json'
    subset_manifest_path = CALIBRATION / 'random_subsets_manifest.json'
    subset_manifest = read(subset_manifest_path)
    subset_relative = Path(subset_manifest['subsets']['Random128']['file'])
    require(not subset_relative.is_absolute() and '..' not in subset_relative.parts, 'Invalid subset path')
    subset_path = CALIBRATION / subset_relative
    split_path = ORIGINAL / 'data/splits/gsm8k_train_split.json'
    source_data_path = ORIGINAL / 'data/source_revisions.json'
    data_sources = read(source_data_path)
    require(all(data_sources[d]['split'] == 'train' for d in ('general', 'math')), 'Expected training calibration sources')
    require(read(split_path)['source'] == data_sources['math'], 'Math split source differs')
    pool_bytes, subset_bytes = pool_path.read_bytes(), subset_path.read_bytes()
    rows, _, math_rows = validate_calibration(pool_bytes, subset_bytes, read(pool_manifest_path), read(audit_path),
                                            subset_manifest, source, read(split_path), config['vocab_size'])
    inventory, shards, count = tensor_inventory(MODEL, config)
    removed = 3 * config['hidden_size'] * (config['intermediate_size'] - KEEP) * config['num_hidden_layers']
    require(KEEP % 256 == 0 and 0 < KEEP < config['intermediate_size'], 'Invalid target width')
    files = sorted(set(shards + [p for name in ('config.json', 'model.safetensors.index.json',
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json')
        if (p := MODEL / name).is_file()]))
    print('Hashing the original BF16 files (read-only)...', flush=True)
    source_hashes = {}
    for path in files:
        print('  ' + path.name, flush=True)
        source_hashes[str(path)] = sha(path)
    external = [source_path, pool_path, pool_manifest_path, audit_path, subset_path,
                subset_manifest_path, split_path, source_data_path]
    evidence = {str(p): sha(p) for p in external}
    gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    require(len(gpu_text.splitlines()) == 1, 'Expected one physical GPU')
    fields = [x.strip() for x in gpu_text.split(',')]
    require(len(fields) == 3, 'Unexpected GPU metadata')
    freeze_text = subprocess.check_output(['uv', 'pip', 'freeze', '--python', sys.executable],
                                         text=True, stderr=subprocess.PIPE)
    protocol = {
        'study': 'qwen3_structured_mlp_pruning_pilot_v1', 'source_model': source,
        'architecture': {k: config[k] for k in ('model_type', 'num_hidden_layers', 'hidden_size', 'intermediate_size',
                                              'hidden_act', 'tie_word_embeddings')},
        'target_intermediate_size': KEEP, 'removed_channels_per_layer': config['intermediate_size'] - KEEP,
        'mlp_width_reduction_percent': 100 * (config['intermediate_size'] - KEEP) / config['intermediate_size'],
        'source_serialized_tensor_elements': count, 'removed_mlp_weight_parameters': removed,
        'predicted_serialized_tensor_element_reduction_percent': 100 * removed / count,
        'predicted_bf16_tensor_payload_reduction_bytes': 2 * removed,
        'methods': {
            'BF16': 'Original unpruned BF16 checkpoint',
            'RandomChannels': {'seed': 42, 'per_layer_seed': 'qwen-mlp-pruning-v1:42:{layer}',
                'selection': 'Python random.Random(per_layer_seed).shuffle on list(range(9728)); keep first 9216, then sort retained indices ascending'},
            'ActivationEnergy': {
                'activation': 'Input to each original BF16 mlp.down_proj: SiLU(gate_proj(x)) * up_proj(x)',
                'score': 'mean_tokens(activation_j ** 2) * sum_output_dimensions(down_proj_weight[:,j] ** 2)',
                'statistics': 'Square/reduce per batch in float32, accumulate channel sums in float64; weight squared norms reduced in float32; form scores in float64',
                'selection': 'Keep top 9216 per layer; exact score ties choose lower channel index; store retained indices ascending',
                'interpretation': 'Per-channel output-contribution energy heuristic; ignores correlations; not an exact loss-sensitivity estimate',
                'collection': 'All 36 layers observed on the original unpruned model; no sequential pruning during collection',
            },
        },
        'calibration': {'method': 'Existing Random128 subset, pool seed 42', 'sequences': 128, 'tokens_per_sequence': 512,
                        'total_tokens': 65536, 'domain_sequences': {'general': 64, 'math': 64},
                        'file': 'data/calibration/random128_seed42.jsonl', 'sha256': digest(subset_bytes),
                        'pool_sha256': digest(pool_bytes), 'sequence_ids': [r['id'] for r in rows],
                        'distinct_math_training_rows': math_rows, 'math_reserved_rows_excluded': True},
        'collection': {'dtype': 'bfloat16', 'attention_backend': 'eager', 'batch_size': 1,
                       'use_cache': False, 'seed': 42, 'gradients': False},
        'surgery': 'Identical retained channel indices slice gate/up rows and down columns in every layer; physical dense tensor shrink; update global and per-MLP intermediate_size',
        'unchanged': ['attention', 'hidden size', 'layer count', 'embeddings and tied LM head', 'normalizations', 'tokenizer'],
        'export_dtype': 'bfloat16', 'fine_tuning_or_recovery': False, 'fp8_combination': False,
        'quality_policy': 'Choose and freeze evaluation prompts/scoring before scoring candidates. Use development data for debugging. Existing official benchmark test sets have been evaluated previously; any reuse must be disclosed, not described as a fresh unseen holdout.',
        'performance_policy': 'Freeze shared serving/benchmark settings before timing; report measured size, memory and latency rather than assuming parameter reduction guarantees speedup',
        'comparison_scope': 'Single width and one random channel seed; exploratory method comparison, not evidence across random seeds or pruning rates',
        'packages': packages, 'python': sys.version.split()[0], 'gpu_identity': dict(zip(('uuid','name','driver'), fields)),
        'environment_sha256': digest(freeze_text.encode()), 'source_file_sha256': source_hashes,
        'calibration_evidence_sha256': evidence, 'preparation_script_sha256': sha(Path(__file__)),
        'note': 'This preparation freezes source and selection rules only; no model was loaded, scored, pruned, exported or evaluated.'
    }
    # Freeze only after all read-only validations pass.
    freeze(ROOT / 'data/protocol_v1.json', json_bytes(protocol))
    freeze(ROOT / 'data/calibration/random128_seed42.jsonl', subset_bytes)
    freeze(ROOT / 'data/source_tensor_inventory.json', json_bytes(inventory))
    freeze(ROOT / 'data/environment_v1.txt', freeze_text.encode())
    print('\nOriginal / target MLP width:', config['intermediate_size'], '/', KEEP)
    print(f'MLP channels removed: {protocol["mlp_width_reduction_percent"]:.4f}%')
    print('MLP weight parameters removed:', f'{removed:,}')
    print(f'Predicted tensor-element reduction: {100 * removed / count:.4f}%')
    print(f'Predicted BF16 tensor-payload saving: {2 * removed / 2**20:.1f} MiB')
    print('Calibration: 128 sequences, 65,536 tokens; general/math = 64/64')
    print('Protocol SHA256:', sha(ROOT / 'data/protocol_v1.json'))
    print('Saved:', ROOT / 'data/protocol_v1.json')
    print('PASS: BF16 source, projection shapes, calibration hashes, math split separation and pilot protocol verified.')
    print('No weights modified and no pruning performance or quality result measured.')


if __name__ == '__main__':
    main()
