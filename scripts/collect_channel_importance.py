#!/usr/bin/env python3
"""Observe the original BF16 model and freeze channel selections; no weight surgery."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / 'data/importance_v1/latest.json'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def rank_channels(scores, keep):
    require(type(keep) is int and 0 < keep < len(scores), 'Invalid retained width')
    require(all(math.isfinite(x) and x >= 0 for x in scores), 'Scores must be finite and nonnegative')
    return sorted(sorted(range(len(scores)), key=lambda j: (-scores[j], j))[:keep])


def random_channels(width, keep, layer):
    require(type(keep) is int and 0 < keep < width and type(layer) is int and layer >= 0,
            'Invalid random selection arguments')
    ids = list(range(width))
    random.Random(f'qwen-mlp-pruning-v1:42:{layer}').shuffle(ids)
    return sorted(ids[:keep])


def channel_diagnostics(scores, selections, layer):
    total = math.fsum(scores)
    row = {'layer': layer, 'score_min': min(scores), 'score_max': max(scores), 'score_sum': total}
    for method, ids in selections.items():
        row[method + '_retained_score_fraction'] = math.fsum(scores[j] for j in ids) / total if total > 0 else None
    return row


def gpu_identity():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    fields = [x.strip() for x in text.split(',')]
    require(len(text.splitlines()) == 1 and len(fields) == 3, 'Expected one physical GPU')
    return dict(zip(('uuid', 'name', 'driver'), fields))


def gpu_pids():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return {int(x.strip()) for x in text.splitlines() if x.strip().isdigit()}


def verify_inputs():
    path = ROOT / 'data/protocol_v1.json'
    p = read(path)
    require(p['study'] == 'qwen3_structured_mlp_pruning_pilot_v1' and p['target_intermediate_size'] == 9216,
            'Unexpected pruning study')
    require(p['architecture'] == {'model_type': 'qwen3', 'num_hidden_layers': 36, 'hidden_size': 2560,
            'intermediate_size': 9728, 'hidden_act': 'silu', 'tie_word_embeddings': True}, 'Architecture differs')
    require(p['collection'] == {'dtype': 'bfloat16', 'attention_backend': 'eager', 'batch_size': 1,
            'use_cache': False, 'seed': 42, 'gradients': False}, 'Collection settings differ')
    require(p['methods']['ActivationEnergy']['score'] ==
            'mean_tokens(activation_j ** 2) * sum_output_dimensions(down_proj_weight[:,j] ** 2)'
            and p['methods']['RandomChannels']['seed'] == 42, 'Selection rule differs')
    require(sha(ROOT / 'scripts/prepare_pruning_study.py') == p['preparation_script_sha256'], 'Preparation script changed')
    require({k: version(k) for k in p['packages']} == p['packages'], 'Recorded package versions changed')
    require(sys.version.split()[0] == p['python'], 'Python version changed since preparation')
    freeze = subprocess.check_output(['uv', 'pip', 'freeze', '--python', sys.executable], text=True, stderr=subprocess.PIPE)
    require(digest(freeze.encode()) == p['environment_sha256'] and sha(ROOT / 'data/environment_v1.txt') == p['environment_sha256'],
            'Environment changed; retain the working environment')
    require(gpu_identity() == p['gpu_identity'], 'Different physical GPU from preparation')
    print('Verifying frozen source and calibration files (read-only)...', flush=True)
    for relative, expected in {**p['source_file_sha256'], **p['calibration_evidence_sha256']}.items():
        require(sha(Path(relative)) == expected, 'Frozen input changed: ' + relative)
    data_path = ROOT / p['calibration']['file']
    require(sha(data_path) == p['calibration']['sha256'], 'Local calibration copy changed')
    rows = [json.loads(line) for line in data_path.read_bytes().splitlines()]
    require(len(rows) == 128 and [x['id'] for x in rows] == p['calibration']['sequence_ids'], 'Calibration order differs')
    require(Counter(x['domain'] for x in rows) == {'general': 64, 'math': 64}
            and all(len(x['input_ids']) == 512 for x in rows), 'Calibration shape/balance differs')
    return p, rows


def verify_saved(path, p):
    import torch
    from safetensors.torch import load_file
    pointer = read(path)
    manifest_path = ROOT / pointer['manifest']
    require(manifest_path.resolve().is_relative_to(ROOT.resolve()), 'Invalid manifest path')
    require(sha(manifest_path) == pointer['manifest_sha256'], 'Importance manifest hash differs')
    r = read(manifest_path)
    require(r['protocol_sha256'] == sha(ROOT / 'data/protocol_v1.json')
            and r['collector_script_sha256'] == sha(Path(__file__))
            and r['environment_sha256'] == p['environment_sha256']
            and r['gpu_identity'] == p['gpu_identity'], 'Importance run identity differs')
    for relative, expected in r['file_sha256'].items():
        require(sha(ROOT / relative) == expected, 'Importance output changed: ' + relative)
    require(sha(Path(r['modeling_source_path'])) == r['modeling_source_sha256'], 'Installed model implementation changed')
    values = load_file(str(ROOT / r['statistics_file']), device='cpu')
    require(set(values) == {'activation_squared_sums', 'activation_mean_squares', 'down_weight_squared_norms', 'scores'},
            'Unexpected importance tensors')
    for x in values.values():
        require(x.dtype == torch.float64 and tuple(x.shape) == (36, 9728)
                and torch.isfinite(x).all().item() and (x >= 0).all().item(), 'Invalid importance statistics')
    require(r['layer_token_counts'] == [65536] * 36 and r['layer_call_counts'] == [128] * 36,
            'Not all layers observed all calibration tokens')
    require(torch.equal(values['activation_mean_squares'], values['activation_squared_sums'] / 65536)
            and torch.equal(values['scores'], values['activation_mean_squares'] * values['down_weight_squared_norms']),
            'Score calculation differs')
    selection = read(ROOT / r['selection_file'])
    require(selection['protocol_sha256'] == r['protocol_sha256'] and selection['statistics_sha256'] == sha(ROOT / r['statistics_file'])
            and selection['target_intermediate_size'] == 9216, 'Selection identity differs')
    for method in ('RandomChannels', 'ActivationEnergy'):
        require(set(selection['methods'][method]) == {str(i) for i in range(36)}, 'Selection layer coverage differs')
    for layer in range(36):
        scores = values['scores'][layer].tolist()
        require(selection['methods']['RandomChannels'][str(layer)] == random_channels(9728, 9216, layer),
                'Random selection differs')
        require(selection['methods']['ActivationEnergy'][str(layer)] == rank_channels(scores, 9216),
                'Activation selection differs')
        expected = channel_diagnostics(scores, {m: selection['methods'][m][str(layer)]
                                               for m in ('RandomChannels', 'ActivationEnergy')}, layer)
        require(r['layer_diagnostics'][layer] == expected, 'Score-mass diagnostic differs')
    passes = read(ROOT / r['calibration_passes_file'])
    require(len(passes) == 128 and [x['id'] for x in passes] == p['calibration']['sequence_ids']
            and all(math.isfinite(x['forward_and_statistics_seconds']) and x['forward_and_statistics_seconds'] > 0
                    for x in passes), 'Saved calibration passes differ')
    require(r['hook_probe_output_exact'] and r['parameter_metadata_preserved'] and r['source_files_unchanged'],
            'Observation-only checks incomplete')
    return r


def print_summary(r):
    for method in ('RandomChannels', 'ActivationEnergy'):
        fractions = [x[method + '_retained_score_fraction'] for x in r['layer_diagnostics']
                     if x[method + '_retained_score_fraction'] is not None]
        if fractions:
            print(f'{method}: median retained score mass={100 * statistics.median(fractions):.4f}%')
    print('Calibration sequences: 128; tokens observed per layer: 65,536')
    print('Layers scored: 36; channels scored:', 36 * 9728)
    print('Retained channels per layer: 9,216 for both methods')
    print('Saved selections:', ROOT / r['selection_file'])
    print('PASS: original-model observation, finite importance scores, exact hook probe, source hashes and both fixed channel selections verified.')
    print('Retained score mass is a proxy, not a model-quality result. No pruning, export or fine-tuning performed.')


def main():
    p, rows = verify_inputs()
    if POINTER.exists():
        r = verify_saved(POINTER, p)
        print('Verified existing completed importance run; no forward passes repeated.')
        print_summary(r)
        return
    require(not gpu_pids(), 'Another GPU process is active; stop its server before collection')
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA/BF16 unavailable')
    setup_pids = gpu_pids()
    require(len(setup_pids) <= 1, 'Multiple GPU contexts appeared during setup')
    allowed = setup_pids | {os.getpid()}
    free, _ = torch.cuda.mem_get_info()
    print(f'Free GPU memory: {free / 2**30:.2f} GiB', flush=True)
    require(free >= 16 * 2**30, 'Need at least 16 GiB free GPU memory')
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    torch.manual_seed(42)
    for key in ('CUDA_LAUNCH_BLOCKING', 'TRITON_INTERPRET'):
        require(os.environ.get(key, '') in ('', '0'), 'Debug mode active: ' + key)
    folder = ROOT / 'outputs/importance_v1' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    print('Loading original BF16 model on CPU, then moving it to GPU...', flush=True)
    load_start = time.perf_counter()
    # Avoid introducing an accelerate dependency into the existing serving environment.
    model, loading = AutoModelForCausalLM.from_pretrained(
        p['source_model']['local_dir'], dtype=torch.bfloat16, attn_implementation='eager',
        local_files_only=True, output_loading_info=True,
    )
    require(not loading.get('unexpected_keys') and not loading.get('mismatched_keys') and not loading.get('error_msgs'),
            'Source weights did not load cleanly: ' + str(loading))
    require(set(loading.get('missing_keys', [])) <= {'lm_head.weight'}, 'Missing model weights: ' + str(loading))
    model = model.to('cuda').eval()
    model.requires_grad_(False)
    torch.cuda.synchronize()
    print(f'Model load/move seconds: {time.perf_counter()-load_start:.2f}', flush=True)
    backbone = model.model
    require(len(backbone.layers) == 36 and model.config.intermediate_size == 9728, 'Model architecture differs')
    require(model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(),
            'Tied embeddings/LM head were not preserved')
    require(sum(x.numel() for x in model.parameters()) == p['source_serialized_tensor_elements'],
            'Loaded unique parameter count differs from source inventory')
    require(all(x.dtype == torch.bfloat16 and x.device.type == 'cuda' and not x.requires_grad for x in model.parameters()),
            'Expected frozen BF16 GPU weights')
    modeling_path = Path(inspect.getfile(type(backbone.layers[0].mlp))).resolve()
    modeling_hash = sha(modeling_path)

    def fingerprint():
        result = {}
        for name, x in model.named_parameters():
            try:
                v = x._version
            except RuntimeError:
                v = None  # Inference tensors do not expose a version counter.
            result[name] = (id(x), x.data_ptr(), tuple(x.shape), str(x.dtype), v)
        return result

    before_parameters = fingerprint()
    sums = torch.zeros((36, 9728), device='cuda', dtype=torch.float64)
    weight_norms = torch.empty((36, 9728), device='cpu', dtype=torch.float64)
    token_counts, call_counts, seen = [0]*36, [0]*36, set()
    handles = []
    for layer, block in enumerate(backbone.layers):
        mlp = block.mlp
        for name, shape in (('gate_proj',(9728,2560)),('up_proj',(9728,2560)),('down_proj',(2560,9728))):
            linear = getattr(mlp, name)
            require(isinstance(linear, torch.nn.Linear) and tuple(linear.weight.shape) == shape and linear.bias is None,
                    'Unsupported MLP layout')
        weight_norms[layer].copy_(mlp.down_proj.weight.detach().float().square().sum(dim=0, dtype=torch.float32).double().cpu())

    def forward(item):
        ids = torch.tensor([item['input_ids']], device='cuda', dtype=torch.long)
        output = backbone(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                          output_hidden_states=False, output_attentions=False, return_dict=True)
        return output.last_hidden_state

    def make_hook(layer):
        def hook(module, inputs):
            a = inputs[0]
            require(a.dtype == torch.bfloat16 and tuple(a.shape) == (1,512,9728)
                    and layer not in seen, 'Unexpected activation shape/dtype or repeated layer')
            sums[layer].add_(a.detach().float().square().sum(dim=(0,1), dtype=torch.float32).double())
            seen.add(layer)
            token_counts[layer] += 512
            call_counts[layer] += 1
            # No return value: this hook observes and does not replace the input.
        return hook

    records = []
    try:
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            baseline = forward(rows[0]).clone()
            for layer, block in enumerate(backbone.layers):
                handles.append(block.mlp.down_proj.register_forward_pre_hook(make_hook(layer)))
            seen.clear()
            probe = forward(rows[0])
            require(seen == set(range(36)) and torch.equal(baseline, probe), 'Observation hook changed the probe output')
            print('PASS: hooked and unhooked backbone outputs match exactly.', flush=True)
            del baseline, probe
            # The probe is excluded from the calibration statistics/counts.
            sums.zero_(); token_counts[:] = [0]*36; call_counts[:] = [0]*36
            for number, item in enumerate(rows, 1):
                seen.clear()
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = forward(item)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                require(seen == set(range(36)), 'A calibration pass missed an MLP layer')
                del output
                records.append({'id': item['id'], 'domain': item['domain'], 'forward_and_statistics_seconds': seconds})
                if number == 1 or number % 8 == 0:
                    print(f'Calibration {number}/128: all 36 layers observed', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    require(token_counts == [65536]*36 and call_counts == [128]*36, 'Calibration count mismatch')
    require(fingerprint() == before_parameters, 'Model parameter identities/shapes/version counters changed')
    require(not (gpu_pids() - allowed), 'Another GPU process appeared during collection')
    values = {
        'activation_squared_sums': sums.cpu().contiguous(),
        'down_weight_squared_norms': weight_norms.contiguous(),
    }
    values['activation_mean_squares'] = values['activation_squared_sums'] / 65536
    values['scores'] = values['activation_mean_squares'] * values['down_weight_squared_norms']
    for x in values.values():
        require(torch.isfinite(x).all().item() and (x >= 0).all().item(), 'Nonfinite/negative score statistics')
    stats_path = folder / 'channel_statistics.safetensors'
    save_file(values, str(stats_path), metadata={'protocol_sha256': sha(ROOT / 'data/protocol_v1.json')})
    selections = {'RandomChannels': {}, 'ActivationEnergy': {}}
    diagnostics = []
    for layer in range(36):
        scores = values['scores'][layer].tolist()
        selections['RandomChannels'][str(layer)] = random_channels(9728, 9216, layer)
        selections['ActivationEnergy'][str(layer)] = rank_channels(scores, 9216)
        diagnostics.append(channel_diagnostics(scores, {m: selections[m][str(layer)] for m in selections}, layer))
    selection_path = folder / 'channel_selections.json'
    save(selection_path, {'protocol_sha256': sha(ROOT / 'data/protocol_v1.json'), 'statistics_sha256': sha(stats_path),
                          'target_intermediate_size': 9216, 'methods': selections})
    sequence_path = folder / 'calibration_passes.json'
    save(sequence_path, records)
    print('Checking that frozen source files remain unchanged...', flush=True)
    for relative, expected in p['source_file_sha256'].items():
        require(sha(Path(relative)) == expected, 'Source checkpoint file changed')
    require(sha(modeling_path) == modeling_hash and gpu_identity() == p['gpu_identity'], 'Implementation or GPU changed')
    r = {
        'study': 'original_bf16_channel_importance_v1', 'protocol_sha256': sha(ROOT / 'data/protocol_v1.json'),
        'collector_script_sha256': sha(Path(__file__)), 'environment_sha256': p['environment_sha256'],
        'gpu_identity': p['gpu_identity'], 'packages': p['packages'],
        'modeling_source_path': str(modeling_path), 'modeling_source_sha256': modeling_hash,
        'layer_token_counts': token_counts, 'layer_call_counts': call_counts, 'layer_diagnostics': diagnostics,
        'hook_probe_output_exact': True, 'parameter_metadata_preserved': True, 'source_files_unchanged': True,
        'statistics_file': str(stats_path.relative_to(ROOT)), 'selection_file': str(selection_path.relative_to(ROOT)),
        'calibration_passes_file': str(sequence_path.relative_to(ROOT)),
        'file_sha256': {str(x.relative_to(ROOT)): sha(x) for x in (stats_path, selection_path, sequence_path)},
        'scope': 'Original-model activation statistics and frozen channel selections. No surgery, export, gradient updates, quality evaluation or speed measurements.',
        'proxy_note': 'Retained score mass ignores correlations between channels and is not a guarantee of quality or output reconstruction.'
    }
    manifest_path = folder / 'manifest.json'
    save(manifest_path, r)
    # Validate saved bytes before publishing a stable completion pointer.
    candidate = folder / 'completion_candidate.json'
    save(candidate, {'manifest': str(manifest_path.relative_to(ROOT)), 'manifest_sha256': sha(manifest_path)})
    verified = verify_saved(candidate, p)
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    with POINTER.open('x') as f:
        f.write(candidate.read_text())
    print_summary(verified)


if __name__ == '__main__':
    main()
