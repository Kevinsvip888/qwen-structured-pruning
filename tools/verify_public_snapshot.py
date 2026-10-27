#!/usr/bin/env python3
"""Verify the public study using saved bytes and CPU arithmetic only."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import runpy
import statistics

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'provenance/original'
REMOTE = '/root/autodl-tmp/qwen-pruning-lab/'
PINS = {'data/protocol_v1.json': 'f4482e32ad3e8b9564659856610fd9129607d194dc832d5faf48460b92d55592',
        'data/evaluation/pruning_dev_v1/protocol.json': '9ec3eb2d74fd9461d3de850a12fca8ac12b6e4eb15ecd48e72ea35b20d41b352',
        'data/benchmark/pruning_v1/protocol.json': '947676903d7210320e4b7e809d75c3474ddc8a212c6f1b04cb7f39ab36ff91c8'}
METHODS = ('BF16', 'RandomChannels', 'ActivationEnergy')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def original_path(relative):
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe archived path')
    if path.parts[0] in ('scripts', 'pruning_lab', 'specs'):
        return ROOT / path
    return ORIGINAL / path


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), label)


def verify_importance_and_exports():
    pruning = read(ORIGINAL / 'data/protocol_v1.json')
    pointer = read(ORIGINAL / 'data/importance_v1/latest.json')
    manifest_path = original_path(pointer['manifest'])
    require(sha(manifest_path) == pointer['manifest_sha256'], 'Importance pointer mismatch')
    manifest = read(manifest_path)
    require(manifest['protocol_sha256'] == PINS['data/protocol_v1.json']
            and manifest['hook_probe_output_exact'] and manifest['source_files_unchanged'], 'Importance provenance differs')
    for name, digest in manifest['file_sha256'].items():
        path = original_path(name)
        if path.is_file():
            require(sha(path) == digest, 'Included importance file changed')
    require(manifest['layer_call_counts'] == [128] * 36
            and manifest['layer_token_counts'] == [65536] * 36, 'Recorded calibration coverage differs')
    selections_path = original_path(manifest['selection_file'])
    selections = read(selections_path)['methods']
    surgery_pointer = read(ORIGINAL / 'data/surgery_validation_v1/latest.json')
    surgery_path = original_path(surgery_pointer['receipt'])
    require(sha(surgery_path) == surgery_pointer['sha256'], 'Surgery receipt pointer differs')
    surgery = read(surgery_path)
    for name, digest in surgery['file_sha256'].items():
        require(sha(original_path(name)) == digest, 'Validated surgery source changed: ' + name)
    require(len(surgery['toy_fp64_checks']) == 42 and len(surgery['real_fp64_checks']) == 144
            and len(surgery['native_bf16_smokes']) == 144
            and surgery['exact_weight_copy_checks'] == 144
            and len(surgery['invalid_inputs_rejected']) == 13, 'Recorded surgery check counts differ')
    require(surgery['importance_manifest_sha256'] == sha(manifest_path), 'Surgery/importance identity differs')
    for method in METHODS[1:]:
        retained = selections[method]
        require(set(retained) == {str(i) for i in range(36)}, 'Missing selected layer')
        for layer, ids in retained.items():
            require(len(ids) == 9216 and ids == sorted(set(ids))
                    and all(type(x) is int and 0 <= x < 9728 for x in ids), 'Invalid retained indices')
            if method == 'RandomChannels':
                expected = list(range(9728))
                random.Random(f'qwen-mlp-pruning-v1:42:{layer}').shuffle(expected)
                require(ids == sorted(expected[:9216]), 'Random-channel reconstruction differs')
        path = ORIGINAL / 'models' / ('Qwen3-4B-Instruct-2507-MLP9216-' + method) / 'pruning_manifest.json'
        exported = read(path)
        identity = exported['identity']
        require(identity['protocol_sha256'] == PINS['data/protocol_v1.json']
                and identity['method'] == method and identity['dtype'] == 'bfloat16'
                and not identity['quantization'] and not identity['fine_tuning'], 'Export identity differs')
        require(identity['importance_manifest_sha256'] == sha(manifest_path)
                and identity['surgery_validation_sha256'] == sha(surgery_path), 'Export source provenance differs')
        require(exported['retained_indices'] == retained, 'Exported channel indices differ')
        require(exported['original_unique_parameters'] == 4022468096
                and exported['pruned_unique_parameters'] == 3880910336
                and exported['removed_parameters'] == 141557760, 'Export parameter totals differ')
        require(all(exported['reload_checks'][key] for key in ('tied_embeddings',
                'probe_hidden_states_exact', 'probe_logits_exact', 'tokenizer_unchanged')),
                'Recorded export reload checks incomplete')
        audit = exported['disk_tensor_audit']
        require(audit['checked_tensors'] == 398 and audit['sliced_mlp_tensors'] == 108
                and audit['unchanged_tensors'] == 290, 'Recorded export tensor counts differ')
    require(pruning['removed_mlp_weight_parameters'] == 36 * 3 * 2560 * (9728 - 9216),
            'Pruning parameter arithmetic differs')
    print('PASS: fixed selections, random-channel reconstruction, source pins and recorded surgery/export receipts.')
    print('Scope: statistics/weight tensors are omitted; activation ranking and tensor surgery were not re-executed.')


def verify_benchmarks():
    plan = read(ORIGINAL / 'data/benchmark/pruning_v1/protocol.json')
    summary = read(ROOT / 'results/benchmark/benchmark_summary.json')
    require(summary == read(ORIGINAL / 'outputs/benchmark/pruning_v1/benchmark_summary.json'), 'Benchmark result copy differs')
    require(summary['protocol_sha256'] == PINS['data/benchmark/pruning_v1/protocol.json']
            and summary['gpu_identity'] == plan['gpu_identity']
            and summary['serve_environment'] == plan['serve_environment'], 'Benchmark summary identity differs')
    require(plan['methods'] == list(METHODS) and len(plan['schedule']) == 18, 'Benchmark scope differs')
    module = runpy.run_path(str(ROOT / 'scripts/summarize_pruning_benchmarks.py'), run_name='snapshot_import')
    runs = []
    for ordinal, item in enumerate(plan['schedule'], 1):
        method, c, rep = item['method'], item['concurrency'], item['repetition']
        relative = f'outputs/benchmark/pruning_v1/formal/{method}/isl1024_osl128_c{c}/rep{rep}'
        path = ORIGINAL / relative
        receipt = read(path / 'receipt.json')
        require(sha(path / 'receipt.json') == (path / 'receipt.json.sha256').read_text().split()[0], 'Receipt hash differs')
        require(receipt['completed_audit'] and receipt['pilot'] is False
                and receipt['protocol_sha256'] == PINS['data/benchmark/pruning_v1/protocol.json'], 'Invalid run receipt')
        for name, digest in receipt['file_sha256'].items():
            require(sha(path / name) == digest, 'Formal run file changed: ' + relative + '/' + name)
        manifest = read(path / 'run_manifest.json')
        require(all(manifest[k] == v and receipt[k] == v for k, v in item.items())
                and manifest['formal_schedule_ordinal'] == ordinal
                and manifest['requests'] == 200 and manifest['warmups'] == 20, 'Run schedule/scope mismatch')
        command = manifest['command']
        require(Path(command[0]).name == 'vllm'
                and command[1:] == module['expected_command'](plan, method, c, Path(REMOTE + relative)),
                'Benchmark command differs')
        require(sha(path / 'environment.txt') == plan['serve_environment']['freeze_sha256'], 'Serving environment differs')
        for name in ('server_before.log', 'server_after.log'):
            matches = [line.split('non-default args: ', 1)[1]
                       for line in (path / name).read_text(errors='replace').splitlines() if 'non-default args: ' in line]
            require(len(matches) == 1, 'Expected one server startup')
            args = ast.literal_eval(matches[0])
            require(args['model'] == plan['models'][method]['path']
                    or str(Path(REMOTE) / args['model']) == plan['models'][method]['path'], 'Server checkpoint differs')
            require(args['served_model_name'] == [plan['models'][method]['alias']], 'Server alias differs')
            require(all(args.get(k) == v for k, v in plan['server'].items()), 'Server settings differ')
        report = read(path / 'benchmark.json')
        require(report['completed'] == 200 and report['input_lens'] == [1024] * 200
                and report['output_lens'] == [128] * 200, 'Recorded token lengths/completions differ')
        if 'errors' in report:
            require(not any(report['errors']), 'Failed benchmark requests')
        samples = list(csv.reader((path / 'gpu_samples.csv').open()))
        require(len(samples) >= 2 and all(len(row) == 6 and row[1].strip() == plan['gpu_identity']['uuid'] for row in samples),
                'GPU sampling identity differs')
        memory = read(path / 'memory_summary.json')
        close(memory['sampled_max_used_mib'], max(float(row[2]) for row in samples), 'Memory maximum differs')
        close(memory['sampled_min_used_mib'], min(float(row[2]) for row in samples), 'Memory minimum differs')
        saved = summary['runs'][ordinal - 1]
        require(saved['metrics'] == {k: report[k] for k in module['METRICS']}
                and saved['memory'] == memory and saved['receipt_sha256'] == sha(path / 'receipt.json'), 'Embedded run differs')
        runs.append(saved)
    require(module['aggregate'](runs, METHODS) == summary['cases'], 'Recomputed performance aggregates differ')
    require(summary['primary_results'] == summary['cases'][0]['versus_bf16']['ActivationEnergy']
            and summary['measured_requests'] == 3600, 'Primary result/scope differs')
    for row in summary['checkpoint_sizes']:
        expected = 4022468096 if row['method'] == 'BF16' else 3880910336
        require(row['unique_parameters'] == expected and row['bf16_parameter_payload_bytes'] == 2 * expected,
                'Recorded size parameter arithmetic differs')
        close(row['safetensors_gib'], row['safetensors_file_bytes'] / 2**30, 'Size unit conversion differs')
    print('PASS: all 18 native benchmark records, 3600 requests, commands, memory samples and mean/SD/speed arithmetic.')


def verify_quality():
    summary = read(ROOT / 'results/quality/development_summary.json')
    require(summary == read(ORIGINAL / 'outputs/evaluation/pruning_dev_v1/development_summary.json'), 'Quality result copy differs')
    require(summary['protocol_sha256'] == PINS['data/evaluation/pruning_dev_v1/protocol.json'], 'Quality protocol differs')
    scores = summary['per_question_primary_scores']
    for row in summary['models']:
        method = row['method']
        for task, n, correct, percent in (('gsm8k', 256, 'gsm8k_correct', 'gsm8k_percent'),
                                         ('arc_challenge', 299, 'arc_correct', 'arc_percent')):
            values = scores[method][task]
            require(len(values) == n and all(x in (0, 1) for x in values) and sum(values) == row[correct], 'Quality score totals differ')
            close(100 * sum(values) / n, row[percent], 'Quality percentage differs')
        run = summary['runs'][method]
        relative = run['path'].removeprefix(REMOTE)
        require(relative != run['path'], 'Unexpected quality run root')
        path = ORIGINAL / relative
        require(sha(path / 'quality_summary.json') == run['quality_summary_sha256']
                and sha(path / 'run_manifest.json') == run['manifest_sha256'], 'Quality run summary hashes differ')
        manifest = read(path / 'run_manifest.json')
        require(manifest['method'] == method and manifest['scope'] == 'full'
                and manifest['protocol_sha256'] == summary['protocol_sha256'], 'Quality run scope differs')
    for pair in summary['paired_comparisons']:
        first, second = (scores[m][pair['task']] for m in (pair['first'], pair['second']))
        gains = sum(a == 1 and b == 0 for a, b in zip(first, second))
        losses = sum(a == 0 and b == 1 for a, b in zip(first, second))
        require((gains, losses) == (pair['gains'], pair['losses']), 'Paired discordant counts differ')
        close(100 * (gains - losses) / len(first), pair['delta_pp'], 'Paired score difference differs')
        n = gains + losses
        p = min(1., 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n)
        close(p, pair['exact_two_sided_mcnemar_p_unadjusted'], 'Exact McNemar p differs')
    print('PASS: three saved quality summaries, primary score vectors and paired differences/exact p-values.')
    print('Scope: raw quality responses and harness samples are omitted; extraction, likelihoods and bootstrap draws were not re-audited.')


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    inventory = read(ROOT / 'provenance/publication_sha256.json')
    for name, digest in inventory.items():
        require(sha(ROOT / name) == digest, 'Public file changed: ' + name)
    source = read(ROOT / 'provenance/review_scope.json')
    for name, digest in source['file_sha256'].items():
        require(sha(original_path(name)) == digest, 'Archived input changed: ' + name)
    for name, digest in PINS.items():
        require(sha(ORIGINAL / name) == digest, 'Frozen protocol changed')
    for p in (ROOT / 'scripts').glob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    print(f'PASS: {len(inventory)} public file hashes and all {len(source["file_sha256"])} original archived file hashes.')
    verify_importance_and_exports()
    verify_benchmarks()
    verify_quality()
    print('CPU snapshot verification complete. No network requests, CUDA execution, inference or benchmark reruns.')


if __name__ == '__main__':
    main()
