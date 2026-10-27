#!/usr/bin/env python3
"""Audit and summarize the 18 saved pruning benchmarks; no model requests.

Existing runs and frozen protocols are read-only. Generates JSON and Markdown
with all run values, arithmetic means and sample SDs, not significance claims.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

PROTOCOL_SHA = '947676903d7210320e4b7e809d75c3474ddc8a212c6f1b04cb7f39ab36ff91c8'
METRICS = ('output_throughput', 'mean_ttft_ms', 'mean_tpot_ms', 'mean_e2el_ms',
           'p95_ttft_ms', 'p95_tpot_ms', 'p95_e2el_ms')
MEMORY_METRICS = ('sampled_min_used_mib', 'sampled_max_used_mib',
                  'sampled_min_temperature_c', 'sampled_max_temperature_c', 'sampled_max_power_w')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def describe(values):
    require(len(values) == 3 and all(math.isfinite(x) for x in values),
            'Expected three finite run-level values')
    return {'values': list(values), 'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values), 'minimum': min(values), 'maximum': max(values)}


def put_once(path, payload):
    if path.exists():
        require(path.read_bytes() == payload, 'Existing summary differs; preserve it: ' + str(path))
    else:
        with path.open('xb') as handle:
            handle.write(payload)


def expected_command(plan, method, concurrency, out):
    alias = plan['models'][method]['alias']
    return ['bench', 'serve', '--backend', 'openai', '--base-url', 'http://127.0.0.1:8000',
        '--endpoint', '/v1/completions', '--model', alias, '--served-model-name', alias,
        '--tokenizer', plan['tokenizer'], '--dataset-name', 'random',
        '--random-input-len', '1024', '--random-output-len', '128', '--random-range-ratio', '0',
        '--random-prefix-len', '0', '--num-prompts', '200', '--num-warmups', '20',
        '--max-concurrency', str(concurrency), '--request-rate', 'inf', '--ignore-eos',
        '--temperature', '0', '--top-p', '1', '--seed', '42',
        '--extra-body', '{"seed":42,"add_special_tokens":false}',
        '--percentile-metrics', 'ttft,tpot,e2el', '--metric-percentiles', '50,95',
        '--save-result', '--save-detailed', '--result-dir', str(out), '--result-filename', 'benchmark.json']


def audit_run(plan, item, ordinal, path, helpers):
    read, sha = helpers['read'], helpers['sha']
    method, concurrency, repetition = item['method'], item['concurrency'], item['repetition']
    receipt = helpers['completed_receipt'](path, PROTOCOL_SHA)
    require(receipt['pilot'] is False and receipt['client_recorded_request_lengths_verified'] is True,
            'Pilot or incomplete receipt in formal runs')
    require(all(receipt[k] == v for k, v in item.items()), 'Receipt cell identity differs')
    require(receipt['checkpoint'] == plan['models'][method]
            and receipt['gpu_identity'] == plan['gpu_identity'], 'Receipt checkpoint/GPU differs')
    manifest = read(path / 'run_manifest.json')
    require(all(manifest[k] == v for k, v in item.items()) and manifest['pilot'] is False
            and manifest['formal_schedule_ordinal'] == ordinal
            and manifest['protocol_sha256'] == PROTOCOL_SHA
            and manifest['requests'] == 200 and manifest['warmups'] == 20,
            'Manifest identity, scope or schedule ordinal differs')
    require(manifest['checkpoint'] == plan['models'][method]
            and manifest['gpu_identity'] == plan['gpu_identity'], 'Manifest checkpoint/GPU differs')
    command = manifest['command']
    require(Path(command[0]).name == 'vllm' and command[1:] == expected_command(plan, method, concurrency, path),
            'Native benchmark CLI settings differ from frozen protocol')
    require(sha(path / 'environment.txt') == plan['serve_environment']['freeze_sha256'],
            'Run environment freeze differs')
    for name in ('server_before.log', 'server_after.log'):
        helpers['check_server_log'](path / name, method, plan)
    cards = read(path / 'models.json')['data']
    require(len(cards) == 1 and cards[0]['id'] == plan['models'][method]['alias']
            and cards[0]['max_model_len'] == 8192
            and Path(cards[0]['root']).resolve() == Path(plan['models'][method]['path']).resolve(),
            'Saved model card differs')
    report = read(path / 'benchmark.json')
    helpers['validate_report'](report, 200, concurrency)
    if 'failed' in report:
        require(report['failed'] == 0, 'Failed requests in formal run')
    for key, expected in (('total_input_tokens', 204800), ('total_output_tokens', 25600)):
        if key in report:
            require(report[key] == expected, 'Reported token total differs')
    if 'duration' in report:
        require(math.isfinite(report['duration']) and report['duration'] > 0, 'Invalid benchmark duration')
        require(math.isclose(report['output_throughput'], 25600 / report['duration'], rel_tol=1e-8),
                'Output token throughput differs from duration/token totals')
    memory = helpers['memory_summary'](path / 'gpu_samples.csv', plan['gpu_identity'])
    require(memory == read(path / 'memory_summary.json'), 'Saved memory summary differs from GPU CSV')
    return {'method': method, 'concurrency': concurrency, 'repetition': repetition,
        'schedule_ordinal': ordinal, 'path': str(path), 'requests': 200,
        'receipt_sha256': sha(path / 'receipt.json'), 'file_sha256': receipt['file_sha256'],
        'checkpoint': plan['models'][method], 'gpu_identity': plan['gpu_identity'],
        'engine_pid': manifest['engine_pid'],
        'metrics': {key: report[key] for key in METRICS},
        'memory': memory}


def aggregate(runs, methods):
    cells = []
    for concurrency in (4, 1):
        method_results = {}
        for method in methods:
            group = sorted([r for r in runs if r['method'] == method and r['concurrency'] == concurrency],
                           key=lambda r: r['repetition'])
            require([r['repetition'] for r in group] == [1, 2, 3], 'Incomplete or duplicate repetition cell')
            result = {key: describe([r['metrics'][key] for r in group]) for key in METRICS}
            result.update({key: describe([r['memory'][key] for r in group]) for key in MEMORY_METRICS})
            method_results[method] = result
        baseline = method_results['BF16']
        comparisons = {}
        for method in methods:
            if method == 'BF16':
                continue
            current = method_results[method]
            comparisons[method] = {
                'output_throughput_ratio_of_means': current['output_throughput']['mean'] / baseline['output_throughput']['mean'],
                'output_throughput_increase_percent': 100 * (current['output_throughput']['mean'] / baseline['output_throughput']['mean'] - 1),
                'latency_reduction_percent': {key: 100 * (1 - current[key]['mean'] / baseline[key]['mean'])
                    for key in ('mean_ttft_ms', 'mean_tpot_ms', 'mean_e2el_ms')},
                'sampled_memory_mean_of_run_maxima_saved_mib': baseline['sampled_max_used_mib']['mean'] - current['sampled_max_used_mib']['mean'],
                'sampled_memory_mean_of_run_maxima_reduction_percent': 100 * (1 - current['sampled_max_used_mib']['mean'] / baseline['sampled_max_used_mib']['mean']),
                'per_repetition_throughput_ratios': [a / b for a, b in zip(current['output_throughput']['values'], baseline['output_throughput']['values'])]}
        cells.append({'input_tokens': 1024, 'output_tokens': 128, 'concurrency': concurrency,
                      'methods': method_results, 'versus_bf16': comparisons})
    return cells


def checkpoint_sizes(plan, read, sha):
    rows = []
    for method in plan['methods']:
        model = Path(plan['models'][method]['path'])
        files = sorted(model.glob('*.safetensors'))
        require(bool(files), 'No checkpoint shards: ' + str(model))
        # Every shard must already be an input pinned by the frozen protocol.
        for path in files:
            require(str(path) in plan['file_sha256'], 'Unexpected unpinned weight file: ' + str(path))
        if method == 'BF16':
            pruning = read(Path('/root/autodl-tmp/qwen-pruning-lab/data/protocol_v1.json'))
            parameters = pruning['source_serialized_tensor_elements']
            require(parameters == 4022468096, 'Original parameter count differs')
        else:
            manifest = read(model / 'pruning_manifest.json')
            require(sha(model / 'pruning_manifest.json') == plan['models'][method]['pruning_manifest_sha256'],
                    'Pruned manifest identity differs')
            parameters = manifest['pruned_unique_parameters']
            require(parameters == 3880910336, 'Pruned parameter count differs')
        config = read(model / 'config.json')
        width = 9728 if method == 'BF16' else 9216
        require(config['intermediate_size'] == width and config['tie_word_embeddings'] is True,
                'Model width or tied embeddings differ')
        rows.append({'method': method, 'unique_parameters': parameters, 'mlp_width': width,
                     'safetensors_files': len(files), 'safetensors_file_bytes': sum(p.stat().st_size for p in files),
                     'bf16_parameter_payload_bytes': 2 * parameters})
    original = rows[0]
    for row in rows:
        row['parameter_reduction_percent'] = 100 * (1 - row['unique_parameters'] / original['unique_parameters'])
        row['safetensors_file_reduction_mib'] = (original['safetensors_file_bytes'] - row['safetensors_file_bytes']) / 2**20
        row['safetensors_file_reduction_percent'] = 100 * (1 - row['safetensors_file_bytes'] / original['safetensors_file_bytes'])
        row['safetensors_gib'] = row['safetensors_file_bytes'] / 2**30
    return rows


def markdown(summary):
    lines = ['# Structured MLP pruning: serving benchmark', '',
        'Fresh measurements of original BF16, RandomChannels and ActivationEnergy on one RTX 4090. '
        'All models use BF16 weights and BF16 KV cache; no FP8 or recovery training.', '',
        'The workload has 1,024 client-recorded input tokens and 128 output tokens. '
        'Each cell has three runs of 200 measured requests after 20 warmup requests. '
        'Concurrency 4 is primary; concurrency 1 is secondary. Pilots are excluded.', '',
        '## Mean performance across three runs', '',
        '| Concurrency | Model | Output tok/s, mean ± SD | Speed ratio vs BF16 | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2E (ms) | Mean sampled run maximum (MiB) |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for cell in summary['cases']:
        for method in summary['methods']:
            values = cell['methods'][method]
            throughput = values['output_throughput']
            ratio = 1.0 if method == 'BF16' else cell['versus_bf16'][method]['output_throughput_ratio_of_means']
            lines.append(f"| {cell['concurrency']} | {method} | {throughput['mean']:.2f} ± {throughput['sample_sd']:.2f} | {ratio:.4f}x | "
                f"{values['mean_ttft_ms']['mean']:.2f} | {values['mean_tpot_ms']['mean']:.2f} | {values['mean_e2el_ms']['mean']:.2f} | "
                f"{values['sampled_max_used_mib']['mean']:.2f} |")
    lines += ['', '## Checkpoint size', '',
        '| Model | MLP width | Unique parameters | Safetensors GiB | File saving vs BF16 (MiB) |',
        '|---|---:|---:|---:|---:|']
    for row in summary['checkpoint_sizes']:
        lines.append(f"| {row['method']} | {row['mlp_width']:,} | {row['unique_parameters']:,} | "
                     f"{row['safetensors_gib']:.4f} | {row['safetensors_file_reduction_mib']:.3f} |")
    lines += ['', '## All formal runs', '',
        '| Order | Model | Rep | Concurrency | Output tok/s | TTFT (ms) | TPOT (ms) | E2E (ms) | Sampled maximum (MiB) |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for run in summary['runs']:
        m = run['metrics']
        lines.append(f"| {run['schedule_ordinal']} | {run['method']} | {run['repetition']} | {run['concurrency']} | "
                     f"{m['output_throughput']:.2f} | {m['mean_ttft_ms']:.2f} | {m['mean_tpot_ms']:.2f} | "
                     f"{m['mean_e2el_ms']:.2f} | {run['memory']['sampled_max_used_mib']:.0f} |")
    lines += ['', '## Scope and limits', '',
        '- The server uses a fixed 4 GiB BF16 KV-cache budget, FlashAttention 2, context length 8,192, '
        'maximum four sequences, token batch limit 4,096 and disabled prefix caching.',
        '- Mean and sample SD describe three repeated runs, not a confidence interval or a significance test. '
        'Throughput speed ratios divide arithmetic means. Model order rotates across repetitions.',
        '- The JSON also summarizes run-level P95 latencies. Their mean is not a pooled P95 over all requests.',
        '- These are warm local HTTP streaming measurements. Startup, model loading and compilation '
        'are outside the measured workload; client and HTTP overhead are included.',
        '- Token lengths come from the native benchmark client. No latency-altering forwarding recorder '
        'was inserted. This audit does not independently inspect every wire request.',
        '- GPU memory is whole-device usage sampled every second while the benchmark client runs, '
        'including warmup and the initial probe. Brief peaks can be missed. The reported cell value is '
        'the mean of three sampled run maxima, not model-weight memory or a guaranteed peak.',
        '- Allocation retained from a preceding workload can affect memory. Concurrency 1 follows '
        'concurrency 4 in repetitions 1 and 3, and runs first in repetition 2, equally for each model. '
        'Read the individual values and ranges, not only the mean.',
        '- One GPU, one pruning width and one random-channel seed were studied. Performance results '
        'do not establish model quality or generalize to other workloads automatically.',
        '- Quality uses the separate, previously used development sets; its scores and uncertainty '
        'are in the development report. No official-test inference was added here.', '',
        'The JSON contains all run-level metrics, file hashes, aggregation inputs and size calculations.', '']
    return '\n'.join(lines)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from prepare_pruning_benchmarks import ROOT, BENCH, METHODS, verify_benchmark_plan
    from prepare_pruning_quality import read, sha
    from run_pruning_benchmark import completed_receipt, run_path, validate_report, memory_summary
    from run_pruning_quality import check_server_log
    require(sha(BENCH / 'protocol.json') == PROTOCOL_SHA, 'Expected the recorded frozen benchmark protocol')
    print('Verifying frozen checkpoint files and all 18 saved formal runs (read-only)...', flush=True)
    plan = verify_benchmark_plan()
    require(plan['methods'] == list(METHODS) and len(plan['schedule']) == 18
            and plan['requests'] == 200 and plan['warmup_requests'] == 20,
            'Benchmark scope differs')
    helpers = {'read': read, 'sha': sha, 'completed_receipt': completed_receipt,
               'check_server_log': check_server_log, 'validate_report': validate_report,
               'memory_summary': memory_summary}
    runs = []
    for ordinal, item in enumerate(plan['schedule'], 1):
        runs.append(audit_run(plan, item, ordinal, run_path(item), helpers))
        print(f"Verified {ordinal:2}/18: {item['method']} rep={item['repetition']} concurrency={item['concurrency']}", flush=True)
    require(len({(r['method'], r['repetition'], r['concurrency']) for r in runs}) == 18,
            'Formal run cells are missing or duplicated')
    cells = aggregate(runs, METHODS)
    sizes = checkpoint_sizes(plan, read, sha)
    primary = cells[0]['versus_bf16']['ActivationEnergy']
    summary = {'study': plan['study'], 'methods': list(METHODS), 'protocol_sha256': PROTOCOL_SHA,
        'summarizer_sha256': sha(Path(__file__).resolve()), 'gpu_identity': plan['gpu_identity'],
        'serve_environment': plan['serve_environment'], 'verified_formal_runs': 18,
        'measured_requests': sum(r['requests'] for r in runs), 'warmup_requests': 360,
        'pilot_runs_included': False,
        'aggregation': 'Arithmetic mean and sample SD of three run-level metrics in each cell; ratio of mean throughputs.',
        'primary_comparison': 'ActivationEnergy versus BF16 at input 1024 / output 128 / concurrency 4',
        'primary_results': primary, 'cases': cells, 'checkpoint_sizes': sizes, 'runs': runs,
        'limitations': {key: plan[key] for key in ('memory_scope', 'token_length_scope', 'repetition_scope',
                        'timing_scope', 'comparison_scope', 'statistical_plan')},
        'new_model_requests': 0, 'new_cuda_execution': False}
    output = ROOT / 'outputs/benchmark/pruning_v1'
    put_once(output / 'benchmark_summary.json', (json.dumps(summary, indent=2, ensure_ascii=False) + '\n').encode())
    put_once(output / 'benchmark_report.md', markdown(summary).encode())
    print('\nC  Model               Mean tok/s   SD tok/s   Speed vs BF16   Mean sampled max MiB')
    for cell in cells:
        for method in METHODS:
            row = cell['methods'][method]
            rate = row['output_throughput']
            ratio = 1.0 if method == 'BF16' else cell['versus_bf16'][method]['output_throughput_ratio_of_means']
            print(f"{cell['concurrency']}  {method:<18} {rate['mean']:10.2f} {rate['sample_sd']:10.3f} {ratio:14.4f}x "
                  f"{row['sampled_max_used_mib']['mean']:22.2f}")
    print(f"\nPrimary ActivationEnergy throughput increase: {primary['output_throughput_increase_percent']:.3f}%")
    print(f"Primary mean sampled run-maximum memory saving: {primary['sampled_memory_mean_of_run_maxima_saved_mib']:.2f} MiB")
    print('Saved:', output / 'benchmark_summary.json')
    print('Saved:', output / 'benchmark_report.md')
    print('PASS: 18 formal runs, 3600 measured requests, recorded token lengths, source/environment/GPU evidence and memory CSVs verified.')
    print('No new inference or CUDA execution; three-repeat descriptive comparison only.')


if __name__ == '__main__':
    main()
