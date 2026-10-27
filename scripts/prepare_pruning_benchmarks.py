#!/usr/bin/env python3
"""Freeze fresh BF16 structured-pruning serving benchmarks; no inference."""
import sys
from pathlib import Path

from prepare_pruning_quality import ROOT, OUT, ALIASES, read, sha, put, gpu, serve_environment, require

QUALITY_SHA = '9ec3eb2d74fd9461d3de850a12fca8ac12b6e4eb15ecd48e72ea35b20d41b352'
BENCH = ROOT / 'data/benchmark/pruning_v1'
METHODS = ('BF16', 'RandomChannels', 'ActivationEnergy')


def quality_plan():
    path = OUT / 'protocol.json'
    require(sha(path) == QUALITY_SHA, 'Frozen pruning quality protocol changed')
    plan = read(path)
    require(plan['methods'] == list(METHODS), 'Expected the fixed three-model comparison')
    return plan


def verify_benchmark_plan():
    plan = read(BENCH / 'protocol.json')
    require(plan['study'] == 'pruning_serving_v1' and plan['quality_protocol_sha256'] == QUALITY_SHA,
            'Benchmark study identity differs')
    require(sha(OUT / 'protocol.json') == QUALITY_SHA, 'Quality protocol changed')
    for path, digest in plan['file_sha256'].items():
        require(sha(path) == digest, 'Frozen benchmark/checkpoint file changed: ' + path)
    require(gpu() == plan['gpu_identity'], 'GPU/driver changed')
    require(serve_environment() == plan['serve_environment'], 'Serving environment changed')
    require(sys.version.split()[0] == '3.12.15', 'Expected Python 3.12.15')
    return plan


def main():
    quality = quality_plan()
    require(sys.version.split()[0] == '3.12.15', 'Expected Python 3.12.15')
    require(serve_environment() == quality['serve_environment'], 'Use the pinned .venv-serve environment')
    require(gpu() == quality['gpu_identity'], 'GPU differs from the pruning study')
    print('Verifying frozen source and all three checkpoints (read-only)...', flush=True)
    files = dict(quality['file_sha256'])
    for path, digest in files.items():
        require(sha(path) == digest, 'Pruning study input changed: ' + path)
    files[str(OUT / 'protocol.json')] = QUALITY_SHA
    for name in ('prepare_pruning_benchmarks.py', 'run_pruning_benchmark.py'):
        path = ROOT / 'scripts' / name
        files[str(path)] = sha(path)
    schedule = []
    # Latin-square model order; each method appears once in each position.
    orders = (('BF16', 'RandomChannels', 'ActivationEnergy'),
              ('RandomChannels', 'ActivationEnergy', 'BF16'),
              ('ActivationEnergy', 'BF16', 'RandomChannels'))
    for repetition, methods in enumerate(orders, 1):
        for method in methods:
            for concurrency in ((4, 1) if repetition != 2 else (1, 4)):
                schedule.append({'method': method, 'repetition': repetition, 'concurrency': concurrency})
    plan = {'study': 'pruning_serving_v1', 'quality_protocol_sha256': QUALITY_SHA,
        'methods': list(METHODS), 'models': quality['models'], 'server': quality['server'],
        'gpu_identity': gpu(), 'serve_environment': quality['serve_environment'],
        'file_sha256': files, 'input_tokens': 1024, 'output_tokens': 128,
        'requests': 200, 'warmup_requests': 20, 'repetitions': 3,
        'concurrencies': [4, 1], 'pilot_requests': 8, 'pilot_warmup_requests': 4,
        'workload_seed': 42, 'generation_seed': 42, 'temperature': 0, 'top_p': 1,
        'ignore_eos': True, 'request_rate': 'inf', 'backend': 'openai',
        'endpoint': '/v1/completions', 'streaming': True, 'tokenizer': quality['models']['BF16']['path'],
        'primary_workload': {'input_tokens': 1024, 'output_tokens': 128, 'concurrency': 4},
        'primary_comparison': 'ActivationEnergy versus original BF16; mean run-level output token throughput',
        'secondary': ['RandomChannels versus BF16', 'all three models at concurrency 1',
                      'TTFT, TPOT, end-to-end latency, sampled whole-GPU memory'],
        'schedule': schedule, 'schedule_policy': 'Formal run receipts must follow this order; pilots excluded.',
        'memory_sample_interval_seconds': 1,
        'memory_scope': 'Whole GPU while benchmark client runs, including warmup and initial probe; excludes startup and can miss brief peaks.',
        'token_length_scope': 'Lengths recorded by the native vLLM benchmark client; no per-request forwarding proxy is inserted.',
        'repetition_scope': 'Repeated benchmark runs, not three independent machine or model-seed samples. A server may stay up between consecutive runs of the same model.',
        'timing_scope': 'Warm local HTTP streaming service; model startup and compilation excluded; client-side timing includes HTTP/client overhead.',
        'comparison_scope': 'One pruning width, one random channel seed, one GPU, two fixed workloads; no general speed guarantee.',
        'prior_results': 'Fresh measurements for all three models. Project 1 timings are not used as the baseline.',
        'statistical_plan': 'For each method/workload report all three run values, arithmetic mean and sample SD. Report ratios of mean throughput and differences of mean sampled maxima. Three repeats are descriptive; do not claim statistical significance.',
        'preparation_inference_requests': 0}
    put(BENCH / 'protocol.json', plan)
    print('Models: BF16, RandomChannels, ActivationEnergy; all BF16, no quantization')
    print('Workload: 1024 input / 128 output tokens; concurrency 4 (primary) and 1')
    print('Formal scope: 18 runs, 3600 measured requests; 20 warmups per run')
    print('KV cache: fixed 4 GiB BF16; context 8192; FlashAttention 2; prefix cache disabled')
    print('Protocol SHA256:', sha(BENCH / 'protocol.json'))
    print('Saved:', BENCH / 'protocol.json')
    print('Formal run order:')
    for ordinal, item in enumerate(schedule, 1):
        print(f"  {ordinal:2}: {item['method']:<17} rep={item['repetition']} concurrency={item['concurrency']}")
    print('PASS: shared benchmark settings, order, checkpoint hashes and environment frozen.')
    print('No inference or benchmark timing performed.')


if __name__ == '__main__':
    main()
