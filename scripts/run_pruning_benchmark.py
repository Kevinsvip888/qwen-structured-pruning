#!/usr/bin/env python3
"""Run one recorded native-vLLM pruning benchmark without stopping the server."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import ProxyHandler, build_opener

from prepare_pruning_benchmarks import ROOT, BENCH, METHODS, verify_benchmark_plan
from prepare_pruning_quality import read, sha, gpu, serve_environment, require
from run_pruning_quality import check_server_log


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def run_path(item):
    return (ROOT / 'outputs/benchmark/pruning_v1/formal' / item['method'] /
            f"isl1024_osl128_c{item['concurrency']}" / f"rep{item['repetition']}")


def completed_receipt(path, protocol_sha):
    receipt_path = path / 'receipt.json'
    receipt = read(receipt_path)
    require(receipt['protocol_sha256'] == protocol_sha and receipt['completed_audit'] is True,
            'Earlier formal run has no matching completed audit: ' + str(path))
    require(sha(receipt_path) == (path / 'receipt.json.sha256').read_text().split()[0],
            'Earlier receipt changed')
    actual = {str(p.relative_to(path)): sha(p) for p in path.rglob('*')
              if p.is_file() and p.name not in ('receipt.json', 'receipt.json.sha256')}
    require(actual == receipt['file_sha256'], 'Earlier formal run files changed: ' + str(path))
    return receipt


def check_schedule(plan, item, protocol_sha):
    require(item in plan['schedule'], 'Run is outside the frozen schedule')
    index = plan['schedule'].index(item)
    for earlier in plan['schedule'][:index]:
        path = run_path(earlier)
        require((path / 'receipt.json').is_file(),
                f"Next required formal run: {earlier['method']} rep={earlier['repetition']} concurrency={earlier['concurrency']}")
        receipt = completed_receipt(path, protocol_sha)
        require(all(receipt[key] == value for key, value in earlier.items()),
                'Earlier receipt method/repetition/concurrency differs')
    return index + 1


def server_status(plan, method):
    opener = build_opener(ProxyHandler({}))
    with opener.open('http://127.0.0.1:8000/health', timeout=10) as response:
        require(response.status == 200, 'Server is not healthy')
    with opener.open('http://127.0.0.1:8000/v1/models', timeout=10) as response:
        models = json.load(response)
    cards = models['data']
    require(len(cards) == 1 and cards[0]['id'] == plan['models'][method]['alias']
            and cards[0]['max_model_len'] == 8192
            and Path(cards[0]['root']).resolve() == Path(plan['models'][method]['path']).resolve(),
            'Wrong running model, checkpoint path or context limit')
    return models


def sole_gpu_pid():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True)
    pids = [line.strip() for line in output.splitlines() if line.strip()]
    require(len(pids) == 1 and pids[0].isdigit(),
            'Expected only the vLLM engine on the GPU; stop unrelated GPU jobs first')
    return int(pids[0])


def validate_report(report, requests, concurrency):
    require(report['completed'] == requests, 'Not all requests completed')
    require(report['input_lens'] == [1024] * requests, 'Client-recorded input token lengths differ')
    require(report['output_lens'] == [128] * requests, 'Recorded output token lengths differ')
    for key in ('output_throughput', 'mean_ttft_ms', 'mean_tpot_ms', 'mean_e2el_ms',
                'p95_ttft_ms', 'p95_tpot_ms', 'p95_e2el_ms'):
        value = report[key]
        require(isinstance(value, (int, float)) and math.isfinite(value) and value > 0,
                'Invalid benchmark metric: ' + key)
    require(report.get('max_concurrency', concurrency) == concurrency, 'Reported concurrency differs')
    if 'errors' in report:
        require(not any(report['errors']), 'Benchmark recorded request errors')


def memory_summary(path, identity):
    samples = []
    with path.open() as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            require(len(row) == 6 and row[1].strip() == identity['uuid'], 'GPU sample row/UUID differs')
            memory, utilization, temperature, power = (float(x.strip()) for x in row[2:])
            require(all(math.isfinite(x) for x in (memory, utilization, temperature, power))
                    and memory >= 0, 'Invalid GPU sample')
            samples.append({'used_mib': memory, 'utilization_percent': utilization,
                            'temperature_c': temperature, 'power_w': power})
    require(len(samples) >= 2, 'Too few GPU samples')
    return {'sample_interval_seconds': 1, 'samples': len(samples),
            'scope': 'whole GPU during benchmark client, including warmup/probe; excludes server startup',
            'limitation': 'One-second sampling can miss brief memory peaks; not allocator-level model memory.',
            'sampled_min_used_mib': min(x['used_mib'] for x in samples),
            'sampled_max_used_mib': max(x['used_mib'] for x in samples),
            'sampled_min_temperature_c': min(x['temperature_c'] for x in samples),
            'sampled_max_temperature_c': max(x['temperature_c'] for x in samples),
            'sampled_max_power_w': max(x['power_w'] for x in samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--concurrency', type=int, choices=(1, 4), default=4)
    parser.add_argument('--repetition', type=int, choices=(1, 2, 3))
    parser.add_argument('--pilot', action='store_true', help='Eight measured requests; excluded from formal results')
    parser.add_argument('--server-log', type=Path)
    args = parser.parse_args()
    require(args.pilot != (args.repetition is not None), 'Specify either --pilot or --repetition 1/2/3')
    print('Verifying frozen benchmark protocol and checkpoint hashes (read-only)...', flush=True)
    plan = verify_benchmark_plan()
    protocol_sha = sha(BENCH / 'protocol.json')
    item = {'method': args.method, 'repetition': args.repetition, 'concurrency': args.concurrency}
    if args.pilot:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        out = ROOT / 'outputs/benchmark/pruning_v1/pilot' / args.method / stamp
        requests, warmups = plan['pilot_requests'], plan['pilot_warmup_requests']
        ordinal = None
    else:
        ordinal = check_schedule(plan, item, protocol_sha)
        out = run_path(item)
        requests, warmups = plan['requests'], plan['warmup_requests']
    require(not out.exists(), 'Run directory already exists; preserve it: ' + str(out))
    models = server_status(plan, args.method)
    log_path = args.server_log or Path('logs') / (args.method + '_server.log')
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    check_server_log(log_path, args.method, plan)
    engine_pid = sole_gpu_pid()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'models.json', models)
    shutil.copy2(log_path, out / 'server_before.log')
    (out / 'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    frozen_env = subprocess.check_output(['uv', 'pip', 'freeze', '--python', sys.executable],
                                       text=True, stderr=subprocess.PIPE)
    (out / 'environment.txt').write_text(frozen_env)
    require(sha(out / 'environment.txt') == plan['serve_environment']['freeze_sha256'],
            'Active environment differs from the pinned serving environment; use .venv-serve')
    alias = plan['models'][args.method]['alias']
    command = [str(Path(sys.executable).parent / 'vllm'), 'bench', 'serve', '--backend', 'openai',
        '--base-url', 'http://127.0.0.1:8000', '--endpoint', '/v1/completions',
        '--model', alias, '--served-model-name', alias, '--tokenizer', plan['tokenizer'],
        '--dataset-name', 'random', '--random-input-len', '1024', '--random-output-len', '128',
        '--random-range-ratio', '0', '--random-prefix-len', '0', '--num-prompts', str(requests),
        '--num-warmups', str(warmups), '--max-concurrency', str(args.concurrency), '--request-rate', 'inf',
        '--ignore-eos', '--temperature', '0', '--top-p', '1', '--seed', '42',
        '--extra-body', '{"seed":42,"add_special_tokens":false}',
        '--percentile-metrics', 'ttft,tpot,e2el', '--metric-percentiles', '50,95',
        '--save-result', '--save-detailed', '--result-dir', str(out), '--result-filename', 'benchmark.json']
    require(Path(command[0]).is_file(), 'vLLM CLI missing from the active environment')
    environment = os.environ.copy()
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        environment.pop(name, None)
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost', PYTHONUNBUFFERED='1')
    save(out / 'run_manifest.json', {'study': plan['study'], 'pilot': args.pilot,
        'method': args.method, 'repetition': args.repetition, 'concurrency': args.concurrency,
        'formal_schedule_ordinal': ordinal, 'protocol_sha256': protocol_sha,
        'checkpoint': plan['models'][args.method], 'gpu_identity': plan['gpu_identity'],
        'engine_pid': engine_pid, 'requests': requests, 'warmups': warmups, 'command': command})
    monitor, process = None, None
    try:
        with (out / 'gpu_samples.csv').open('w') as samples, (out / 'gpu_monitor.log').open('w') as errors:
            monitor = subprocess.Popen(['nvidia-smi', '--id=0',
                '--query-gpu=timestamp,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits', '--loop-ms=1000'], stdout=samples, stderr=errors)
            with (out / 'client.log').open('wb') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                while chunk := process.stdout.read1(4096):
                    log.write(chunk); log.flush()
                    sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
                returncode = process.wait()
            require(monitor.poll() is None, 'GPU monitor exited before the benchmark completed')
        require(returncode == 0, 'Benchmark failed; partial files preserved: ' + str(out))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        if monitor is not None and monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill(); monitor.wait()
        shutil.copy2(log_path, out / 'server_after.log')
    (out / 'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    require(sole_gpu_pid() == engine_pid, 'Server GPU process changed during the benchmark')
    require(gpu() == plan['gpu_identity'] and serve_environment() == plan['serve_environment'],
            'GPU/driver or serving environment changed during the run')
    server_status(plan, args.method)
    check_server_log(out / 'server_after.log', args.method, plan)
    report = read(out / 'benchmark.json')
    validate_report(report, requests, args.concurrency)
    memory = memory_summary(out / 'gpu_samples.csv', plan['gpu_identity'])
    save(out / 'memory_summary.json', memory)
    for path, digest in plan['file_sha256'].items():
        if '/scripts/' in path:
            require(sha(path) == digest, 'Frozen script changed during the run')
    receipt = {'study': plan['study'], 'completed_audit': True, 'pilot': args.pilot,
        'method': args.method, 'repetition': args.repetition, 'concurrency': args.concurrency,
        'protocol_sha256': protocol_sha, 'checkpoint': plan['models'][args.method],
        'gpu_identity': plan['gpu_identity'], 'client_recorded_request_lengths_verified': True,
        'file_sha256': {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()},
        'note': 'Native-client warm service measurement. Token lengths are client-recorded; sampled memory includes warmup and can miss peaks.'}
    save(out / 'receipt.json', receipt)
    (out / 'receipt.json.sha256').write_text(sha(out / 'receipt.json') + '  receipt.json\n')
    print(f'\nPASS: {requests} successful requests; recorded input/output lengths 1024/128; server/protocol verified.')
    print('Sampled maximum whole-GPU memory:', memory['sampled_max_used_mib'], 'MiB')
    print('Saved:', out)
    if args.pilot:
        print('Pilot only; excluded from formal benchmark results.')
    elif ordinal < len(plan['schedule']):
        following = plan['schedule'][ordinal]
        print(f"Next formal run: {following['method']} rep={following['repetition']} concurrency={following['concurrency']}")


if __name__ == '__main__':
    main()
