#!/usr/bin/env python3
"""Recorded pruning development evaluation; never modifies or stops vLLM."""
import argparse
import ast
import copy
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from transformers import AutoTokenizer
from prepare_pruning_quality import ROOT, OLD, OUT, PREPARED, TASKS, COUNTS, ALIASES
from prepare_pruning_quality import require, read, sha, verify_protocol
from pruning_eval_recording import Recorder, audit_saved, http_json, save_json


def check_server_log(path, method, plan):
    lines = path.read_text(errors='replace').splitlines()
    matches = [line.split('non-default args: ', 1)[1] for line in lines if 'non-default args: ' in line]
    require(len(matches) == 1, 'Expected one server startup in the log; use a fresh log')
    settings = ast.literal_eval(matches[0])
    model = settings.get('model', settings.get('model_tag'))
    require(model and Path(model).resolve() == Path(plan['models'][method]['path']).resolve(),
            'Server log checkpoint differs from the selected method')
    require(settings.get('served_model_name') == [ALIASES[method]], 'Server alias differs')
    for key, value in plan['server'].items():
        require(settings.get(key) == value, 'Server startup setting differs: ' + key)
    require(settings.get('quantization') in (None, ''), 'Expected BF16 without quantization')
    require(any('Using FlashAttention version 2' in line for line in lines),
            'Expected FlashAttention version 2 startup evidence')


def expected_requests(tokenizer, counts):
    expected = defaultdict(deque)
    for task in TASKS:
        for line in (PREPARED / (task + '_requests.jsonl')).read_text().splitlines():
            entry = json.loads(line)
            if entry['doc_id'] >= counts[task]:
                continue
            context = tokenizer.encode(entry['context'], add_special_tokens=False)
            tokens = tokenizer.encode(entry['context'] + (entry.get('continuation') or ''),
                                      add_special_tokens=False)
            require(tokens[:len(context)] == context, 'Tokenizer continuation boundary changed')
            kind = 'math' if task == TASKS[0] else 'arc'
            require(len(tokens) + (2048 if kind == 'math' else 1) <= 8192, 'Context budget exceeded')
            metadata = {'task': task, 'doc_id': entry['doc_id'], 'source_id': entry['source_id'],
                        'kind': kind, 'context_tokens': len(context), 'prompt_tokens': len(tokens)}
            if kind == 'arc':
                metadata['choice_index'] = entry['choice_index']
            expected[(kind, tuple(tokens))].append(metadata)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=tuple(ALIASES))
    parser.add_argument('--limit', type=int, choices=(8, 100),
                        help='Pilot only; omit for all 256 math and 299 science questions')
    parser.add_argument('--audit-only', type=Path, help='Verify saved responses without server/inference')
    parser.add_argument('--server-log', type=Path)
    args = parser.parse_args()
    print('Verifying frozen protocol and checkpoint hashes (read-only)...', flush=True)
    plan = verify_protocol()
    counts = {task: min(args.limit, count) if args.limit else count for task, count in COUNTS.items()}
    alias = ALIASES[args.method]
    saved = None
    if args.audit_only:
        saved = args.audit_only.resolve()
        manifest = read(saved / 'run_manifest.json')
        require(manifest['method'] == args.method and manifest['protocol_sha256'] == sha(OUT / 'protocol.json'),
                'Saved run identity differs')
        counts = manifest['question_counts']
        require(counts in (COUNTS, {t: 8 for t in TASKS}, {t: 100 for t in TASKS}), 'Saved scope differs')
    tokenizer = AutoTokenizer.from_pretrained(plan['models']['BF16']['path'],
                                              local_files_only=True, trust_remote_code=False)
    expected = expected_requests(tokenizer, counts)
    total = sum(len(x) for x in expected.values())
    expected_for_audit = copy.deepcopy(expected)
    if saved:
        require(manifest['expected_http_requests'] == total, 'Saved request count differs')
        for relative, digest in manifest['runner_files_sha256'].items():
            require(sha(ROOT / relative) == digest, 'Saved run source differs')
        check_server_log(saved / 'server_before.log', args.method, plan)
        previous_path = saved / 'quality_summary.json'
        previous_wall = None
        if previous_path.exists():
            previous = read(previous_path)
            actual = {str(p.relative_to(saved)): sha(p) for p in saved.rglob('*')
                      if p.is_file() and p.name != 'quality_summary.json'}
            require(actual == previous['raw_response_and_result_sha256'],
                    'Previously audited run files were changed or added')
            previous_wall = previous.get('wall_seconds_including_client_startup')
        print('Auditing saved responses only; no server connection or inference.', flush=True)
        audit_saved(saved, alias, counts, expected_for_audit, total, previous_wall)
        return

    status, models = http_json('/v1/models')
    require(status == 200, 'Server not ready')
    cards = [card for card in models['data'] if card['id'] == alias]
    require(len(cards) == 1 and cards[0].get('max_model_len') == 8192, 'Server alias/context differs')
    require(Path(cards[0].get('root', '')).resolve() == Path(plan['models'][args.method]['path']).resolve(),
            'Server model card checkpoint path differs')
    server_log = (args.server_log or Path('logs') / (args.method + '_server.log'))
    if not server_log.is_absolute():
        server_log = ROOT / server_log
    check_server_log(server_log, args.method, plan)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    scope = 'full' if args.limit is None else 'first' + str(args.limit)
    out = ROOT / 'outputs/evaluation/pruning_dev_v1' / args.method / (scope + '_' + stamp)
    raw_dir = out / 'http_records'
    raw_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(server_log, out / 'server_before.log')
    save_json(out / 'models.json', models)
    recorder = Recorder(raw_dir, expected, alias)
    thread = threading.Thread(target=recorder.serve_forever, daemon=True)
    thread.start()
    model_args = ','.join((f'model={alias}',
        f'base_url=http://127.0.0.1:{recorder.server_port}/v1/completions',
        f"tokenizer={plan['models']['BF16']['path']}", 'tokenizer_backend=huggingface',
        'tokenized_requests=True', 'add_bos_token=False', 'max_length=8192',
        'max_gen_toks=2048', 'num_concurrent=1', 'max_retries=1', 'timeout=300', 'seed=42'))
    command = [sys.executable, '-m', 'lm_eval', 'run', '--model', 'local-completions',
        '--model_args', model_args, '--tasks', ','.join(TASKS), '--include_path', str(OUT / 'tasks'),
        '--batch_size', '1', '--seed', '42,42,42,42', '--log_samples', '--output_path', str(out / 'harness')]
    if args.limit:
        command += ['--limit', str(args.limit)]
    environment = os.environ.copy()
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        environment.pop(name, None)
    environment.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost',
        OPENAI_API_KEY='local-evaluation', PYTHONUNBUFFERED='1')
    save_json(out / 'run_manifest.json', {'study': 'pruning_dev_v1', 'scope': scope,
        'method': args.method, 'question_counts': counts, 'expected_http_requests': total,
        'protocol_sha256': sha(OUT / 'protocol.json'), 'command': command,
        'checkpoint': plan['models'][args.method], 'gpu_identity': plan['gpu_identity'],
        'runner_files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            (ROOT / 'scripts/prepare_pruning_quality.py', ROOT / 'scripts/run_pruning_quality.py',
             ROOT / 'scripts/pruning_eval_recording.py')},
        'timing_scope': 'Recorder/client overhead included; not serving performance measurements'})
    print(f'Running DEVELOPMENT questions: {counts}; API requests: {total}', flush=True)
    started = time.perf_counter()
    process = None
    try:
        with (out / 'client.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while chunk := process.stdout.read1(4096):
                log.write(chunk); log.flush()
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            returncode = process.wait()
        require(returncode == 0, 'Harness failed; preserved outputs: ' + str(out))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        recorder.shutdown(); recorder.server_close(); thread.join(timeout=5)
        shutil.copy2(server_log, out / 'server_after.log')
    require(sum(len(items) for items in expected.values()) == 0, 'Some frozen requests were not sent')
    check_server_log(out / 'server_after.log', args.method, plan)
    audit_saved(out, alias, counts, expected_for_audit, total, time.perf_counter() - started)


if __name__ == '__main__':
    main()
