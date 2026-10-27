#!/usr/bin/env python3
"""Freeze the pruning DEVELOPMENT comparison without any model requests."""
import hashlib
import json
from importlib.metadata import distribution, version
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/qwen-pruning-lab')
OLD = Path('/root/autodl-tmp/qwen-inference-lab')
PREPARED = OLD / 'data/evaluation/day7_quality_v1'
OUT = ROOT / 'data/evaluation/pruning_dev_v1'
TASKS = ('day7_gsm8k_dev', 'day7_arc_challenge_dev')
COUNTS = dict(zip(TASKS, (256, 299)))
ALIASES = {'BF16': 'qwen-prune-bf16', 'RandomChannels': 'qwen-prune-random',
           'ActivationEnergy': 'qwen-prune-energy'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, 'Frozen file differs; preserve it: ' + str(path))
    else:
        with path.open('xb') as handle:
            handle.write(payload)


def gpu():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    require(len(text.splitlines()) == 1, 'Expected one physical GPU')
    fields = [x.strip() for x in text.split(',')]
    require(len(fields) == 3, 'Unexpected GPU metadata')
    return dict(zip(('uuid', 'name', 'driver'), fields))


def check_environment():
    expected = {'lm_eval': '0.4.13', 'datasets': '5.1.0',
                'transformers': '5.19.0', 'huggingface_hub': '2.3.0'}
    require({k: version(k) for k in expected} == expected, 'Evaluation package versions differ')
    harness = Path(distribution('lm_eval').locate_file('lm_eval'))
    actual = {str(p.relative_to(harness)): sha(p) for p in harness.rglob('*.py')}
    require(actual == read(PREPARED / 'harness_source_hashes.json'), 'Harness source changed')
    return expected


def serve_environment():
    python = OLD / '.venv-serve/bin/python'
    command = "import json,sys;from importlib.metadata import version;print(json.dumps({'python':sys.version.split()[0],'packages':{k:version(k) for k in ('torch','vllm','triton','transformers','safetensors')}}))"
    installed = json.loads(subprocess.check_output([str(python), '-c', command], text=True))
    require(installed == {'python': '3.12.15', 'packages': {
        'torch': '2.13.0+cu132', 'vllm': '0.32.0', 'triton': '3.7.1',
        'transformers': '5.19.0', 'safetensors': '0.8.0'}}, 'Serving package versions differ')
    freeze = subprocess.check_output(['uv', 'pip', 'freeze', '--python', str(python)],
                                      text=True, stderr=subprocess.PIPE)
    return {'installed': installed, 'freeze_sha256': hashlib.sha256(freeze.encode()).hexdigest()}


def verify_protocol():
    plan = read(OUT / 'protocol.json')
    require(plan['methods'] == list(ALIASES), 'Method scope changed')
    require(plan['question_counts'] == COUNTS, 'Question counts changed')
    for path, digest in plan['file_sha256'].items():
        require(sha(path) == digest, 'Pinned file changed: ' + path)
    require(check_environment() == plan['eval_packages'], 'Evaluation environment differs')
    require(serve_environment() == plan['serve_environment'], 'Serving environment differs')
    require(gpu() == plan['gpu_identity'], 'GPU/driver changed')
    return plan


def main():
    require(sys.version.split()[0] == '3.12.15', 'Expected Python 3.12.15')
    packages = check_environment()
    require(sha(PREPARED / 'development_protocol.json') ==
            '44dc329b87c700a065e15c7395f040d8beb4ea13d6063ab28efa9f806ca7449f',
            'Expected the original frozen Day 7 development protocol')
    old_protocol = read(PREPARED / 'development_protocol.json')
    require(old_protocol['context_limit'] == 8192 and old_protocol['gsm8k_output_limit'] == 2048,
            'Unexpected inherited token budgets')
    require(old_protocol['generation_seed'] == 42 and not old_protocol['apply_chat_template'],
            'Unexpected inherited prompting settings')
    files = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, 'File changed: ' + str(path))
        files[str(path)] = digest

    for relative, digest in old_protocol['file_sha256'].items():
        pin(OLD / relative, digest)
    pin(PREPARED / 'development_protocol.json')
    for name in ('prepare_pruning_quality.py', 'run_pruning_quality.py', 'pruning_eval_recording.py'):
        pin(ROOT / 'scripts' / name)
    source = read(ROOT / 'data/protocol_v1.json')
    pin(ROOT / 'data/protocol_v1.json')
    require(sha(ROOT / 'data/protocol_v1.json') ==
            'f4482e32ad3e8b9564659856610fd9129607d194dc832d5faf48460b92d55592',
            'Expected the original frozen pruning protocol')
    require(source['target_intermediate_size'] == 9216, 'Pruning width differs')
    require(gpu() == source['gpu_identity'], 'GPU differs from pruning preparation')
    print('Verifying original and both exported checkpoint files (read-only)...', flush=True)
    for path, digest in source['source_file_sha256'].items():
        pin(path, digest)
    models = {'BF16': {'path': source['source_model']['local_dir'], 'alias': ALIASES['BF16']}}
    shared_identity = None
    for method in ('RandomChannels', 'ActivationEnergy'):
        path = ROOT / 'models' / ('Qwen3-4B-Instruct-2507-MLP9216-' + method)
        manifest_path = path / 'pruning_manifest.json'
        manifest = read(manifest_path)
        identity = manifest['identity']
        require(identity['method'] == method and identity['protocol_sha256'] == sha(ROOT / 'data/protocol_v1.json'),
                'Checkpoint method/protocol mismatch')
        require(identity['target_intermediate_size'] == 9216 and identity['dtype'] == 'bfloat16'
                and not identity['quantization'] and not identity['fine_tuning'], 'Checkpoint recipe differs')
        comparable = {k: identity[k] for k in ('protocol_sha256', 'importance_manifest_sha256',
                       'surgery_validation_sha256', 'exporter_script_sha256', 'surgery_helper_sha256')}
        require(shared_identity is None or comparable == shared_identity, 'Export provenance differs')
        shared_identity = comparable
        require(manifest['source_files_unchanged'] and all(manifest['reload_checks'][k] for k in
                ('tied_embeddings', 'probe_hidden_states_exact', 'probe_logits_exact', 'tokenizer_unchanged')),
                'Export validation incomplete')
        require(manifest['pruned_unique_parameters'] == 3880910336, 'Parameter count differs')
        pin(manifest_path)
        for relative, info in manifest['export_files'].items():
            require((path / relative).resolve().is_relative_to(path.resolve()), 'Unsafe export path')
            require((path / relative).stat().st_size == info['bytes'], 'Export file size differs')
            pin(path / relative, info['sha256'])
        require(read(path / 'config.json')['intermediate_size'] == 9216, 'Export config width differs')
        models[method] = {'path': str(path), 'alias': ALIASES[method],
                          'pruning_manifest_sha256': sha(manifest_path)}

    for path, digest in source['calibration_evidence_sha256'].items():
        pin(path, digest)
    calibration_path = ROOT / source['calibration']['file']
    pin(calibration_path, source['calibration']['sha256'])
    calibration = [json.loads(line) for line in calibration_path.read_text().splitlines() if line.strip()]
    require(len(calibration) == 128 and [x['id'] for x in calibration] == source['calibration']['sequence_ids'],
            'Pruning calibration sequence identities differ')
    split = read(OLD / 'data/splits/gsm8k_train_split.json')['partitions']
    excluded = {x['row'] for key in ('development', 'fewshot_reserve') for x in split[key]}
    eligible = {x['row'] for x in split['calibration']}
    math_rows = {span['row'] for record in calibration if record['domain'] == 'math' for span in record['sources']}
    require(len(math_rows) == source['calibration']['distinct_math_training_rows']
            and math_rows.issubset(eligible) and math_rows.isdisjoint(excluded),
            'Pruning calibration overlaps development/few-shot rows')
    for task in TASKS:
        request_path = PREPARED / (task + '_requests.jsonl')
        rows = [json.loads(line) for line in request_path.read_text().splitlines() if line.strip()]
        require({x['doc_id'] for x in rows} == set(range(COUNTS[task])), 'Request document IDs differ')
        require(len(rows) == old_protocol['request_counts'][task], 'Request count differs')
    # Copy task YAML bytes only; referenced original data stay read-only and hash-pinned.
    for task in TASKS:
        path = OUT / 'tasks' / (task + '.yaml')
        payload = (PREPARED / 'tasks/development' / (task + '.yaml')).read_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            require(path.read_bytes() == payload, 'Copied task configuration differs')
        else:
            path.write_bytes(payload)
        pin(path)
    plan = {'study': 'pruning_dev_v1', 'scope': 'development only; previously used data',
            'methods': list(ALIASES), 'models': models, 'question_counts': COUNTS,
            'request_counts': old_protocol['request_counts'], 'context_limit': 8192,
            'math_max_output_tokens': 2048, 'generation_seed': 42,
            'apply_chat_template': False, 'math_fewshot_ids': old_protocol['fewshot_ids'],
            'primary_scores': {'gsm8k': 'exact_match,strict-match', 'arc_challenge': 'acc_norm,none'},
            'primary_comparison': 'ActivationEnergy minus RandomChannels; paired by question',
            'secondary_comparisons': ['RandomChannels minus BF16', 'ActivationEnergy minus BF16'],
            'reporting': 'Report both task scores, paired differences, extraction failures and length finishes; no equivalence claim from nonsignificance.',
            'scope_limit': 'One pruning width and one random channel seed; no recovery training. Development sets already used in Project 1; not fresh holdouts.',
            'server': {'dtype': 'bfloat16', 'kv_cache_dtype': 'bfloat16',
                       'kv_cache_memory_bytes': 4294967296, 'attention_backend': 'FLASH_ATTN',
                       'max_model_len': 8192, 'max_num_seqs': 4, 'max_num_batched_tokens': 4096,
                       'generation_config': 'vllm', 'enable_prefix_caching': False, 'seed': 42},
            'eval_packages': packages, 'serve_environment': serve_environment(),
            'gpu_identity': gpu(), 'file_sha256': files,
            'official_test_inference': False, 'model_requests_sent_during_preparation': 0}
    put(OUT / 'protocol.json', plan)
    print('Models: BF16, RandomChannels, ActivationEnergy')
    print('Development questions: 256 math + 299 science per model')
    print('API requests for a full run: 256 math generations + 1194 science choices = 1450')
    print('Protocol SHA256:', sha(OUT / 'protocol.json'))
    print('Saved:', OUT / 'protocol.json')
    print('PASS: checkpoint hashes, existing prompts/scoring and calibration separation frozen.')
    print('No model requests or scores computed. Existing Project 1 development data are reused.')


if __name__ == '__main__':
    main()
