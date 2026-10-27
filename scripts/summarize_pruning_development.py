#!/usr/bin/env python3
"""Compare the three audited pruning DEVELOPMENT runs without model requests.

Reads existing run files without changing them. Writes a new JSON summary and
Markdown report. This is a single-width, single-random-seed exploratory study.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path('/root/autodl-tmp/qwen-pruning-lab')
PROTOCOL_SHA = '9ec3eb2d74fd9461d3de850a12fca8ac12b6e4eb15ecd48e72ea35b20d41b352'
METHODS = ('BF16', 'RandomChannels', 'ActivationEnergy')
TASKS = {'gsm8k': ('day7_gsm8k_dev', 256, 'strict-match', 'exact_match'),
         'arc_challenge': ('day7_arc_challenge_dev', 299, 'none', 'acc_norm')}
DEFAULT_RUNS = {
    'BF16': 'BF16/full_20261027T172217745397Z',
    'RandomChannels': 'RandomChannels/full_20261027T171132772864Z',
    'ActivationEnergy': 'ActivationEnergy/full_20261027T170153659348Z',
}
COMPARISONS = (('ActivationEnergy', 'RandomChannels'),
               ('RandomChannels', 'BF16'), ('ActivationEnergy', 'BF16'))
BOOTSTRAP_REPETITIONS = 20000
BOOTSTRAP_SEED = 20261027


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_once(path, payload):
    """Identical reruns are allowed; different prior summaries are preserved."""
    if path.exists():
        require(path.read_bytes() == payload, 'Existing report differs; preserve it: ' + str(path))
    else:
        with path.open('xb') as handle:
            handle.write(payload)


def unwrap(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def verified_run(path, method, protocol, protocol_sha, check_server_log):
    manifest_path, summary_path = path / 'run_manifest.json', path / 'quality_summary.json'
    manifest, summary = read(manifest_path), read(summary_path)
    counts = {task: n for task, n, _, _ in TASKS.values()}
    require(manifest['method'] == method and manifest['scope'] == 'full'
            and manifest['protocol_sha256'] == protocol_sha
            and manifest['question_counts'] == counts
            and manifest['expected_http_requests'] == 1450, method + ': wrong or incomplete run scope')
    require(manifest['checkpoint'] == protocol['models'][method]
            and manifest['gpu_identity'] == protocol['gpu_identity'], method + ': checkpoint/GPU mismatch')
    require(summary['questions_per_task'] == counts and summary['http_requests_verified'] == 1450,
            method + ': full audit missing')
    require(summary['auditor_sha256'] == protocol['file_sha256'][str(ROOT / 'scripts/pruning_eval_recording.py')],
            method + ': auditor differs')
    for relative, digest in manifest['runner_files_sha256'].items():
        require(protocol['file_sha256'].get(str(ROOT / relative)) == digest,
                method + ': runner source differs from frozen protocol')
    actual_hashes = {str(p.relative_to(path)): sha(p) for p in path.rglob('*')
                     if p.is_file() and p.name != 'quality_summary.json'}
    require(actual_hashes == summary['raw_response_and_result_sha256'], method + ': saved run bytes changed')
    check_server_log(path / 'server_before.log', method, protocol)
    check_server_log(path / 'server_after.log', method, protocol)
    cards = [c for c in read(path / 'models.json')['data'] if c['id'] == protocol['models'][method]['alias']]
    require(len(cards) == 1 and cards[0]['max_model_len'] == 8192
            and Path(cards[0]['root']).resolve() == Path(protocol['models'][method]['path']).resolve(),
            method + ': saved model card differs')

    scores, identities, sample_groups = {}, {}, {}
    for short, (task, count, primary_filter, primary_metric) in TASKS.items():
        files = [p for p in (path / 'harness').rglob('*.jsonl') if task in p.name]
        require(len(files) == 1, method + '/' + task + ': expected one samples file')
        groups = defaultdict(dict)
        for line in files[0].read_text().splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            doc, filt = sample['doc_id'], sample['filter']
            require(type(doc) is int and doc not in groups[filt], 'Invalid/duplicate sample ID')
            groups[filt][doc] = sample
        filters = ('strict-match', 'flexible-extract') if short == 'gsm8k' else ('none',)
        require(set(groups) == set(filters), method + ': sample filters differ')
        for filt in filters:
            group = groups[filt]
            require(set(group) == set(range(count)), method + ': missing question/filter pairs')
            for metric in (('exact_match',) if short == 'gsm8k' else ('acc', 'acc_norm')):
                values = [group[i][metric] for i in range(count)]
                require(all(v in (0, 1) for v in values), 'Expected binary per-question scores')
                require(math.isclose(sum(values) / count, summary['metrics'][task][metric + ',' + filt],
                                     rel_tol=0, abs_tol=1e-10), 'Per-question scores differ from aggregate')
        group = groups[primary_filter]
        scores[short] = np.array([group[i][primary_metric] for i in range(count)], dtype=np.int8)
        identities[short] = [(value_sha(group[i]['doc']), group[i]['doc_hash'],
                              group[i]['prompt_hash'], group[i]['target_hash']) for i in range(count)]
        sample_groups[short] = group

    records = [read(p) for p in sorted((path / 'http_records').glob('record_*.json'))]
    require(len(records) == 1450, 'Saved raw request count differs')
    math_records, source_ids, question_prompts, flagged = {}, {}, {}, []
    for item in records:
        require('error' not in item and item.get('http_status') == 200, 'Failed request in audited run')
        metadata, response, payload = item['expected'], item['response'], item['request']
        task, doc = metadata['task'], metadata['doc_id']
        require(task in counts and type(doc) is int and 0 <= doc < counts[task], 'Invalid raw document identity')
        key = (task, doc)
        require(key not in source_ids or source_ids[key] == metadata['source_id'], 'Inconsistent source ID')
        source_ids[key] = metadata['source_id']
        choice_index = metadata.get('choice_index', None)
        prompt_key = (task, doc, choice_index)
        require(prompt_key not in question_prompts, 'Duplicate raw question/choice')
        question_prompts[prompt_key] = value_sha(payload['prompt'])
        require(payload['model'] == protocol['models'][method]['alias']
                and response['model'] == payload['model']
                and payload.get('seed') == 42 and payload.get('temperature') == 0,
                'Raw model or generation setting differs')
        require(response['usage']['prompt_tokens'] == metadata['prompt_tokens'], 'Raw prompt token count differs')
        if task == TASKS['gsm8k'][0]:
            require(doc not in math_records and metadata['kind'] == 'math', 'Duplicate/invalid math request')
            require(payload['max_tokens'] == 2048, 'Math token budget differs')
            math_records[doc] = item
            raw = response['choices'][0]['text']
            sample = sample_groups['gsm8k'][doc]
            require(unwrap(sample['resps'][0]).strip() == raw.strip(), 'Math sample/raw text mismatch')
            missing_marker = re.search(r'#### (\-?[0-9\.\,]+)', raw) is None
            length = response['choices'][0]['finish_reason'] == 'length'
            if missing_marker or length:
                flagged.append({'source_id': metadata['source_id'], 'doc_id': doc,
                    'strict_score': int(sample['exact_match']), 'missing_strict_marker': missing_marker,
                    'length_finish': length, 'ending': raw[-300:]})
    require(len(math_records) == 256 and len(source_ids) == 555, 'Raw question coverage differs')
    length_count = sum(x['response']['choices'][0]['finish_reason'] == 'length' for x in math_records.values())
    require(length_count == summary['math_length_finishes'], 'Length-finish totals differ')
    # ARC has several choice requests per question: never count them as separate questions.
    for doc, sample in sample_groups['arc_challenge'].items():
        require((TASKS['arc_challenge'][0], doc) in source_ids, 'Missing raw science question')
        require(sum(k[:2] == (TASKS['arc_challenge'][0], doc) for k in question_prompts)
                == len(sample['doc']['choices']['text']), 'Science choice count differs')
    return {'scores': scores, 'identities': identities, 'source_ids': source_ids,
            'prompt_hashes': question_prompts, 'flagged': flagged,
            'row': {'method': method,
                    'gsm8k_correct': int(scores['gsm8k'].sum()), 'gsm8k_percent': float(scores['gsm8k'].mean() * 100),
                    'arc_correct': int(scores['arc_challenge'].sum()), 'arc_percent': float(scores['arc_challenge'].mean() * 100),
                    'math_length_finishes': length_count,
                    'math_without_strict_marker': sum(x['missing_strict_marker'] for x in flagged)},
            'receipt': {'path': str(path), 'manifest_sha256': sha(manifest_path),
                        'quality_summary_sha256': sha(summary_path), 'audited_files': len(actual_hashes)}}


def paired(first, second, repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED):
    """Resample the same question indices in both models; conditional on these fixed checkpoints."""
    first, second = np.asarray(first), np.asarray(second)
    require(first.ndim == second.ndim == 1 and first.shape == second.shape and len(first) > 0,
            'Paired arrays must have identical nonempty question axes')
    require(np.isin(first, (0, 1)).all() and np.isin(second, (0, 1)).all(), 'Expected binary scores')
    first, second = first.astype(np.int8), second.astype(np.int8)
    delta = first - second
    gains, losses = int((delta == 1).sum()), int((delta == -1).sum())
    discordant = gains + losses
    p = min(1., 2 * sum(math.comb(discordant, k)
                       for k in range(min(gains, losses) + 1)) / 2 ** discordant)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 256):
        size = min(256, repetitions - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        draws[start:start + size] = delta[indices].mean(axis=1) * 100
    return {'questions': len(delta), 'gains': gains, 'losses': losses,
            'both_correct': int(((first == 1) & (second == 1)).sum()),
            'both_incorrect': int(((first == 0) & (second == 0)).sum()),
            'delta_pp': float(delta.mean() * 100),
            'paired_question_bootstrap_95_interval_pp': [float(x) for x in np.percentile(draws, [2.5, 97.5])],
            'exact_two_sided_mcnemar_p_unadjusted': p}


def report_markdown(summary):
    lines = ['# Structured MLP pruning: development comparison', '',
        'Original BF16 versus two BF16 checkpoints with MLP width reduced from 9,728 to 9,216. '
        'No recovery training. Same frozen prompts, server settings, question identities and scoring.', '',
        '| Model | Math correct / 256 | Math % | Science correct / 299 | Science % | Math length finishes |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary['models']:
        lines.append(f"| {row['method']} | {row['gsm8k_correct']} | {row['gsm8k_percent']:.2f} | "
                     f"{row['arc_correct']} | {row['arc_percent']:.2f} | {row['math_length_finishes']} |")
    lines += ['', '## Paired differences', '', 'Gains/losses are questions on which only the first/second model is correct.', '',
        '| First minus second | Task | Gains | Losses | Difference (pp) | Approximate 95% interval (pp) | Exact p, unadjusted |',
        '|---|---|---:|---:|---:|---:|---:|']
    for item in summary['paired_comparisons']:
        lo, hi = item['paired_question_bootstrap_95_interval_pp']
        lines.append(f"| {item['first']} − {item['second']} | {item['task']} | {item['gains']} | {item['losses']} | "
                     f"{item['delta_pp']:+.4f} | [{lo:.4f}, {hi:.4f}] | {item['exact_two_sided_mcnemar_p_unadjusted']:.6f} |")
    lines += ['', '## Interpretation and limits', '',
        '- This is reused development data from Project 1, not a fresh unseen holdout.',
        '- Intervals use 20,000 paired question resamples; math seed 20261027, science seed 20261028. They are conditional on these '
        'three checkpoints and these evaluation settings; they do not capture random-channel seed variation.',
        '- The comparison is exploratory: one pruning width, one random-channel seed and two tasks. '
        'P-values and individual intervals are unadjusted for multiple comparisons.',
        '- An interval containing zero does not establish equal quality or non-inferiority.',
        '- GSM8K uses the frozen strict answer-marker rule. A model can earn credit for an early answer '
        'and continue repeating or reach the length limit. No answers were rescored or repaired.',
        '- Math length finishes and missing-marker diagnostics are in the JSON; they may overlap. '
        'A missing marker means the frozen extraction regex found no match, not necessarily a wrong numerical answer.',
        '- Checkpoint size reduction does not demonstrate serving speedup. No performance measurements '
        'are derived from these evaluation runs.', '',
        'Source run hashes, per-question scores and flagged response endings are preserved in development_summary.json.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for method, flag in (('BF16', 'bf16'), ('RandomChannels', 'random'), ('ActivationEnergy', 'energy')):
        parser.add_argument('--' + flag + '-run', type=Path,
                            default=ROOT / 'outputs/evaluation/pruning_dev_v1' / DEFAULT_RUNS[method])
    args = parser.parse_args()
    from prepare_pruning_quality import OUT, verify_protocol
    from run_pruning_quality import check_server_log
    protocol_path = OUT / 'protocol.json'
    require(sha(protocol_path) == PROTOCOL_SHA, 'Expected the recorded frozen pruning evaluation protocol')
    print('Verifying frozen protocol, checkpoint files and three saved full runs (read-only)...', flush=True)
    protocol = verify_protocol()
    require(protocol['methods'] == list(METHODS), 'Method ordering differs')
    paths = dict(zip(METHODS, (args.bf16_run, args.random_run, args.energy_run)))
    runs = {}
    for method, path in paths.items():
        runs[method] = verified_run(path.resolve(), method, protocol, PROTOCOL_SHA, check_server_log)
        print('Verified', method + ': 256 math + 299 science questions; saved audit hashes match', flush=True)
    baseline = runs['BF16']
    for method in METHODS[1:]:
        for key in ('identities', 'source_ids', 'prompt_hashes'):
            require(runs[method][key] == baseline[key], 'Paired question/prompt identity differs: ' + method)
    comparisons = []
    for first, second in COMPARISONS:
        for offset, task in enumerate(TASKS):
            item = paired(runs[first]['scores'][task], runs[second]['scores'][task],
                          seed=BOOTSTRAP_SEED + offset)
            item.update(first=first, second=second, task=task)
            comparisons.append(item)
    summary = {'study': 'pruning_dev_v1', 'scope': 'previously used development sets only',
        'protocol_sha256': PROTOCOL_SHA, 'summarizer_sha256': sha(Path(__file__).resolve()),
        'analysis': {'bootstrap_repetitions': BOOTSTRAP_REPETITIONS, 'math_seed': BOOTSTRAP_SEED,
                     'science_seed': BOOTSTRAP_SEED + 1, 'numpy_version': np.__version__,
                     'inference_unit': 'paired question', 'multiple_comparison_adjustment': 'none; exploratory'},
        'models': [runs[m]['row'] for m in METHODS], 'paired_comparisons': comparisons,
        'runs': {m: runs[m]['receipt'] for m in METHODS},
        'per_question_primary_scores': {m: {t: runs[m]['scores'][t].tolist() for t in TASKS} for m in METHODS},
        'paired_question_source_ids': {t: [baseline['source_ids'][(spec[0], i)] for i in range(spec[1])]
                                       for t, spec in TASKS.items()},
        'flagged_math_responses': {m: runs[m]['flagged'] for m in METHODS},
        'model_requests_sent': 0, 'serving_performance_measured': False}
    out = ROOT / 'outputs/evaluation/pruning_dev_v1'
    write_once(out / 'development_summary.json', (json.dumps(summary, indent=2, ensure_ascii=False) + '\n').encode())
    write_once(out / 'development_report.md', report_markdown(summary).encode())
    print('\nModel                Math %   Science %   Length finishes')
    for row in summary['models']:
        print(f"{row['method']:<20} {row['gsm8k_percent']:7.2f} {row['arc_percent']:11.2f} {row['math_length_finishes']:17}")
    print('\nPaired comparisons: first minus second')
    for item in comparisons:
        lo, hi = item['paired_question_bootstrap_95_interval_pp']
        print(f"{item['first']} - {item['second']}; {item['task']}: "
              f"gains={item['gains']} losses={item['losses']} delta={item['delta_pp']:+.4f} pp; "
              f"95% interval [{lo:.4f}, {hi:.4f}]; exact p={item['exact_two_sided_mcnemar_p_unadjusted']:.6f}")
    print('\nSaved:', out / 'development_summary.json')
    print('Saved:', out / 'development_report.md')
    print('PASS: three full development runs, saved audit hashes, paired identities and score totals verified.')
    print('No inference performed. Single-width, single-random-seed exploratory results; speed remains unmeasured.')


if __name__ == '__main__':
    main()
