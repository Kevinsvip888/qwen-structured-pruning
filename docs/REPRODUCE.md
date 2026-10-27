# Reproduction guide

There are two different workflows: verify the published snapshot on any CPU, or rerun the GPU experiment with its original dependencies. Snapshot verification is self-contained. The original experiment is deliberately pinned to its recorded paths, software and physical GPU identity.

## 1 Verify the published results

From the unpacked package or a Git checkout:

```bash
python3 tools/verify_public_snapshot.py
```

This needs Python's standard library and no GPU, network or ML packages. It checks hashes, original selections and recorded receipts, the complete native benchmark evidence, and derived score/performance arithmetic. See [EVIDENCE.md](EVIDENCE.md) for the omitted tensors and raw quality files it cannot audit.

## 2 Prepare the original prerequisites

The original absolute project roots are:

```text
/root/autodl-tmp/qwen-inference-lab
/root/autodl-tmp/qwen-pruning-lab
```

Project 1 is [qwen-fp8-inference-lab](https://github.com/Kevinsvip888/qwen-fp8-inference-lab), published at commit `d790f31`. Its [reproduction guide](https://github.com/Kevinsvip888/qwen-fp8-inference-lab/blob/d790f31/docs/REPRODUCE.md) describes the source preparation. For this extension you need the original BF16 model, seed-42 audited calibration pool/Random128 subset, reserved GSM8K split, and prepared Day 7 development tasks and tokenized-request metadata. FP8 checkpoint generation is not needed for the pruning method itself.

The model is `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Parent source revisions are:

| Dataset | Revision | Use |
|---|---|---|
| Salesforce/wikitext | `b08601e04326c79dfdd32d625aee71d232d685c3` | General calibration text |
| openai/gsm8k | `740312add88f781978c0658806c59bc2815b9866` | Math calibration and reserved development/few-shot rows |
| allenai/ai2_arc | `210d026faf9955653af8916fad021475a3f00453` | ARC-Challenge validation questions |

Use the existing pinned Project 1 environments:

- `.venv-serve` for preparation, observation, surgery, export, serving and performance benchmarks.
- `.venv-eval` for quality preparation, inference clients and the quality summary.

See [the environment record](../environment/README.md). These instructions do not upgrade an environment or assume that every historical CUDA wheel remains available.

The original GPU UUID is recorded in the archived protocols. A new UUID, package set, installed model implementation, source byte sequence or absolute path may cause a frozen check to fail. That is expected protection, not a suggestion to bypass the check. For a different machine, derive an explicitly named new study/protocol and audit its changed dependencies; do not label it an exact replay of these receipts.

For the original setup, the following read-only check should find the parent assets:

```bash
cd /root/autodl-tmp/qwen-pruning-lab
python3 - <<'PY'
from pathlib import Path
base = Path('/root/autodl-tmp/qwen-inference-lab')
for name in (
    'models/Qwen3-4B-Instruct-2507/config.json',
    'logs/model_revision.json',
    'data/calibration/seed_42/pool.jsonl',
    'data/calibration/seed_42/manifest.json',
    'data/calibration/seed_42/pool_audit.json',
    'data/calibration/seed_42/random_subsets_manifest.json',
    'data/splits/gsm8k_train_split.json',
    'data/evaluation/day7_quality_v1/development_protocol.json',
    'data/evaluation/day7_quality_v1/harness_source_hashes.json',
):
    path = base / name
    print(path.is_file(), path)
    assert path.is_file(), path
PY
```

The real preparation scripts perform the stronger hash, shape, split and environment checks. This existence check does not substitute for them.

## 3 Freeze the pruning study

Stop any previous model server before model observation or surgery so the GPU has sufficient free memory. Start from the original source checkpoint; never modify it in place.

```bash
cd /root/autodl-tmp/qwen-pruning-lab
source /root/autodl-tmp/qwen-inference-lab/.venv-serve/bin/activate
mkdir -p logs
set -o pipefail

python -u scripts/prepare_pruning_study.py \
    2>&1 | tee logs/prepare_pruning_study.log
```

This freezes width 9,216, both channel-selection rules, 128 calibration sequences, source file hashes, parameter arithmetic and the environment. It writes `data/protocol_v1.json` and copies the exact existing Random128 token blocks. It does not prune or score a model.

## 4 Collect importance and validate surgery

```bash
HF_HUB_OFFLINE=1 python -u scripts/collect_channel_importance.py \
    2>&1 | tee logs/collect_channel_importance.log

HF_HUB_OFFLINE=1 python -u scripts/validate_mlp_surgery.py \
    2>&1 | tee logs/validate_mlp_surgery.log
```

The collector observes every original MLP on 65,536 tokens and writes activation statistics, pass metadata, both selections and a completion pointer. Selection uses the original unpruned model throughout. The validator compares compact versus masked MLPs in FP64, checks native BF16 behavior and rejects invalid selections/layouts. It does not measure model quality or speed.

Do not copy the archived importance/surgery pointers or receipts into live `data/` and `outputs/` directories to skip these stages. The archived activation-statistics tensor is omitted; it must be generated by a real collection run.

## 5 Export both compact BF16 checkpoints

```bash
HF_HUB_OFFLINE=1 python -u scripts/export_pruned_checkpoint.py \
    --method RandomChannels \
    2>&1 | tee logs/export_RandomChannels.log

HF_HUB_OFFLINE=1 python -u scripts/export_pruned_checkpoint.py \
    --method ActivationEnergy \
    2>&1 | tee logs/export_ActivationEnergy.log
```

Each export starts from fresh original weights, copies the prescribed rows/columns, preserves other weights and tokenizer state, saves to a temporary directory and reloads for exact checks. Completed checkpoint directories are:

```text
models/Qwen3-4B-Instruct-2507-MLP9216-RandomChannels
models/Qwen3-4B-Instruct-2507-MLP9216-ActivationEnergy
```

An existing completed checkpoint is verified rather than overwritten. Export elapsed time is not a serving benchmark.

## 6 Check serving

Use two terminals. In Terminal A, start one model at a time. Stop it with Ctrl+C before changing methods.

```bash
cd /root/autodl-tmp/qwen-pruning-lab
set -o pipefail
bash tools/serve_model.sh ActivationEnergy \
    2>&1 | tee logs/ActivationEnergy_server.log
```

For other methods, use `RandomChannels` or `BF16` and the corresponding log filename. The aliases are `qwen-prune-energy`, `qwen-prune-random` and `qwen-prune-bf16`. The wrapper activates the pinned serving environment and supplies the shared server options.

In Terminal B, after the server is ready:

```bash
curl --noproxy '*' --fail-with-body --silent --show-error \
    --max-time 120 http://127.0.0.1:8000/v1/chat/completions \
    --header 'Content-Type: application/json' \
    --data '{"model":"qwen-prune-energy","messages":[{"role":"user","content":"Calculate 17 multiplied by 23. Give the answer and one brief explanation."}],"temperature":0,"top_p":1,"max_tokens":128,"seed":42}'
```

Adapt only the alias for other methods. A correct arithmetic answer is a smoke check, not a quality benchmark or a deterministic guarantee on different wording.

## 7 Freeze and run development quality

Keep the selected server in Terminal A. In Terminal B:

```bash
cd /root/autodl-tmp/qwen-pruning-lab
source /root/autodl-tmp/qwen-inference-lab/.venv-eval/bin/activate
set -o pipefail

HF_HUB_OFFLINE=1 python -u scripts/prepare_pruning_quality.py \
    2>&1 | tee logs/prepare_pruning_quality.log

HF_HUB_OFFLINE=1 python -u scripts/run_pruning_quality.py \
    --method ActivationEnergy --limit 8 \
    2>&1 | tee logs/ActivationEnergy_pilot8.log
```

The pilot checks eight questions per task. Its scores are not the final development results. Omit `--limit` for 256 math plus 299 science questions:

```bash
HF_HUB_OFFLINE=1 python -u scripts/run_pruning_quality.py \
    --method ActivationEnergy \
    2>&1 | tee logs/ActivationEnergy_development_full.log
```

Run the same full command for `RandomChannels` and `BF16` after switching Terminal A to the matching checkpoint/log. Each full run sends 1,450 API requests: 256 math generations and 1,194 science answer-choice likelihood requests. Science choices are not separate test questions.

The runner records raw API results, checks prompt identities/token counts, and cross-checks the harness. Wait for the final audit PASS before changing servers. Saved responses can be audited without inference:

```bash
python -u scripts/run_pruning_quality.py --method ActivationEnergy \
    --audit-only outputs/evaluation/pruning_dev_v1/ActivationEnergy/full_TIMESTAMP
```

`full_TIMESTAMP` is a placeholder: replace it with the actual directory printed by your run. To summarize the original recorded run paths:

```bash
python -u scripts/summarize_pruning_development.py \
    2>&1 | tee logs/summarize_pruning_development.log
```

For fresh run timestamps, supply `--bf16-run`, `--random-run` and `--energy-run` with those actual directories. The original default timestamps identify the specific archived study, not whichever run scores best.

The primary metrics remain GSM8K strict-match and ARC acc_norm. Do not change the parser, add fallback answers, choose new pruning rates or tune on these results while claiming the original frozen comparison.

## 8 Freeze performance and run a pilot

In Terminal B, switch back to the serving environment:

```bash
source /root/autodl-tmp/qwen-inference-lab/.venv-serve/bin/activate
python -u scripts/prepare_pruning_benchmarks.py \
    2>&1 | tee logs/prepare_pruning_benchmarks.log
```

Start BF16 in Terminal A using `tools/serve_model.sh BF16` and `logs/BF16_server.log`. After it is ready:

```bash
python -u scripts/run_pruning_benchmark.py --method BF16 --pilot --concurrency 4 \
    2>&1 | tee logs/BF16_benchmark_pilot.log
```

The pilot has four warmup and eight measured requests and is excluded from formal results. The runner checks the model path/alias, server options, sole GPU process, native client lengths, result metrics and memory samples.

## 9 Run all 18 formal benchmarks

Use the model order and workload order below. Each block keeps one model server active for its two workloads. Keep the same environment/settings, run no competing GPU work, and wait for the final audit receipt before stopping a server.

| Repetition | Model block order | Concurrency order within each block |
|---|---|---|
| 1 | BF16, RandomChannels, ActivationEnergy | 4, then 1 |
| 2 | RandomChannels, ActivationEnergy, BF16 | 1, then 4 |
| 3 | ActivationEnergy, BF16, RandomChannels | 4, then 1 |

For example, with BF16 running in Terminal A, the first block in Terminal B is:

```bash
for concurrency in 4 1; do
    python -u scripts/run_pruning_benchmark.py \
        --method BF16 --repetition 1 --concurrency "$concurrency" \
        2>&1 | tee "logs/BF16_benchmark_rep1_c${concurrency}.log" || break
done
```

Switch Terminal A to the next model using the same wrapper and matching log. Adapt the method, repetition and loop order for each block in the table. The runner enforces the complete 18-run schedule and preserves existing run directories. Do not rerun completed repetitions under the same output path or copy receipts to advance the schedule.

Each formal run has 20 warmups and 200 measured requests, fixed 1,024 input/128 output lengths recorded by the native client, an infinite offered request rate and the specified concurrency cap. Detailed results and 1-second whole-GPU memory samples are saved together.

## 10 Verify and summarize

After the final audit PASS, the server may be stopped. No running model is needed for this step:

```bash
python -u scripts/summarize_pruning_benchmarks.py \
    2>&1 | tee logs/summarize_pruning_benchmarks.log
```

This checks the frozen source/environment/GPU metadata, all 18 receipts, native records, CLI arguments, startup evidence and memory CSVs. It writes `outputs/benchmark/pruning_v1/benchmark_summary.json` and `benchmark_report.md`. It computes means, sample SDs and ratios of means; it does not add inference or a significance claim.

For publication, keep runtime outputs intact and copy only reviewed evidence into the archived `provenance/` and `results/` layout. [Publishing instructions](PUBLISH.md) describe the reviewed file set. A small publication snapshot is not a substitute for backing up omitted weights, raw quality records and calibration statistics if you need them later.
