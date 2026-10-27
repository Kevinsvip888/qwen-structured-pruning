# Qwen structured MLP channel pruning

An exploratory pruning study of Qwen3-4B-Instruct-2507 on an RTX 4090. It physically removes channels from every MLP, compares activation-energy selection with random selection, and measures checkpoint size, development quality and vLLM serving performance without recovery training.

The activation-energy checkpoint removes **141.6M parameters (3.52% of the model)** and delivers **3.22% higher output throughput** in the primary workload, with **286 MiB lower mean sampled GPU-memory maxima**. Its development scores are close to the original BF16 model; this study does **not** establish equal quality. The gain is modest and workload-specific.

This is an extension of the [Qwen FP8 inference lab](https://github.com/Kevinsvip888/qwen-fp8-inference-lab), with a separate BF16 pruning experiment. Pruning and FP8 are not combined here.

## Results

| Measurement | Original BF16 | RandomChannels | ActivationEnergy |
|---|---:|---:|---:|
| MLP intermediate width in all 36 layers | 9,728 | 9,216 | 9,216 |
| Unique parameters | 4,022,468,096 | 3,880,910,336 | 3,880,910,336 |
| Safetensors storage, GiB | 7.4925 | 7.2288 | 7.2288 |
| GSM8K development strict match, % | 89.84 | 85.94 | 91.02 |
| ARC-Challenge validation acc_norm, % | 53.85 | 49.83 | 52.84 |
| Output tokens/s, concurrency 4, mean ± sample SD | 322.66 ± 0.345 | 332.89 ± 0.187 | 333.07 ± 0.163 |
| Output tokens/s, concurrency 1, mean ± sample SD | 98.87 ± 0.083 | 101.89 ± 0.076 | 101.86 ± 0.072 |
| Sampled GPU maximum, concurrency 4, mean MiB | 12,911 | 12,625 | 12,625 |

Serving uses 1,024 client-recorded input tokens, 128 output tokens, a fixed 4 GiB BF16 KV cache, FlashAttention 2, an 8,192-token context and disabled prefix caching. Each cell has three runs of 200 measured requests after 20 warmups: **18 runs and 3,600 measured requests**. The primary comparison is ActivationEnergy versus BF16 at concurrency 4. Ratios divide mean run-level throughput; the repetitions are descriptive, not a significance test.

Removing 512 of 9,728 channels is **5.26% of MLP width**, not 5.26% of total model parameters. The parameter payload saving is 270 MiB. Sampled whole-GPU memory includes allocations beyond model weights and may miss brief peaks.

## Quality interpretation

The same 256 GSM8K development questions and 299 ARC-Challenge validation questions were used for all three checkpoints. GSM8K uses five fixed reserved examples, plain completion prompts and strict answer extraction; ARC uses zero-shot answer-choice likelihoods and length normalization. These are previously used Project 1 development sets, **not fresh unseen test data**. No additional official-test evaluation was run for pruning.

| Paired difference | Math, percentage points | Science, percentage points |
|---|---:|---:|
| ActivationEnergy minus RandomChannels | +5.08 [0.78, 9.38] | +3.01 [-1.00, 7.02] |
| ActivationEnergy minus BF16 | +1.17 [-1.56, 3.91] | -1.00 [-4.01, 1.67] |

Brackets show approximate paired question-bootstrap 95% intervals. The activation-energy math advantage over random selection has an unadjusted exact McNemar p-value of 0.0351. This is an exploratory comparison with one pruning width, one random-channel seed and multiple comparisons. An interval containing zero does not establish equivalence or non-inferiority. A correct early answer marker can earn credit even if generation later repeats or reaches the output limit.

## Method

For the original MLP, let `a = SiLU(gate_proj(x)) * up_proj(x)`. Channel `j` receives the score

```text
score[j] = mean_tokens(a[j] ** 2) * sum_output_dims(down_proj.weight[:, j] ** 2)
```

Statistics are collected from the original unpruned model on 128 existing calibration sequences of 512 tokens: 64 WikiText and 64 GSM8K sequences, totaling 65,536 tokens. Math calibration rows exclude the reserved development and few-shot rows.

ActivationEnergy keeps the top 9,216 channels per layer; score ties prefer lower indices. RandomChannels uses a fixed, layer-specific seed and keeps the same number. Both selections are stored in ascending index order. The same indices slice gate/up rows and down columns. Attention, hidden size, normalizations, tokenizer, tied embeddings and BF16 dtype are retained.

The score is an output-contribution energy heuristic. It ignores correlations between channels and is not an exact estimate of task-loss change.

## Verification

The public snapshot can be checked without a GPU or ML packages:

```bash
python3 tools/verify_public_snapshot.py
```

This verifies original/public file hashes, selected channel identities, recorded validation/export receipts, all 18 included native benchmark records and memory CSVs, performance arithmetic, and saved quality score vectors and paired counts. It does not rerun CUDA or reconstruct omitted quality responses, model weights or activation-statistics tensors. It does not reproduce the bootstrap draws.

The original experiment recorded 42 toy FP64 surgery checks, 144 real-layer FP64 comparisons, 144 native BF16 smokes, 144 exact projection-copy checks and 13 input-rejection checks. Both exports recorded 398 exact saved-tensor checks and 398 reloaded-parameter checks. These receipts are included; their omitted tensors are not independently re-audited by the CPU snapshot checker.

## Reproduction and files

- [Reproduction guide](docs/REPRODUCE.md): prerequisite Project 1 assets, pinned environments, commands and the 18-run order.
- [Evidence and limits](docs/EVIDENCE.md): what the public files can verify.
- [Engineering notes](docs/ENGINEERING.md): score collection, safe tensor surgery and fair measurement.
- [Script map](docs/SCRIPT_MAP.md): inputs and outputs of each script.
- [Publishing steps](docs/PUBLISH.md): stage and review only the public files.
- [Environment record](environment/README.md): recorded package versions and replay constraints.
- [Quality report](results/quality/development_report.md) and [serving report](results/benchmark/benchmark_report.md): full precision JSON is adjacent.

The 13 original Python files and the validation specification are unchanged. Archived runtime records are under `provenance/original/`, separate from live `data/`, `models/` and `outputs/` directories. Model weights and raw quality responses are not distributed.

The original scripts intentionally pin absolute paths, source hashes, package versions, protocol hashes and the original GPU identity. **CPU snapshot verification is portable; unchanged experimental replay is not a one-command setup on an arbitrary machine.** A new environment/GPU requires an explicitly versioned new study. See the reproduction guide before attempting replay.

Model and dataset access remains subject to the upstream repositories' terms. No new software license has been selected in this publication package.
