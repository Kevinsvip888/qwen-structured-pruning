# Structured MLP pruning: serving benchmark

Fresh measurements of original BF16, RandomChannels and ActivationEnergy on one RTX 4090. All models use BF16 weights and BF16 KV cache; no FP8 or recovery training.

The workload has 1,024 client-recorded input tokens and 128 output tokens. Each cell has three runs of 200 measured requests after 20 warmup requests. Concurrency 4 is primary; concurrency 1 is secondary. Pilots are excluded.

## Mean performance across three runs

| Concurrency | Model | Output tok/s, mean ± SD | Speed ratio vs BF16 | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2E (ms) | Mean sampled run maximum (MiB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | BF16 | 322.66 ± 0.34 | 1.0000x | 186.62 | 11.02 | 1586.29 | 12911.00 |
| 4 | RandomChannels | 332.89 ± 0.19 | 1.0317x | 180.89 | 10.68 | 1537.54 | 12625.00 |
| 4 | ActivationEnergy | 333.07 ± 0.16 | 1.0322x | 179.39 | 10.69 | 1536.76 | 12625.00 |
| 1 | BF16 | 98.87 ± 0.08 | 1.0000x | 75.73 | 9.59 | 1293.86 | 12797.00 |
| 1 | RandomChannels | 101.89 ± 0.08 | 1.0306x | 74.87 | 9.30 | 1255.38 | 12511.67 |
| 1 | ActivationEnergy | 101.86 ± 0.07 | 1.0303x | 75.51 | 9.29 | 1255.74 | 12511.67 |

## Checkpoint size

| Model | MLP width | Unique parameters | Safetensors GiB | File saving vs BF16 (MiB) |
|---|---:|---:|---:|---:|
| BF16 | 9,728 | 4,022,468,096 | 7.4925 | 0.000 |
| RandomChannels | 9,216 | 3,880,910,336 | 7.2288 | 270.000 |
| ActivationEnergy | 9,216 | 3,880,910,336 | 7.2288 | 270.000 |

## All formal runs

| Order | Model | Rep | Concurrency | Output tok/s | TTFT (ms) | TPOT (ms) | E2E (ms) | Sampled maximum (MiB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | BF16 | 1 | 4 | 322.98 | 188.20 | 11.00 | 1584.70 | 12907 |
| 2 | BF16 | 1 | 1 | 98.96 | 75.63 | 9.58 | 1292.59 | 12907 |
| 3 | RandomChannels | 1 | 4 | 333.02 | 180.35 | 10.68 | 1536.95 | 12625 |
| 4 | RandomChannels | 1 | 1 | 101.98 | 74.66 | 9.29 | 1254.39 | 12625 |
| 5 | ActivationEnergy | 1 | 4 | 333.10 | 177.70 | 10.70 | 1536.57 | 12625 |
| 6 | ActivationEnergy | 1 | 1 | 101.88 | 75.14 | 9.29 | 1255.53 | 12625 |
| 7 | RandomChannels | 2 | 1 | 101.84 | 74.00 | 9.31 | 1256.10 | 12285 |
| 8 | RandomChannels | 2 | 4 | 332.68 | 181.82 | 10.68 | 1538.54 | 12625 |
| 9 | ActivationEnergy | 2 | 1 | 101.93 | 75.04 | 9.29 | 1255.05 | 12285 |
| 10 | ActivationEnergy | 2 | 4 | 332.89 | 180.28 | 10.69 | 1537.57 | 12625 |
| 11 | BF16 | 2 | 1 | 98.85 | 75.80 | 9.59 | 1294.16 | 12571 |
| 12 | BF16 | 2 | 4 | 322.29 | 187.17 | 11.03 | 1588.12 | 12913 |
| 13 | ActivationEnergy | 3 | 4 | 333.21 | 180.21 | 10.68 | 1536.14 | 12625 |
| 14 | ActivationEnergy | 3 | 1 | 101.79 | 76.34 | 9.29 | 1256.64 | 12625 |
| 15 | BF16 | 3 | 4 | 322.72 | 184.49 | 11.04 | 1586.05 | 12913 |
| 16 | BF16 | 3 | 1 | 98.80 | 75.75 | 9.60 | 1294.82 | 12913 |
| 17 | RandomChannels | 3 | 4 | 332.98 | 180.49 | 10.68 | 1537.14 | 12625 |
| 18 | RandomChannels | 3 | 1 | 101.86 | 75.96 | 9.29 | 1255.65 | 12625 |

## Scope and limits

- The server uses a fixed 4 GiB BF16 KV-cache budget, FlashAttention 2, context length 8,192, maximum four sequences, token batch limit 4,096 and disabled prefix caching.
- Mean and sample SD describe three repeated runs, not a confidence interval or a significance test. Throughput speed ratios divide arithmetic means. Model order rotates across repetitions.
- The JSON also summarizes run-level P95 latencies. Their mean is not a pooled P95 over all requests.
- These are warm local HTTP streaming measurements. Startup, model loading and compilation are outside the measured workload; client and HTTP overhead are included.
- Token lengths come from the native benchmark client. No latency-altering forwarding recorder was inserted. This audit does not independently inspect every wire request.
- GPU memory is whole-device usage sampled every second while the benchmark client runs, including warmup and the initial probe. Brief peaks can be missed. The reported cell value is the mean of three sampled run maxima, not model-weight memory or a guaranteed peak.
- Allocation retained from a preceding workload can affect memory. Concurrency 1 follows concurrency 4 in repetitions 1 and 3, and runs first in repetition 2, equally for each model. Read the individual values and ranges, not only the mean.
- One GPU, one pruning width and one random-channel seed were studied. Performance results do not establish model quality or generalize to other workloads automatically.
- Quality uses the separate, previously used development sets; its scores and uncertainty are in the development report. No official-test inference was added here.

The JSON contains all run-level metrics, file hashes, aggregation inputs and size calculations.
