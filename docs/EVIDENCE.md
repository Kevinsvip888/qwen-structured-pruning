# Evidence and verification limits

The source archive supplied 294 files and an inventory. Every supplied file hash was checked before packaging. Original bytes are retained under their source path for code/specification files and under `provenance/original/` for runtime records. The public inventory also covers the added documentation and tools.

## Included evidence

| Component | Included records | What can be checked on CPU |
|---|---|---|
| Source | 13 original Python files and validation specification | Syntax and exact original hashes |
| Calibration observation | Frozen protocol, source tensor inventory, collector manifest, pass records, two channel selections | Identities, coverage counts and random-channel reconstruction |
| Surgery | Validation specification and detailed recorded receipt | Check counts, tolerances, recorded errors and validated source hashes |
| Export | Two manifests and export receipts | Retained-index agreement, parameter arithmetic and recorded reload checks |
| Development quality | Protocol/task YAML, three run manifests and audited summaries, combined report and primary per-question score vectors | Counts, percentages, paired gains/losses and exact McNemar p-values |
| Serving | Frozen protocol and all 18 formal run folders | Receipt hashes, CLI arguments, startup settings, recorded lengths, GPU CSV maxima and mean/SD/ratio arithmetic |

The archive contains complete native benchmark records, including client logs, generated text, memory samples and server-log snapshots. Published performance values are recomputed from these records. No additional profiling or GPU execution occurs during publication verification.

## Omitted evidence

Model weights, activation-statistics safetensors, original Project 1 data/tokenizer assets, virtual environments, raw quality API responses and per-filter harness sample files are omitted. Their recorded hashes remain as provenance references.

Consequently, the snapshot verifier does not independently execute or re-audit weight slicing/reloading, activation-energy ranking, numerical FP64/BF16 validation, quality extraction, token likelihood reconstruction or bootstrap draws. Those are recorded results from the original experiment. Public quality score-vector arithmetic can be checked, but its correctness cannot be established from the omitted raw generations alone.

The source safetensors sizes and exported parameter counts are recorded measurements. The public snapshot checks their consistency and unit conversions; it cannot remeasure omitted weight files. Raw benchmark token lengths are client-recorded, not a new packet-level audit of every request.

## Interpretation

- This is a pilot with one width and one random-channel seed. It is not evidence across pruning rates or random seeds.
- The quality sets were used in Project 1 and are development data. No new official-test evaluation was performed for pruning.
- Question-bootstrap intervals are conditional on these fixed checkpoints. Six paired task/comparison results are exploratory and unadjusted for multiplicity.
- Failure to detect a difference from BF16 is not proof of equivalent quality.
- Benchmark repetitions show local variation on one GPU. Their SD is not a confidence interval over machines, model seeds or production traffic.
- Startup and compilation are outside warmed throughput measurements. Sampled GPU maxima include warmup, can miss brief peaks and depend on preceding allocations.

The earlier FP8 project used other serving experiments. Its historical throughput numbers are not substituted for the fresh BF16 baseline here.
