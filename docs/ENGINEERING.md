# Engineering notes

## Observe before modifying

The collector observes the input to each original `down_proj` through pre-hooks. It does not prune one layer and then collect later-layer statistics from the changed model. An exact hooked/unhooked backbone probe checks that observation has not changed outputs. Parameter metadata and source-file hashes are checked around collection.

Each calibration batch computes channel squared-activation reductions in FP32, accumulates in FP64, and forms scores with the down-projection column squared norms. The 98.76% median retained score mass for ActivationEnergy, versus 94.80% for RandomChannels, is a proxy diagnostic. It is not retained accuracy or a guarantee about model output.

## Compact tensors

For retained indices `I`, the compact weights are:

```text
gate_new = gate_original[I, :]
up_new   = up_original[I, :]
down_new = down_original[:, I]
```

The helper returns a new MLP with new parameter storage and a copied configuration. It validates bias-free compatible projections and sorted unique indices. It does not mutate the supplied MLP. The caller replaces all 36 MLPs and updates the model's global intermediate size.

The mathematical reference masks removed channels in the original-width MLP. FP64 tests compare that result with the compact MLP. Native BF16 GEMM shapes change after compaction, so native outputs are checked for finite valid behavior rather than required to be bitwise equal to a differently shaped masked GEMM.

## Export and reload

Each method starts from a fresh original BF16 model. Export audits the prescribed slices for 108 MLP tensors and verifies that 290 other tensors are unchanged. Reload checks compare all 398 loaded unique parameter tensors with the saved checkpoint, preserve tied embedding/LM-head storage, and require exact pre/post-export hidden-state and last-token-logit probes.

Temporary build directories are renamed only after checks pass. Existing checkpoints are not silently replaced. Tokenizer and generation files are preserved. No recovery training, distillation, quantization or sparse-kernel representation is used.

## Quality and performance

Quality uses a recording proxy to validate frozen prompts, prompt-token counts and returned science token likelihoods against the harness. The two GSM8K filter records per question are checked separately. The recorder handles the installed harness's nested string-serialized likelihood pairs. The public archive omits those raw quality records; it preserves audited summaries and primary score vectors.

Performance uses the native vLLM streaming benchmark directly, with no forwarding recorder in the timed path. All methods share the tokenizer, random workload seed, fixed output length, server settings and physical GPU. A Latin-square model order reduces position effects; workload order also rotates. The runner rejects an out-of-order formal run or an existing output directory.

The two pruned models have identical dimensions, so a similar serving speed is expected as a hypothesis. Their measured quality can differ because they retain different channels. This study measures both instead of inferring task quality from retained score mass or speed from parameter count.

Memory is sampled whole-device usage, not allocator-level model memory. Allocations retained after concurrency 4 explain why concurrency-1 readings must be reported with their workload order and individual repetitions. Keep checkpoint storage, theoretical parameter payload and sampled GPU memory as separate quantities.
