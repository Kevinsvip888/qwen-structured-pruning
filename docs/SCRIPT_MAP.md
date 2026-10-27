# Script map

All experimental paths below are relative to `/root/autodl-tmp/qwen-pruning-lab`, with dependencies on `/root/autodl-tmp/qwen-inference-lab`. Original files are unchanged.

| Script | Purpose | Main output |
|---|---|---|
| `prepare_pruning_study.py` | Validate BF16 source and reserved math split; freeze width, selection rules and calibration identities | `data/protocol_v1.json` |
| `collect_channel_importance.py` | Observe original-model activations and freeze both selections | `outputs/importance_v1/` and pointer |
| `validate_mlp_surgery.py` | Toy/real FP64 comparisons, BF16 smokes and rejection tests | `outputs/surgery_validation_v1/` and pointer |
| `export_pruned_checkpoint.py` | Compact and audit one BF16 checkpoint, then reload it | `models/…-MLP9216-METHOD/` and export receipt |
| `prepare_pruning_quality.py` | Pin three checkpoints and existing development prompts/scoring | `data/evaluation/pruning_dev_v1/protocol.json` |
| `pruning_eval_recording.py` | Recording HTTP server and saved quality auditor, imported by the runner | Per-request JSON and `quality_summary.json` |
| `run_pruning_quality.py` | Pilot/full development inference or saved-run audit | `outputs/evaluation/pruning_dev_v1/METHOD/` |
| `summarize_pruning_development.py` | Check full-run identities and compare primary score vectors | Development summary/report |
| `prepare_pruning_benchmarks.py` | Freeze the shared native benchmark and 18-run order | `data/benchmark/pruning_v1/protocol.json` |
| `run_pruning_benchmark.py` | One pilot/formal run with memory samples and completion receipt | `outputs/benchmark/pruning_v1/` |
| `summarize_pruning_benchmarks.py` | Verify 18 runs and recompute means, SDs and ratios | Benchmark summary/report |

`pruning_lab/surgery.py` contains the shared compact-MLP helper; `pruning_lab/__init__.py` defines the package.

Added publication tools:

- `tools/verify_public_snapshot.py`: portable CPU verification of included records and derived arithmetic.
- `tools/serve_model.sh`: convenience wrapper for the already recorded server settings. It does not change the experimental source or frozen protocols.

Use `--help` on the original runners to inspect their options before invoking an expensive stage. Do not restore archived completion pointers into live runtime directories: rerun the real stages to create genuine receipts.
