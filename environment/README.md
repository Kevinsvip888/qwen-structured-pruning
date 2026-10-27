# Environment record

The recorded serving environment uses Python 3.12.15, PyTorch 2.13.0+cu132, Triton 3.7.1, Transformers 5.19.0, vLLM 0.32.0 and safetensors 0.8.0. The full freeze is [requirements-experiment.txt](requirements-experiment.txt).

Quality runs use the separate Project 1 `.venv-eval`: lm_eval 0.4.13, datasets 5.1.0, Transformers 5.19.0 and huggingface_hub 2.3.0. PyTorch is not required in that client environment because inference occurs in vLLM. Its installed harness source is hash-pinned by Project 1 preparation.

The GPU is an RTX 4090 with driver 580.76.05. The recorded protocols identify the physical GPU by UUID. The original source and calibration observation use the serving environment, not the separate older quantization environment from the FP8 study.

The freeze is a record, not a guarantee of future wheel availability or a portable lockfile. CUDA wheel selection and access to upstream datasets/models must be handled when reconstructing an environment. The original scripts reject changed package/source/GPU identities. See [reproduction instructions](../docs/REPRODUCE.md) before attempting replay.
