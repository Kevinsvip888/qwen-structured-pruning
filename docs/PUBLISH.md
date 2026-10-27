# Publishing the reviewed package

This package is ready for local staging and review. It does not create a GitHub repository, upload weights or push changes automatically. The proposed standalone repository name is `qwen-structured-pruning-lab`; choose the actual repository name when creating it.

## Install and verify

Upload `pruning-github-public-files.zip` to `/root/autodl-tmp/qwen-pruning-lab`, then run:

```bash
cd /root/autodl-tmp/qwen-pruning-lab
unzip -n pruning-github-public-files.zip
python3 tools/verify_public_snapshot.py
```

The archive contains the original source, documentation and archived evidence. It does not contain live `data/`, `models/` or `outputs/` directories, so extraction keeps the working experiment separate. Existing source files are preserved by `unzip -n`; verification will flag a byte mismatch.

## Stage only reviewed paths

For a new repository:

```bash
git init -b main
git add .gitignore .gitattributes README.md docs environment provenance results scripts pruning_lab specs tools
git diff --cached --check
git diff --cached --stat
git status --short
```

The `.gitignore` excludes live weights, environments, logs, outputs, data, backups and key files. The archived evidence under `provenance/original/` is deliberately tracked. `.gitattributes` preserves original file bytes so source/protocol hashes survive Git checkout. Whitespace lint is disabled for the frozen original scripts and archived records because raw logs include padding/carriage returns and one frozen script has an extra final blank line; changing those bytes would invalidate their hashes. Added tools and documentation remain subject to the normal whitespace check.

After reviewing the staged files:

```bash
git commit -m "Publish structured Qwen MLP pruning and controlled evaluation"
```

Create an empty GitHub repository and configure its own authentication key or an existing authorized account. If using a repository deploy key, the keys for the earlier Qwen/Triton repositories do not automatically authorize a new repository. Never add a private key to Git.

Add the actual remote URL and push only after it is created and authentication is ready. No remote URL is assumed in these instructions. No software license is invented; select one explicitly if you intend to grant reuse rights.

## Fresh checkout check

Clone the published repository into a new directory and run:

```bash
python3 tools/verify_public_snapshot.py
git log -1 --oneline
git status --short
```

This validates the portable CPU snapshot. It does not constitute a new clean-environment GPU reproduction. Experimental replay has the additional dependencies and machine-specific pins described in [REPRODUCE.md](REPRODUCE.md).
