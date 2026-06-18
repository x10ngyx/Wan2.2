# Quick Start For A Fresh Remote Machine

This guide is for a new machine with no prepared Python environment. It uses:

- source code from GitHub
- packed runtime from OSS
- Wan2.2 T2V-A14B weights from OSS or another mounted location

## 1. Prepare Directories

Use `/hy-tmp` for code, environment, caches, weights, and experiment outputs.
Avoid putting large files under `/root` or the system disk.

```bash
mkdir -p /hy-tmp/work /hy-tmp/models /hy-tmp/env /hy-tmp/hf-cache
cd /hy-tmp/work
```

## 2. Pull Code

```bash
git clone https://github.com/x10ngyx/Wan2.2.git
cd Wan2.2
```

For later updates:

```bash
cd /hy-tmp/work/Wan2.2
git pull
```

## 3. Download And Restore Runtime

The packed runtime is stored in OSS:

```text
oss://datasets/Wan2.2-conda-env.tar.gz
```

Known SHA256:

```text
348f63583d2a3ea742b80341dbb97043c6a497065e593a1329b1aad1a0551f03
```

Download and unpack it:

```bash
oss cp oss://datasets/Wan2.2-conda-env.tar.gz /hy-tmp/env/Wan2.2-conda-env.tar.gz
sha256sum /hy-tmp/env/Wan2.2-conda-env.tar.gz

mkdir -p /hy-tmp/env/Wan2.2
tar -xzf /hy-tmp/env/Wan2.2-conda-env.tar.gz -C /hy-tmp/env/Wan2.2

/hy-tmp/env/Wan2.2/bin/conda-unpack || true
```

Use the environment directly:

```bash
export PATH=/hy-tmp/env/Wan2.2/bin:$PATH
export HF_HOME=/hy-tmp/hf-cache
export TRANSFORMERS_CACHE=/hy-tmp/hf-cache
export HF_HUB_CACHE=/hy-tmp/hf-cache/hub
```

Optional shell persistence:

```bash
cat >> ~/.bashrc <<'EOF'
export PATH=/hy-tmp/env/Wan2.2/bin:$PATH
export HF_HOME=/hy-tmp/hf-cache
export TRANSFORMERS_CACHE=/hy-tmp/hf-cache
export HF_HUB_CACHE=/hy-tmp/hf-cache/hub
EOF
```

## 4. Download Or Mount Model Weights

The T2V-A14B weights should be available at:

```text
/hy-tmp/models/Wan2.2-T2V-A14B
```

If the weights are stored in OSS, download them into that directory. The exact
OSS path depends on the uploaded model artifact. After download, the directory
should contain files and subdirectories like:

```text
configuration.json
high_noise_model/
low_noise_model/
Wan2.1_VAE.pth
models_t5_umt5-xxl-enc-bf16.pth
```

Expected extracted size is about `118G`.

If the weights are placed elsewhere, pass that path to experiment scripts with
their checkpoint argument or environment variable.

## 5. Basic Validation

Confirm the GPU mode before running T2V generation:

```bash
nvidia-smi
```

Check code import syntax and model files:

```bash
cd /hy-tmp/work/Wan2.2
python -m py_compile generate.py wan/text2video.py wan/modules/model.py
ls -lah /hy-tmp/models/Wan2.2-T2V-A14B
```

Read the project handoff state before launching experiments:

```bash
sed -n '1,220p' PROGRESS.md
```

## 6. Default T2V Experiment Settings

Unless a task says otherwise, use:

```text
task/model: t2v-A14B
checkpoint: /hy-tmp/models/Wan2.2-T2V-A14B
seed: 42
size: 832*480
frame_num: 45
sample_steps: 50
sample_solver: dpm++
offload: --offload_model
dtype conversion: --convert_model_dtype
```

Cache CLI aliases currently used by this project:

```text
--timestep_cache zeus-threshold --timestep_threshold <float>
--block_cache block-group --block_threshold <float>
--cfg_cache threshold --cfg_threshold <float>
```

Prefer single-process batch runners for threshold sweeps so a process loads the
pipeline/checkpoint once and then runs candidates sequentially.

## 7. Push Results Back To GitHub

Commit code, prompt-set, report, and metadata changes. Do not commit model
weights, generated videos, or large experiment archives.

```bash
cd /hy-tmp/work/Wan2.2
git status --short
git add <files>
git commit -m "<message>"
git push x10ngyx main
```

Large experiment outputs should stay under `/hy-tmp` and can be uploaded to OSS
separately when needed.
