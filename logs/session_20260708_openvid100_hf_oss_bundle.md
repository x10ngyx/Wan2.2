# 2026-07-08 OpenVid-100 HF Dataset OSS Bundle

## Goal

Package `/hy-tmp/openvid_100_seacache_trace_data` for Hugging Face publication without exposing the original multi-machine experiment shard layout, and prepare an OSS transfer bundle plus instructions for uploading from another machine.

## Changes

- Added `scripts/hf_dataset_packaging/prepare_openvid100_hf_dataset.py`.
  - Builds a Hugging Face-ready staging layout under `/hy-tmp/hf_staging/wan22_openvid100_seacache_trace`.
  - Normalizes public paths to `prompt_001` through `prompt_100` and `threshold_0p10` through `threshold_0p80`.
  - Generates `prompts`, `candidates`, `videos`, and 50k step-level `train` indexes as Parquet plus JSONL.
  - Stages large media/trace/artifact files as symlinks so archives can dereference them without copying the full dataset twice.
- Added `scripts/hf_dataset_packaging/pack_openvid100_hf_dataset_for_oss.py`.
  - Writes `scripts/verify_dataset.py` and `HOW_TO_UPLOAD_TO_HUGGINGFACE.md` into the staging dataset.
  - Creates OSS transfer archives using normalized public paths and `tar --dereference`.
  - Generates `archive_manifest.json` and `checksums/archive_sha256s.txt`.
- Updated `PROGRESS.md` with this packaging result.

## Outputs

- Staging dataset: `/hy-tmp/hf_staging/wan22_openvid100_seacache_trace`
- OSS bundle: `/hy-tmp/oss_upload/wan22_openvid100_seacache_trace_20260708`
- Bundle size: about `134G`
- Archive count: `27`
- Upload instructions for the second machine:
  - `/hy-tmp/oss_upload/wan22_openvid100_seacache_trace_20260708/HOW_TO_UPLOAD_TO_HUGGINGFACE.md`
  - Also included inside `archives/repo_metadata_and_indexes.tar`

## Dataset Content

- `100` prompts.
- `1000` candidate inferences.
- `50000` step-level training rows:
  - one row per prompt/threshold/denoising step
  - includes reuse/recompute decision, branch keys, cache metrics, raw latent `.pt` path, baseline/candidate video paths, logs, PSNR, elapsed time, and speedup
- Raw latent tensors are referenced as normalized paths such as:
  - `traces/seacache/threshold_0p10/prompt_001/step_000.pt`
  - the tensor key is `latent`
- Baseline and candidate media are normalized under:
  - `media/baseline/prompt_001.mp4`
  - `media/seacache/threshold_0p10/prompt_001.mp4`

## Validation

Staging verification passed:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  /hy-tmp/hf_staging/wan22_openvid100_seacache_trace/scripts/verify_dataset.py \
  /hy-tmp/hf_staging/wan22_openvid100_seacache_trace
```

Result:

- prompts: OK
- candidates: OK
- train rows: `50000`
- key baseline media, SeaCache media, and trace path checks: OK

Archive checksum validation passed:

```bash
cd /hy-tmp/oss_upload/wan22_openvid100_seacache_trace_20260708
sha256sum -c checksums/archive_sha256s.txt
```

All `27` archives reported `OK`.

## Notes

- One source PSNR text log is missing/NaN in the original summary for `prompt_034`, `threshold_0p10`: `artifacts/psnr/threshold_0p10/prompt_034.log`. The PSNR JSON and summary metrics are present, so the candidate and training rows remain usable.
- OSS upload was not run in this session because this machine does not currently have `ossutil`/`ossutil64` configured and no bucket/endpoint/prefix was provided.
- The 124M pooled feature cache was intentionally not included, per user confirmation.
