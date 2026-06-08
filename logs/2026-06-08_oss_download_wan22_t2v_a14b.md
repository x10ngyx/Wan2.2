# 2026-06-08 OSS Download Wan2.2 T2V-A14B

- User requested cleanup of incomplete local Wan2.2 T2V-A14B model artifacts and a fresh OSS download.
- Confirmed no leftover OSS transfer process was running.
- Removed incomplete local paths:
  - `/hy-tmp/models/Wan2.2-T2V-A14B`
  - `/hy-tmp/models/Wan2.2-T2V-A14B.tar`
  - `/hy-tmp/Wan2.2-T2V-A14B.tar`
- OSS object used: `oss://datasets/Wan2.2-T2V-A14B.tar`.
- Download target: `/hy-tmp/models/Wan2.2-T2V-A14B.tar`.
- Download command used temp files under `/hy-tmp/oss-tmp` and parallelism `-p=8`; OSS reported success for `117.53GB` in about `30m21s`.
- Extracted with `tar -xf /hy-tmp/models/Wan2.2-T2V-A14B.tar -C /hy-tmp/models`.
- Verification:
  - Archive manifest non-directory entries: `31`.
  - Extracted file count: `31`.
  - Found all 12 diffusion safetensor shards: 6 high-noise and 6 low-noise.
  - Key files present: `configuration.json`, `Wan2.1_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`.
  - Extracted directory size: `118G`; archive size: `118G`; remaining `/hy-tmp` space: about `154G`.
- Password was not recorded in logs.
