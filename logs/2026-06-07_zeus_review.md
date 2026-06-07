# 2026-06-07 ZEUS Review

- User asked to read the ZEUS repository and the "ZEUS: Zero-shot Efficient Unified Sparsity" paper.
- Read local `PROGRESS.md` as required by project workflow.
- Cloned `https://github.com/Ting-Justin-Jiang/ZEUS` to `/hy-tmp/ZEUS` and reviewed commit `ceff240 (Hello ZEUS)`.
- Checked the ZEUS project page at `https://yixiao-wang-stats.github.io/zeus/`.
- Public paper PDF/arXiv was not available; project page shows `Paper (Coming Soon)`.
- Key files reviewed:
  - `README.md`
  - `zeus/patch.py`
  - `zeus/model.py`
  - `zeus/solver.py`
  - `zeus/module.py`
  - `wan2_demo.py`
  - `generate_video.py`
  - `generate_dit.py`
- Main finding: ZEUS is a fixed-pattern denoiser step-skipping method using Diffusers monkey patches, second-order extrapolation / reuse pattern, optional Lagrange interpolation, and scheduler-level reconstruction. Wan support is for Diffusers `WanPipeline` / `WanTransformer3DModel`, not this repo's native Wan2.2 pipeline.
