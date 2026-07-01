# 2026-06-29 Transformer Predictor Conv3d Patch Revision

- User requested using the report's Conv3d patch embedding implementation and not focusing on control groups for now.
- Revised `MiniDiTCLSAdaptiveThresholdPredictor`:
  - now consumes raw latent `[B,16,12,60,104]`
  - uses learnable `Conv3d(16, d_model, kernel_size=(3,12,8), stride=(3,12,8))`
  - produces token grid `[4,5,13]` and 260 latent tokens before CLS
  - retains factorized learned 3D position embeddings, CLS readout, and AdaLN-style conditioning.
- Revised `train_gate.py`:
  - `--model_type mini_dit_cls` uses `TraceStepThresholdDataset` directly
  - no longer requires `--grid_cache_dir`
  - checkpoint payload now includes self-contained `feature_extractor` metadata
  - shared defaults now match recommended MiniDiT training settings.
- Updated documentation:
  - `adaptive_threshold_predictor/README.md`
  - `reports/report_transformer_predictor_architecture.md`
- Validation:
  - `python -m py_compile adaptive_threshold_predictor/models.py adaptive_threshold_predictor/train_gate.py`
  - CPU smoke train from raw latent:
    `/hy-tmp/wan22_mini_dit_convpatch_metadata_smoke_20260629`
  - Confirmed checkpoint feature extractor metadata:
    `type=learned_conv3d_patch_embedding`, `input_shape=[16,12,60,104]`,
    `patch_size=[3,12,8]`, `token_grid_shape=[4,5,13]`, `token_count=260`.
- No full training run was launched.
