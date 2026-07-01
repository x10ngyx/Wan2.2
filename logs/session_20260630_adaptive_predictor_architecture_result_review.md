# Session Log: Adaptive Predictor Architecture/Result Review

Date: 2026-06-30

## Scope

- Checked `adaptive_threshold_predictor/` MiniDiT-CLS Transformer and 5-feature gated MLP implementation against:
  - `reports/report_transformer_predictor_architecture.md`
  - `reports/report_gated_multifeature_mlp_architecture.md`
- Reviewed training configs, metrics, checkpoints, and validation predictions under:
  - `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_20260629_214241`
  - `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906`
  - `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659`
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_rowsplit_gpu_20260630_021641`
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_rowsplit_gpu_long100_20260630_021641`
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_samplesplit_20260630_021641`

## Findings

- MiniDiT architecture mostly matches the report: raw latent input, learned Conv3d patch embedding `(3,12,8)`, token grid `[4,5,13]`, 260 latent tokens, CLS readout, factorized learned 3D position embedding, 2 conditioned Transformer blocks, and threshold output range `[0.10,0.80]`.
- 5-feature gated MLP architecture matches the report: five pooled feature tensors, separate feature encoders, condition embedding, condition-dependent softmax gate, gated feature sum, and fused prediction head. Parameter count is 83,526 as expected.
- MiniDiT zero gate initialization creates a cold-start behavior: first backward pass gives zero gradient to patch embedding, attention, block MLP, and condition embedding. Only CLS/head and modulation parameters receive gradient initially. Saved checkpoints show those modules eventually move, so this is not a fatal training failure, but it should be ablated with a small nonzero `--dit_gate_init`.
- Row split results are much better than sample split because train and validation both include all 100 sample IDs. Treat row split as a memorization/capacity diagnostic, not as held-out prompt generalization.
- The first MiniDiT directory has no metrics/checkpoints and should be excluded from comparisons.

## Key Metrics Observed

- MiniDiT sample split, raw latent, batch 128: best epoch 4, best validation MAE `0.114459`, early stopped at epoch 9.
- MiniDiT row split, packed raw latent, batch 128: best epoch 29, best validation MAE `0.038002`.
- Gated MLP sample split: best epoch 7, best validation MAE `0.114253`, early stopped at epoch 12.
- Gated MLP row split 30 epochs: best epoch 30, best validation MAE `0.075670`.
- Gated MLP row split 100 epochs: best epoch 98, best validation MAE `0.060112`.

## Files Changed

- Updated `PROGRESS.md` with this review summary.
- Added this session log.

## Validation

- No code changes were made.
- Ran file/config/metric inspection commands and a MiniDiT gradient sanity check.
