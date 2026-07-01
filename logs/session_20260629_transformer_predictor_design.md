# 2026-06-29 Transformer Predictor Design

- User asked whether a DiT-like Transformer predictor is reasonable for the adaptive threshold network and then requested a concrete architecture/hyperparameter report.
- Reviewed Wan2.2 latent tokenization in `wan/modules/model.py` and T2V latent shape logic in `wan/text2video.py`.
- Recommended a lightweight MiniDiT-style predictor with CLS readout:
  - patch size `(3, 12, 8)`
  - token grid `[4, 5, 13]`
  - `260` latent tokens plus CLS
  - `d_model=96`, `2` layers, `4` heads
  - factorized learned 3D positional embeddings
  - AdaLN-style conditioning on step fraction and target PSNR
  - threshold output constrained to `[0.10, 0.80]`
- Added report: `reports/report_transformer_predictor_architecture.md`.
- Updated `PROGRESS.md` with the design decision and recommended first configuration.

Validation:
- No training or inference was run. This session only created design documentation.
