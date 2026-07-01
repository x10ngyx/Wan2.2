# MiniDiT-CLS Transformer Predictor Comprehensive Report

Date: 2026-06-30

## 1. Scope

This report summarizes the current Transformer-style adaptive SeaCache threshold predictor:

- architecture and module parameter settings;
- training configuration and offline metrics for sample split and row split checkpoints;
- training loss curves;
- online inference settings and real T2V performance on VBench10 and OpenVid100 train prompts.

The predictor checkpoint output is a per-step SeaCache threshold in `[0.10, 0.80]`. It is evaluated here as an online adaptive SeaCache gate inside Wan2.2 T2V-14B inference.

## 2. Architecture

![MiniDiT-CLS predictor architecture](assets/mini_dit_cls_predictor_architecture.svg)

### 2.1 High-Level Flow

```text
latent [B, 16, 12, 60, 104]
  -> Conv3d patch embedding
  -> latent tokens [B, 260, 96]
  -> prepend CLS token
  -> add learned factorized 3D positional embeddings
  -> 2 AdaLN-conditioned Transformer blocks
  -> CLS readout head
  -> scalar threshold in [0.10, 0.80]
```

### 2.2 Input And Tokenization

| Item | Value |
|---|---:|
| model class | `MiniDiTCLSAdaptiveThresholdPredictor` |
| implementation | `adaptive_threshold_predictor/models.py` |
| input layout | `[B, C, T, H, W]` |
| input latent shape | `[16, 12, 60, 104]` |
| patch size | `(3, 12, 8)` |
| patch embedding | `Conv3d(16, 96, kernel_size=(3,12,8), stride=(3,12,8))` |
| token grid | `[4, 5, 13]` |
| latent token count | `260` |
| CLS token | yes |
| sequence length after CLS | `261` |
| runtime dtype in feature extractor metadata | `float32` |

### 2.3 Transformer And Conditioning

| Item | Value |
|---|---:|
| model dimension | `96` |
| layers | `2` |
| attention heads | `4` |
| attention head dim | `24` |
| MLP ratio | `2.0` |
| MLP hidden dim | `192` |
| attention module | PyTorch `nn.MultiheadAttention(batch_first=True)` |
| attention dropout | `0.05` |
| MLP dropout | `0.05` |
| positional dropout | `0.05` |
| normalization in blocks | `LayerNorm(elementwise_affine=False)` |
| readout normalization | `LayerNorm(elementwise_affine=True)` |
| block activation | `GELU` |
| condition activation | `SiLU` |
| head activation | `SiLU` |
| condition input | `[step_fraction, normalized_target_psnr]` |
| PSNR normalization | `(target_psnr - 10) / (50 - 10)`, clamped to `[0, 1]` |
| condition embedder | `Linear(2,96) -> SiLU -> Linear(96,96) -> SiLU` |
| block conditioning | AdaLN-style shift/scale/gate for attention and MLP |
| modulation output per block | `Linear(96, 576)` = 6 vectors of dim 96 |
| `dit_gate_init` | `0.0` |
| output mapping | `0.10 + sigmoid(raw) * 0.70` |

Important implementation detail: with `dit_gate_init=0.0`, each Transformer block starts with zero attention/MLP residual gates. This stabilizes the initial function but means the earliest gradient flow into patch embedding and block internals is delayed until modulation gates move away from zero.

### 2.4 Positional Encoding

The predictor uses learned factorized 3D positional embeddings:

```text
pos_t: [4, 96]
pos_h: [5, 96]
pos_w: [13, 96]
pos[f,h,w] = pos_t[f] + pos_h[h] + pos_w[w]
```

The flattened order follows PyTorch Conv3d output order:

```text
token_index = (t * H_grid + h) * W_grid + w
```

This is deliberately simpler than Wan2.2 3D RoPE. The current experiment uses a fixed latent shape, so learned factorized embeddings are sufficient for this first Transformer predictor.

### 2.5 Parameter Count

Total trainable parameters: `724,513`.

| Module | Parameters |
|---|---:|
| Conv3d patch embedding | `442,464` |
| CLS token + CLS position | `192` |
| factorized 3D position embeddings | `2,112` |
| condition embedder | `9,600` |
| Transformer blocks, total | `260,544` |
| readout head | `9,601` |
| total | `724,513` |

Per Transformer block:

| Block submodule | Parameters |
|---|---:|
| attention | `37,248` |
| MLP | `37,152` |
| AdaLN modulation | `55,872` |
| total per block | `130,272` |

## 3. Training Setup

Two MiniDiT checkpoints were trained and then evaluated online:

| Split | Checkpoint | Split meaning |
|---|---|---|
| sample split | `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906/best_model_checkpoint.pt` | train and validation/test are split by `sample_id`; this is the stricter generalization test |
| row split | `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659/best_model_checkpoint.pt` | train and validation/test are random row split; all 100 sample IDs appear in both train and validation/test |

The first incomplete MiniDiT run `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_20260629_214241` has no `metrics.json` or checkpoint and is excluded.

### 3.1 Shared Training Parameters

| Item | Value |
|---|---:|
| script | `python -m adaptive_threshold_predictor.train_gate` |
| dataset mode | `candidate_inverse` |
| model type | `mini_dit_cls` |
| examples | `50,000` |
| train fraction | `0.8` |
| train rows | `40,000` |
| validation/test rows | `10,000` |
| batch size | `128` |
| max epochs | `30` |
| optimizer LR | `3e-4` |
| minimum LR | `1e-5` |
| warmup steps | `500` |
| weight decay | `1e-4` |
| loss | Smooth L1 |
| Smooth L1 beta | `0.02` |
| grad clip | `1.0` |
| early-stop patience | `5` |
| split seed | `42` |
| threshold range | `[0.10, 0.80]` |
| PSNR normalization range | `[10, 50]` |

### 3.2 Split-Specific Training Parameters

| Item | sample split | row split |
|---|---:|---:|
| split key | `group_by_sample_id` | `row` |
| train sample IDs | `80` | `100` |
| validation/test sample IDs | `20` | `100` |
| input mode | `raw_latent` | `packed_raw_latent` |
| packed latent cache | none | `/hy-tmp/wan22_adaptive_threshold_raw_latent_packed_cache_candidate_inverse_fp16_20260629_221805` |
| preload packed latents | no | yes |
| num workers | `8` | `0` |
| early stopped | yes | no |
| stopped epoch | `9` | n/a |
| best epoch | `4` | `29` |
| best validation/test MAE | `0.114459` | `0.038002` |

Interpretation: the row split offline metric is much easier because validation/test rows share all source videos with training. It measures interpolation over rows/steps/threshold labels, not sample-level generalization.

## 4. Training Loss Curves

Train and validation/test loss are overlaid within each split. The two splits are separated into subplots because the row-split run has 30 epochs while the sample-split run early-stopped after 9 epochs.

![MiniDiT-CLS training loss curves](assets/mini_dit_cls_training_loss_curves.svg)

### 4.1 Best-Epoch Metrics

| Split | Best epoch | Train loss | Train MAE | Val/test loss | Val/test MAE | Val/test bias |
|---|---:|---:|---:|---:|---:|---:|
| sample split | 4 | 0.067382 | 0.076298 | 0.105273 | 0.114459 | -0.008515 |
| row split | 29 | 0.032757 | 0.040767 | 0.030339 | 0.038002 | -0.002942 |

### 4.2 Last-Epoch Metrics

| Split | Last epoch | Train loss | Train MAE | Val/test loss | Val/test MAE | Val/test bias |
|---|---:|---:|---:|---:|---:|---:|
| sample split | 9 | 0.051269 | 0.059861 | 0.118137 | 0.127152 | -0.016222 |
| row split | 30 | 0.032527 | 0.040503 | 0.030355 | 0.038097 | +0.000394 |

For sample split, training loss continues to improve while validation/test loss worsens after the best epoch, so early stopping is appropriate. For row split, train and validation/test curves track closely, consistent with the easier row-level interpolation setting.

## 5. Online Inference Settings

Online inference used the same Wan2.2 T2V-14B generation defaults as the cache experiments:

| Item | Value |
|---|---:|
| task | `t2v-A14B` |
| checkpoint | `/hy-tmp/models/Wan2.2-T2V-A14B` |
| size | `832*480` |
| frame count | `45` |
| sample steps | `50` |
| solver | `dpm++` |
| seed | `42` |
| offload | enabled |
| model dtype conversion | enabled |
| baseline policy | reuse existing no-cache baseline |
| quality metric | FFmpeg PSNR against same prompt/seed/shape baseline |
| speed metric | `inference_compute_elapsed_seconds` |
| runner | `experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_batch.py` |
| result root | `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328` |

Candidate grid:

```text
2 splits * 2 target PSNRs * 2 datasets * 3 prompts = 24 candidates
```

| Axis | Values |
|---|---|
| model split | `sample_split`, `row_split` |
| target PSNR | `22`, `28` |
| VBench10 prompts | `vbench10_001`, `vbench10_002`, `vbench10_003` |
| OpenVid train prompts | `openvid_002`, `openvid_004`, `openvid_005` |

Batch-runner behavior:

- one WanT2V pipeline load for the whole run;
- baseline videos reused, not regenerated;
- one fresh adaptive SeaCache factory per candidate;
- cache runtime state cleared after each candidate;
- `wan.text2video.SeaCacheTimestepCache` restored after each candidate;
- `torch.cuda.empty_cache()` called after cleanup.

The online result table includes `predictor_call_count` fields, but this run recorded `0` calls and blank predictor elapsed fields. Therefore predictor overhead is not reported here; online performance is based on T2V compute elapsed time and PSNR.

## 6. Online Results: Per Prompt

| dataset | prompt | split | target | speedup | PSNR | target error | reuse | threshold mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vbench10 | vbench10_001 | sample_split | 22 | 1.895 | 14.928 | -7.072 | 50 | 0.273 |
| vbench10 | vbench10_001 | sample_split | 28 | 1.624 | 20.288 | -7.712 | 40 | 0.204 |
| vbench10 | vbench10_001 | row_split | 22 | 1.904 | 20.087 | -1.913 | 50 | 0.274 |
| vbench10 | vbench10_001 | row_split | 28 | 1.579 | 20.520 | -7.480 | 38 | 0.182 |
| vbench10 | vbench10_002 | sample_split | 22 | 2.619 | 19.902 | -2.098 | 66 | 0.486 |
| vbench10 | vbench10_002 | sample_split | 28 | 1.841 | 30.840 | +2.840 | 48 | 0.247 |
| vbench10 | vbench10_002 | row_split | 22 | 2.497 | 20.747 | -1.253 | 64 | 0.442 |
| vbench10 | vbench10_002 | row_split | 28 | 1.630 | 31.898 | +3.898 | 40 | 0.197 |
| vbench10 | vbench10_003 | sample_split | 22 | 1.960 | 15.380 | -6.620 | 54 | 0.317 |
| vbench10 | vbench10_003 | sample_split | 28 | 1.356 | 20.260 | -7.740 | 38 | 0.182 |
| vbench10 | vbench10_003 | row_split | 22 | 1.906 | 20.564 | -1.436 | 50 | 0.271 |
| vbench10 | vbench10_003 | row_split | 28 | 1.423 | 23.989 | -4.011 | 30 | 0.155 |
| openvid100_train | openvid_002 | sample_split | 22 | 2.484 | 20.213 | -1.787 | 64 | 0.444 |
| openvid100_train | openvid_002 | sample_split | 28 | 1.722 | 27.956 | -0.044 | 44 | 0.217 |
| openvid100_train | openvid_002 | row_split | 22 | 2.281 | 23.404 | +1.404 | 60 | 0.379 |
| openvid100_train | openvid_002 | row_split | 28 | 1.530 | 28.916 | +0.916 | 36 | 0.193 |
| openvid100_train | openvid_004 | sample_split | 22 | 3.022 | 24.902 | +2.902 | 72 | 0.641 |
| openvid100_train | openvid_004 | sample_split | 28 | 2.104 | 33.996 | +5.996 | 56 | 0.338 |
| openvid100_train | openvid_004 | row_split | 22 | 3.194 | 24.160 | +2.160 | 74 | 0.743 |
| openvid100_train | openvid_004 | row_split | 28 | 2.191 | 28.196 | +0.196 | 58 | 0.346 |
| openvid100_train | openvid_005 | sample_split | 22 | 2.375 | 21.339 | -0.661 | 62 | 0.375 |
| openvid100_train | openvid_005 | sample_split | 28 | 1.623 | 25.104 | -2.896 | 40 | 0.203 |
| openvid100_train | openvid_005 | row_split | 22 | 2.108 | 21.457 | -0.543 | 56 | 0.331 |
| openvid100_train | openvid_005 | row_split | 28 | 1.376 | 26.018 | -1.982 | 28 | 0.156 |

## 7. Online Results: Aggregate

| dataset | split | target | n | speedup | mean PSNR | target error | mean reuse | mean threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| openvid100_train | row_split | 22 | 3 | 2.447 | 23.007 | +1.007 | 63.3 | 0.484 |
| openvid100_train | row_split | 28 | 3 | 1.633 | 27.710 | -0.290 | 40.7 | 0.232 |
| openvid100_train | sample_split | 22 | 3 | 2.598 | 22.151 | +0.151 | 66.0 | 0.487 |
| openvid100_train | sample_split | 28 | 3 | 1.794 | 29.019 | +1.019 | 46.7 | 0.252 |
| vbench10 | row_split | 22 | 3 | 2.068 | 20.466 | -1.534 | 54.7 | 0.329 |
| vbench10 | row_split | 28 | 3 | 1.539 | 25.469 | -2.531 | 36.0 | 0.178 |
| vbench10 | sample_split | 22 | 3 | 2.113 | 16.737 | -5.263 | 56.7 | 0.359 |
| vbench10 | sample_split | 28 | 3 | 1.582 | 23.796 | -4.204 | 42.0 | 0.211 |

## 8. Findings

1. Offline row-split metrics are much better than sample-split metrics, but this is expected because row split leaks source-video identity across train and validation/test rows.

2. In online VBench10 inference, row split is still better calibrated than sample split:
   - target 22 absolute error improves from `5.263 dB` to `1.534 dB`;
   - target 28 absolute error improves from `4.204 dB` to `2.531 dB`;
   - speedup is only slightly lower, around `0.04x`.

3. On OpenVid train prompts, calibration is mixed:
   - sample split is closer at target 22;
   - row split is closer at target 28;
   - row split is slower by roughly `0.15x-0.16x`.

4. Both MiniDiT splits undershoot VBench10 target PSNR on average. Row split undershoots less, but it does not yet solve target-quality control.

5. Compared with fixed-threshold SeaCache on the same three VBench10 prompts, MiniDiT row split is slightly faster but less accurate against the target:
   - target 22: fixed SeaCache threshold `0.30` gives `1.979x`, PSNR `22.218`; MiniDiT row split gives `2.068x`, PSNR `20.466`.
   - target 28: fixed SeaCache threshold `0.15` gives `1.410x`, PSNR `27.728`; MiniDiT row split gives `1.539x`, PSNR `25.469`.

## 9. Artifacts

| Artifact | Path |
|---|---|
| architecture diagram | `reports/assets/mini_dit_cls_predictor_architecture.svg` |
| training loss curves | `reports/assets/mini_dit_cls_training_loss_curves.svg` |
| sample-split training dir | `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906` |
| row-split training dir | `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659` |
| online inference result root | `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328` |
| online summary CSV | `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/summary.csv` |
| online aggregate CSV | `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/aggregate_by_dataset_model_target.csv` |

