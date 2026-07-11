# 2026-07-11 Predictor-RL Hyperparameter Scale Analysis

## Scope

- Quantified `predicotr-rl/` default reward, speed-proxy, target-augmentation, and IQL optimizer settings against the OpenVid-100 `summary.csv`.
- Used the available 2,000-step latent-MSE smoke cache only for a provisional latent-reward scale check.
- No GPU work, training run, checkpoint change, or implementation change was performed.

## Measured Data Scale

- Dataset: `1000` candidates from `100` prompts and `10` SeaCache thresholds. The default five target offsets produce `250000` transitions before the `80/20` sample split; at batch size `256`, this is about `781` train updates per epoch and `23438` updates in `30` epochs.
- Measured speedup has mean `2.313x`, median `2.199x`, range `1.092-3.568x`. Reuse rate is `54.97%`.
- With `reuse_cost_ratio=0.081`, final action-cost proxy absolute speedup error has mean `0.0064x`, median `0.0059x`, p95 `0.0149x`, and max `0.0411x`. The proxy is not the material calibration risk.
- Mean PSNR is `26.47 dB`, standard deviation `7.78 dB`, and range `13.60-48.86 dB`. PSNR variation across prompts at one fixed threshold is still `3.18-4.80 dB`, so raw absolute PSNR introduces prompt-dependent terminal-return offsets.

## Reward Analysis

- With local offsets `[-0.30, -0.15, 0, 0.15, 0.30]`, `lambda_speedup=10` gives terminal speed penalties with median `1.538`, mean `1.768`, p95 `3.078`, and max `3.411`. The center offset has only proxy mismatch, while the nominal `0.15x` and `0.30x` mismatches contribute about `1.5` and `3.0` reward units respectively.
- In the 2,000-step smoke MSE cache, per-trajectory MSE sum has median `0.2041` and mean `0.2886`; `lambda_latent=5` therefore yields median latent penalty `1.020` and mean `1.443`. The corresponding full-trajectory recompute penalty is median `0.82` and mean `0.90`. These two immediate components are well balanced, subject to confirmation with the full MSE cache.
- The raw PSNR term remains much larger than the immediate terms. This is valid if PSNR is intentionally the primary objective, but quality-vs-speed trade-off must be set by `lambda_speedup`, not inferred from its absolute magnitude alone.

## Target-Speed Weight

- For each prompt and adjacent threshold pair, calculated the speedup coefficient at which the faster trajectory becomes preferable at its faster target: `max(0, PSNR_slow - PSNR_fast) / (speedup_fast - speedup_slow)`.
- Median break-even weights by pair are: `0.10->0.15: 30.13`, `0.15->0.20: 17.20`, `0.20->0.25: 9.42`, `0.25->0.30: 8.95`, `0.30->0.40: 4.58`, and below `2.7` for subsequent pairs.
- Consequently `lambda_speedup=10` is adequate for many moderate/high-speed local comparisons but underweights target adherence around the sharp low-speed quality cliff. This also explains why a `2.5x` trajectory expanded to a faster target can help but only when its quality advantage is less than the resulting speed penalty.
- An oracle that selects the maximum current terminal reward among all ten real threshold trajectories for each prompt has mean target-speed MAE `0.823x` at `lambda_speedup=10`, `0.208x` at `20`, and `0.101x` at `30`; this is a reward-shape diagnostic, not a policy result. Because actual augmentation only supplies local target values, the recommended experiment is `lambda_speedup=10/20/30`, reported separately by target-speed bands.
- The `-0.30` and `-0.15` offsets each clamp `100` low-speed candidates to target `1.0x`; this duplicates the low target. Consider deduplicating clipped target values only if target-frequency balance becomes an observed issue.

## IQL Parameters

- `tau=0.7` is a conventional moderate expectile. `gamma=1` is appropriate for a fixed 50-step finite episode with a terminal quality reward. `rho=0.995` means a target-Q update rate of `0.005` and half-life about `138` optimizer updates, or `0.18` epoch at the default dataset/batch size; it is stable rather than excessively stale.
- `hidden_dim=256`, three layers, `lr=3e-4`, no dropout, and 30 epochs are reasonable initial values for a `644`-dimensional normalized state and about `200000` train transitions after augmentation.
- The nonstandard high-sensitivity setting is batch-normalized advantage followed by `beta=3`. If standardized advantages are approximately normal, weight is `exp(3A)`: `A=-1` gives `0.050`, `A=1` gives `20.1`, `A=1.5` gives `90.0`, and all `A>=1.535` (about `6.2%` of samples) are clipped to `100`; expected clipped weight is about `12.7`. This is much sharper than a neutral behavior-cloning update. Ablate `beta=1`, `1.5`, and `3`, or remove advantage standardization and tune beta in raw-Q units.
- Best-checkpoint selection by held-out action accuracy remains unsuitable as the final selection criterion. Select among checkpoints by closed-loop target-speed error and PSNR once the runtime adapter exists.

## Recommendation

- Retain `reuse_cost_ratio=0.081`, `lambda_latent=5`, `lambda_recompute=0.04`, `tau=0.7`, `gamma=1`, `rho=0.995`, and the initial optimizer/model capacity.
- Do not change defaults from this static analysis alone. In the first GPU experiment, train the focused grid `lambda_speedup={10,20,30}` by `beta={1,1.5,3}`, then evaluate the adapter on held-out prompts at target speeds spanning the low, middle, and high ranges.
- Consider centering PSNR per prompt for critic/value training, while retaining a separately reported raw PSNR objective, to remove the `3-5 dB` prompt difficulty offset from cross-prompt advantage weighting.
