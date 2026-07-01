# ZEUS 方法与模型差异实验报告

生成时间：2026-06-30 CST

更新时间：2026-06-30 03:50 CST；已纳入 Wan2.2 strict ZEUS UniPC VBench10 high/low reset 补跑后的 10/10 完整结果。

本报告整理三条 ZEUS 实验线，用于说明：在 ZEUS 方法逻辑、prompt、seed、分辨率、帧数、采样步数和 ZEUS skip schedule 基本对齐后，Wan2.2 上的 ZEUS paired PSNR 明显低于 Wan2.1；因此当前证据支持“模型/架构/inference path 差异是 ZEUS 表现差异的主要原因”，而不是单纯由 ZEUS 代码逻辑未对齐导致。

三条实验线：

1. `/hy-tmp/work/zeus-official-new`：ZEUS 官方 demo / 官方仓库代码，Wan2.1 Diffusers。
2. `/hy-tmp/work/Wan2.1-official-zeus-strict`：按 ZEUS 论文方法在 Wan2.1 官方推理代码上的复现，包含 Euler 和 UniPC 两个采样器结果。
3. `/hy-tmp/work/Wan2.2/experiments/strict_zeus_vbench10_unipc_50step_45f_480p`：Wan2.2 strict guided-output ZEUS UniPC VBench10 实验。

注意：Wan2.2 high/low reset 实验已于 2026-06-30 03:46 CST 补齐 10/10 个 prompt。补跑使用原实验目录和原参数，通过 `--resume_existing` 跳过已完成的 `vbench10_001` 到 `vbench10_007`，并完成 `vbench10_008` 到 `vbench10_010` 的 baseline、ZEUS、ffprobe 和 PSNR。当前报告使用补跑后的最终 10-prompt aggregate。

## 方法简述

ZEUS 的核心思想是：在 diffusion/flow sampling 的中间区间跳过一部分昂贵的 denoiser forward，用最近 full evaluation 的输出构造近似 denoiser output，并把该近似输出交给 scheduler 继续走采样轨迹。

本报告关注的 strict ZEUS 复现逻辑是：

- fresh/recompute step：正常执行 cond/uncond denoiser forward，CFG 后得到最终提交给 scheduler 的 guided `noise_pred`。
- 记录 observed information：保存当前 fresh guided output `psi_t`，以及上一 solver step 实际提交给 scheduler 的输出 `hat{psi}_{t+1}`。
- 构造外推值：

```text
extrapolated_output = 2 * fresh_output - previous_submitted_output
```

- reuse step：连续跳步时在 `extrapolated_output` 和当前 `anchor_output` 之间交替，避免连续外推导致漂移。
- 每次实际提交给 scheduler 的输出，无论 fresh 还是 reuse，都会更新 `last_submitted_output`。

ZEUS schedule 使用同一组参数族：

| 参数 | 值 |
| --- | --- |
| `acc_range` | `(8, 47)` |
| `denominator` | `3` |
| `modular` | `(0, 1)` |
| `max_interval` | `6` |
| `lagrange_term` | `4` |
| `lagrange_int` | `4` |
| `lagrange_step` | `24` |

Wan2.2 的特殊处理：

- Wan2.2 T2V-A14B 有 high-noise / low-noise 两个 denoiser。
- high/low stage 切换时，strict ZEUS 会调用 `reset_observed_info()`，清空 `last_submitted_output`、`anchor_output`、`extrapolated_output` 和连续 skip 计数。
- 因此 Wan2.2 high/low reset 版本的默认 50-step UniPC 路径为 `26 reuse / 24 recompute`，比 Wan2.1 strict 的 `27 reuse / 23 recompute` 少跳 1 步。

## 实验目录与状态

| 实验线 | 代码路径 | 结果路径 | 状态 |
| --- | --- | --- | --- |
| ZEUS 官方 demo | `/hy-tmp/work/zeus-official-new` | `/hy-tmp/work/zeus-official-new/outputs/vbench10_wan21_zeus_seed42` | 完整，10/10 |
| Wan2.1 strict ZEUS 复现 | `/hy-tmp/work/Wan2.1-official-zeus-strict` | Euler: `/hy-tmp/work/Wan2.1-official-zeus-strict/outputs/vbench10_strict_zeus_euler_seed42`; UniPC: `/hy-tmp/work/Wan2.1-official-zeus-strict/outputs/vbench10_strict_zeus_seed42` | 完整，Euler 10/10，UniPC 10/10 |
| Wan2.2 strict ZEUS UniPC | `/hy-tmp/work/Wan2.2` | `/hy-tmp/wan22_strict_zeus_vbench10_unipc_50step_45f_480p_highlow_reset_20260629_1648` | 完整，10/10 |

## 统一实验协议

| 项 | 设置 |
| --- | --- |
| Prompt set | VBench10 |
| Prompt 数量 | 10 |
| Seed | 42 |
| Resolution | `832x480` |
| Frames | 45 |
| Sampling steps | 50 |
| Negative prompt | Wan2.1 / official demo 为 `bad quality, static` |
| 质量指标 | Paired PSNR：同模型 no-cache baseline vs ZEUS candidate |
| PSNR 脚本 | Wan2.1 strict UniPC 与 Wan2.2 使用 `/hy-tmp/work/compute_psnr.py`；official demo 使用原 ffmpeg PSNR 汇总，prompt 00 另做过统一脚本抽查 |

不能完全强行一致的模型特定参数：

| 参数 | Wan2.1 strict | Wan2.2 strict |
| --- | ---: | ---: |
| `shift` / `sample_shift` | `1.0` | `12.0` |
| `guidance_scale` | `5.0` | `(3.0, 4.0)` |
| 模型结构 | 单 denoiser | high-noise / low-noise 双 denoiser |

这些差异属于模型推荐推理配置和模型结构差异。本报告的比较对象是“ZEUS 相对各自 no-cache baseline 的质量保持和加速”，不是两个模型在完全相同 shift/guidance 下的直接生成质量比较。

## 总览结果

| 实验 | 模型/栈 | 采样器 | pairs | baseline total (s) | ZEUS total (s) | speedup | mean-of-mean PSNR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ZEUS 官方 demo | Wan2.1 Diffusers + official ZEUS patch | FlowMatch Euler | 10 | 4854.470 | 2258.600 | 2.1493x | 30.0372 dB |
| Wan2.1 strict 复现 | Wan2.1 官方代码 + strict ZEUS | Euler | 10 | 5376.029 | 2860.008 | 1.8797x | 32.8156 dB |
| Wan2.1 strict 复现 | Wan2.1 官方代码 + strict ZEUS | UniPC | 10 | 5552.712 | 2810.613 | 1.9756x | 30.4504 dB |
| Wan2.2 strict high/low reset | Wan2.2 官方代码 + strict ZEUS | UniPC | 10 | 5435.022 | 2619.430 | 2.0749x | 23.6675 dB |

核心观察：

- Wan2.1 上，无论 official demo 还是我们按论文方法复现，paired PSNR 都在约 `30 dB` 或以上。
- Wan2.2 在完整 10 个 VBench10 prompt 上有相近甚至略高的 speedup，但 mean-of-mean PSNR 只有 `23.6675 dB`。
- 与 Wan2.1 strict UniPC 全 10 个 prompt 对齐比较，Wan2.2 平均低 `6.7829 dB`。
- 因为 Wan2.2 当前代码已经使用 CFG 后 guided-output-level ZEUS，并且加入 high/low stage reset，低 PSNR 不再能主要归因于旧 branch-output cache 或跨 high/low 边界直接复用。

## 实验 1：ZEUS 官方 demo

### 配置

| 项 | 值 |
| --- | --- |
| 代码目录 | `/hy-tmp/work/zeus-official-new` |
| 运行脚本 | `/hy-tmp/work/zeus-official-new/run_vbench10_wan21.py` |
| 汇总脚本 | `/hy-tmp/work/zeus-official-new/summarize_vbench10_wan21.py` |
| 模型 | `/hy-tmp/models/Wan2.1-T2V-14B-Diffusers` |
| Pipeline | `diffusers.WanPipeline` |
| Scheduler | `FlowMatchEulerDiscreteScheduler` |
| VAE | `AutoencoderKLWan` |
| Prompt file | `/hy-tmp/work/Wan2.2/test_sets/Vbench10/prompts.txt` |
| height / width | `480 / 832` |
| frames | `45` |
| steps | `50` |
| guidance scale | `5.0` |
| negative prompt | `bad quality, static` |
| seed | `42` |
| fps | `15` |
| shards | `2` |

ZEUS patch：

```python
patch.apply_patch(
    zeus_pipe,
    acc_range=(8, 47),
    interp_mode="psi",
    denominator=3,
    modular=(0, 1),
    lagrange_int=4,
    lagrange_step=24,
    lagrange_term=4,
    max_interval=6,
)
```

### 聚合结果

| 指标 | 值 |
| --- | ---: |
| pairs | 10 |
| total baseline elapsed | `4854.470 s` |
| total ZEUS elapsed | `2258.600 s` |
| overall speedup | `2.1493x` |
| mean-of-mean PSNR | `30.0372 dB` |
| global min PSNR | `18.53 dB` |
| global max PSNR | `42.17 dB` |

计时 caveat：official demo 的 runner 在 `pipe(...)` 后调用 `export_to_video(...)`，日志计时包含视频导出；后续 strict/Wan2.2 实验的 speedup 口径更偏 inference/compute，不应把绝对时间直接横向比较。

### 逐 prompt 结果

| prompt | speedup | mean PSNR | min PSNR | max PSNR |
| ---: | ---: | ---: | ---: | ---: |
| 00 | 2.1405x | 31.3802 | 28.92 | 33.93 |
| 01 | 2.1415x | 34.5489 | 31.54 | 36.65 |
| 02 | 2.1547x | 23.4878 | 22.30 | 25.62 |
| 03 | 2.1534x | 29.7071 | 27.32 | 32.26 |
| 04 | 2.1533x | 20.1660 | 18.53 | 22.29 |
| 05 | 2.1505x | 32.8369 | 29.38 | 36.14 |
| 06 | 2.1514x | 37.4820 | 34.98 | 42.17 |
| 07 | 2.1495x | 27.6820 | 25.53 | 28.90 |
| 08 | 2.1496x | 33.1513 | 31.78 | 34.21 |
| 09 | 2.1490x | 29.9293 | 29.12 | 30.32 |

## 实验 2：Wan2.1 strict ZEUS 论文方法复现

该实验线在 Wan2.1 官方推理代码上复现 ZEUS 论文的 guided-output-level reuse 逻辑。它与 official demo 的区别是：不是 diffusers pipeline + official patch，而是在 Wan 官方推理循环里显式维护 ZEUS observed information 和 reuse 输出。

### 共同配置

| 项 | 值 |
| --- | --- |
| 代码目录 | `/hy-tmp/work/Wan2.1-official-zeus-strict` |
| 模型 | `/hy-tmp/models/Wan2.1-T2V-14B` |
| Prompt file | `/hy-tmp/work/Wan2.2/test_sets/Vbench10/prompts.txt` |
| size | `832*480` |
| frames | `45` |
| steps | `50` |
| shift | `1.0` |
| guidance scale | `5.0` |
| negative prompt | `bad quality, static` |
| seed | `42` |
| ZEUS cache object | CFG 后提交给 scheduler 的 guided `noise_pred` |
| PSNR | ffmpeg `psnr_avg`，UniPC canonical summary 使用 `/hy-tmp/work/compute_psnr.py` |

ZEUS 默认 skip path：

```text
[9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46]
```

即 `27 reuse / 23 recompute`。

### Euler 子实验

结果目录：

```text
/hy-tmp/work/Wan2.1-official-zeus-strict/outputs/vbench10_strict_zeus_euler_seed42
```

聚合结果：

| 指标 | 值 |
| --- | ---: |
| pairs | 10 |
| total baseline elapsed | `5376.029 s` |
| total ZEUS elapsed | `2860.008 s` |
| overall speedup | `1.8797x` |
| mean-of-mean PSNR | `32.8156 dB` |
| min mean PSNR | `22.6882 dB` |
| max mean PSNR | `39.0847 dB` |

逐 prompt：

| prompt | speedup | mean PSNR | min PSNR | max PSNR |
| ---: | ---: | ---: | ---: | ---: |
| 00 | 1.9683x | 31.8520 | 29.51 | 34.00 |
| 01 | 1.8873x | 37.7731 | 30.90 | 41.52 |
| 02 | 1.9210x | 22.9907 | 21.37 | 24.53 |
| 03 | 1.7989x | 32.0436 | 28.88 | 33.94 |
| 04 | 1.8985x | 22.6882 | 20.12 | 26.00 |
| 05 | 1.7869x | 38.4849 | 34.71 | 42.26 |
| 06 | 1.9408x | 39.0847 | 33.71 | 45.54 |
| 07 | 1.7898x | 34.7284 | 30.85 | 37.09 |
| 08 | 1.9311x | 33.7909 | 31.56 | 35.81 |
| 09 | 1.8991x | 34.7198 | 33.06 | 35.47 |

### UniPC 子实验

结果目录：

```text
/hy-tmp/work/Wan2.1-official-zeus-strict/outputs/vbench10_strict_zeus_seed42
```

重要历史说明：

- ZEUS 视频生成于 2026-06-27。
- 2026-06-29 发现原先 UniPC 结果 baseline 复用了 Euler baseline，不适合作为最终报告。
- 2026-06-29 已补跑同采样器 UniPC no-cache baseline，并用 `/hy-tmp/work/compute_psnr.py` 重算 canonical `metrics_summary.json/csv`。
- 下表使用补跑后的 canonical summary。

聚合结果：

| 指标 | 值 |
| --- | ---: |
| pairs | 10 |
| total baseline elapsed | `5552.712 s` |
| total ZEUS elapsed | `2810.613 s` |
| overall speedup | `1.9756x` |
| mean-of-mean PSNR | `30.4504 dB` |
| min mean PSNR | `21.6556 dB` |
| max mean PSNR | `38.1949 dB` |

逐 prompt：

| prompt | speedup | mean PSNR | min PSNR | max PSNR |
| ---: | ---: | ---: | ---: | ---: |
| 00 | 2.1004x | 30.4827 | 26.29 | 34.75 |
| 01 | 2.1616x | 37.2342 | 31.21 | 40.99 |
| 02 | 1.9697x | 23.0224 | 20.88 | 26.01 |
| 03 | 1.9401x | 29.4993 | 27.70 | 31.33 |
| 04 | 1.9915x | 21.6556 | 18.84 | 26.07 |
| 05 | 1.9463x | 34.0627 | 29.35 | 38.12 |
| 06 | 1.8966x | 38.1949 | 35.08 | 42.65 |
| 07 | 1.9653x | 29.2051 | 26.99 | 30.69 |
| 08 | 1.8259x | 31.9711 | 29.91 | 33.38 |
| 09 | 1.9906x | 29.1762 | 28.53 | 30.63 |

## 实验 3：Wan2.2 strict ZEUS UniPC VBench10

结果目录：

```text
/hy-tmp/wan22_strict_zeus_vbench10_unipc_50step_45f_480p_highlow_reset_20260629_1648
```

runner：

```text
/hy-tmp/work/Wan2.2/experiments/strict_zeus_vbench10_unipc_50step_45f_480p/run_batch.py
```

当前状态：

- 当前没有 tmux session。
- 完整 baseline+ZEUS+PSNR 的样本为 `vbench10_001` 到 `vbench10_010`，共 10 个。
- `failed/` 下未发现失败文件。
- runner 已写出 `results/summary.csv`、`results/summary.json` 和 `results/aggregate.json`。
- 补跑日志：`/hy-tmp/wan22_strict_zeus_vbench10_unipc_50step_45f_480p_highlow_reset_20260629_1648/logs/runner.resume_20260630_025638.console.log`。

### 配置

| 项 | 值 |
| --- | --- |
| 模型 | `/hy-tmp/models/Wan2.2-T2V-A14B` |
| task | `t2v-A14B` |
| prompt file | `/hy-tmp/work/Wan2.2/test_sets/Vbench10/prompts.jsonl` |
| size | `832*480` |
| frame_num | `45` |
| sample_steps | `50` |
| sample_solver | `unipc` |
| sample_shift | `12.0` |
| sample_guide_scale | `(3.0, 4.0)` |
| base_seed | `42` |
| offload_model | `True` |
| convert_model_dtype | `True` |
| timestep_cache | `strict_guided_zeus` |
| block cache | `none` |
| block group cache | `none` |
| cfg cache | `none` |
| PSNR script | `/hy-tmp/work/compute_psnr.py` |
| timing source | `inference_compute_elapsed_seconds` |

Wan2.2 high/low reset ZEUS path：

```text
[9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 25, 26, 27, 29, 30, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46]
```

即每个样本为 `26 reuse / 24 recompute`；10 个 prompt 合计 `260 reuse / 240 recompute`。

### 10-prompt 聚合结果

| 指标 | 值 |
| --- | ---: |
| completed pairs | 10 |
| total baseline compute time | `5435.022 s` |
| total ZEUS compute time | `2619.430 s` |
| overall speedup | `2.0749x` |
| mean-of-mean PSNR | `23.6675 dB` |
| min mean PSNR | `18.7207 dB` |
| max mean PSNR | `31.2300 dB` |
| global min PSNR | `17.00 dB` |
| global max PSNR | `35.01 dB` |

逐 prompt：

| sample | baseline time | ZEUS time | speedup | mean PSNR | min PSNR | max PSNR | reuse/recompute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vbench10_001` | 543.531 | 262.109 | 2.0737x | 20.1033 | 17.00 | 22.80 | 26/24 |
| `vbench10_002` | 544.224 | 261.907 | 2.0779x | 26.2687 | 19.76 | 30.49 | 26/24 |
| `vbench10_003` | 544.378 | 262.517 | 2.0737x | 18.9731 | 17.30 | 20.92 | 26/24 |
| `vbench10_004` | 544.915 | 262.336 | 2.0772x | 23.6547 | 22.83 | 24.08 | 26/24 |
| `vbench10_005` | 544.224 | 262.612 | 2.0724x | 18.7207 | 17.33 | 19.41 | 26/24 |
| `vbench10_006` | 544.593 | 262.491 | 2.0747x | 21.6251 | 18.93 | 25.56 | 26/24 |
| `vbench10_007` | 544.334 | 262.400 | 2.0744x | 31.2300 | 28.07 | 35.01 | 26/24 |
| `vbench10_008` | 540.107 | 261.146 | 2.0682x | 28.4349 | 26.21 | 29.57 | 26/24 |
| `vbench10_009` | 542.789 | 261.026 | 2.0794x | 21.0478 | 20.20 | 21.86 | 26/24 |
| `vbench10_010` | 541.927 | 260.886 | 2.0773x | 26.6169 | 25.86 | 27.20 | 26/24 |

## 关键对比：Wan2.1 strict UniPC vs Wan2.2 strict UniPC

为了排除采样器混杂，这里只比较 UniPC：

- Wan2.1 strict UniPC：完整 10/10。
- Wan2.2 strict UniPC high/low reset：完整 10/10。

| 指标 | Wan2.1 strict UniPC | Wan2.2 strict UniPC | Wan2.2 - Wan2.1 |
| --- | ---: | ---: | ---: |
| baseline total | `5552.712 s` | `5435.022 s` | `-117.690 s` |
| ZEUS total | `2810.613 s` | `2619.430 s` | `-191.183 s` |
| speedup | `1.9756x` | `2.0749x` | `+0.0993x` |
| mean-of-mean PSNR | `30.4504 dB` | `23.6675 dB` | `-6.7829 dB` |
| min mean PSNR | `21.6556 dB` | `18.7207 dB` | `-2.9349 dB` |
| max mean PSNR | `38.1949 dB` | `31.2300 dB` | `-6.9649 dB` |

逐 prompt mean PSNR：

| prompt | Wan2.1 strict UniPC | Wan2.2 strict UniPC | delta |
| ---: | ---: | ---: | ---: |
| 00 / `vbench10_001` | 30.4827 | 20.1033 | -10.3793 |
| 01 / `vbench10_002` | 37.2342 | 26.2687 | -10.9656 |
| 02 / `vbench10_003` | 23.0224 | 18.9731 | -4.0493 |
| 03 / `vbench10_004` | 29.4993 | 23.6547 | -5.8447 |
| 04 / `vbench10_005` | 21.6556 | 18.7207 | -2.9349 |
| 05 / `vbench10_006` | 34.0627 | 21.6251 | -12.4376 |
| 06 / `vbench10_007` | 38.1949 | 31.2300 | -6.9649 |
| 07 / `vbench10_008` | 29.2051 | 28.4349 | -0.7702 |
| 08 / `vbench10_009` | 31.9711 | 21.0478 | -10.9233 |
| 09 / `vbench10_010` | 29.1762 | 26.6169 | -2.5593 |

解释：

- Wan2.2 并不是因为跳得更多才质量差。它实际 `reuse=26`，比 Wan2.1 strict 的 `reuse=27` 更保守。
- Wan2.2 的 speedup 与 Wan2.1 strict UniPC 相近，且全 10 个 prompt 略高。
- 在相近 speedup、更少 reuse 的情况下，Wan2.2 paired PSNR 仍低 `6.7829 dB`。
- 这说明当前质量下降更可能来自 Wan2.2 的模型/推理管线特性，包括 high/low 双 denoiser、stage boundary、`shift=12.0` 的 timestep 分布、guide scale tuple 和模型输出动态，而不是 ZEUS 复现逻辑本身。

## 论证结论

本报告支持以下结论：

1. ZEUS 在 Wan2.1 上表现稳定。官方 demo 的 mean-of-mean PSNR 为 `30.0372 dB`，我们在 Wan2.1 官方推理代码上的 strict UniPC 复现为 `30.4504 dB`，strict Euler 复现为 `32.8156 dB`。

2. Wan2.2 在使用当前 strict guided-output ZEUS、UniPC、同一 VBench10 prompt/seed/resolution/frames/steps 和同一 ZEUS schedule 的情况下，完整 10 个 paired PSNR 的 mean-of-mean 只有 `23.6675 dB`。

3. Wan2.2 的 ZEUS 路径已经修正为 CFG 后 guided-output-level cache，并在 high/low stage 切换时重置 observed information；因此当前差异不能再主要归因于旧 branch-output cache 或直接跨 high/low stage 复用。

4. Wan2.2 的 reuse 次数比 Wan2.1 更少，但 PSNR 仍显著更低，说明不是“Wan2.2 跳得更多”导致的低质量。

5. 因此，当前证据支持“模型/架构/inference path 差异导致 ZEUS 表现差异”的判断。更严谨的表述是：Wan2.2-specific model and inference pipeline behavior 是主要原因；若要证明单一因素，例如纯模型权重或纯 high/low 架构，需要进一步消融。

## Caveats

1. Wan2.1 和 Wan2.2 保留了各自模型推荐/既有的 `shift` 和 `guidance_scale`，因此比较的是“ZEUS 相对各自模型 baseline 的影响”，不是完全相同采样超参下的直接模型横评。

2. official demo 与 strict 复现的 timing scope 不完全一致：official demo 计时包含 `export_to_video(...)`，strict/Wan2.2 更接近 inference/generation 或 compute-only 计时。

3. 当前质量指标主要是 paired PSNR。更完整的视频质量结论还需要 VBench 官方指标、LPIPS/video-LPIPS、CLIPScore/FVD 或人工检查。

4. Wan2.2 highlow_reset 目录已有 final runner summary；本报告中的 Wan2.2 聚合直接来自 `results/aggregate.json` 和 `results/summary.csv`。补跑日志为 `logs/runner.resume_20260630_025638.console.log`。

## 后续建议

1. 做 Wan2.2 弱 skip schedule 消融，例如降低 skip 密度或缩窄 `acc_range`，观察 PSNR 是否平滑恢复。

2. 做 Wan2.2 stage-boundary 消融，例如 high/low 切换前后扩大强制 recompute 窗口，判断误差是否集中在 stage transition。

3. 补充 VBench 官方指标或 LPIPS/video-LPIPS，以避免只依赖 paired PSNR。
