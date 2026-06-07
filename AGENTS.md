# AGENTS.md

## 机器与环境约定

- 本项目工作目录位于 `/hy-tmp/work/Wan2.2`。
- 这台云实例支持两种启动方式：GPU 启动和无卡启动。需要运行推理、评测、视频生成、PSNR 等 GPU 相关任务时，应确认当前是 GPU 启动模式。
- 当前已确认 GPU 模式下可见的显卡为 `NVIDIA A100 80GB PCIe`，显存约 `81920 MiB`，驱动版本 `570.211.01`，CUDA 版本 `12.8`。
- 数据盘挂载在 `/hy-tmp`，当前容量约 `400G`。所有项目代码、模型权重、缓存、实验输出、日志和结果表都应放在 `/hy-tmp` 下。
- 系统盘约 `30G`，不要把模型权重、数据集、checkpoint、视频输出或大型缓存放到 `/root`、`/`、`/usr` 等系统盘路径下。
- 用户已说明 `/hy-tmp` 下的数据不需要担心丢失；但是最重要的代码、实验结果、agent 聊天概要等仍然必须纳入 git 版本管理。
- Wan2.2 T2V-14B 权重体积约 `240.5 GB`，当前下载和后续推理默认使用的权重目录为 `/hy-tmp/models/Wan2.2-T2V-A14B`。
- Hugging Face / transformers 相关缓存建议显式指向 `/hy-tmp`，例如：

```bash
export HF_HOME=/hy-tmp/hf-cache
export TRANSFORMERS_CACHE=/hy-tmp/hf-cache
export HF_HUB_CACHE=/hy-tmp/hf-cache/hub
```

## 任务概述

我们的任务分为多个阶段。当前阶段，我们在 `Wan2.2` 上用 `T2V-14B` 权重在 `T2V` 任务上进行推理加速实验，评估 timestep cache, block cache, cfg cache 三种推理加速方法。我们的 conda 实验环境是 `Wan2.2`。

三种 cache 的组合逻辑约定如下：

- CFG cache 是最外层的分支选择逻辑。每个 denoising step 先运行 cond 分支；根据当前 cond 预测与历史 cond 预测的差异判断是否复用 CFG delta。
- 如果 CFG cache 命中，则跳过 uncond 分支，并用缓存的 `cond - uncond` delta 重构 CFG 输出。
- 如果 CFG cache 未命中，则继续运行 uncond 分支，并刷新 CFG delta。
- 对每个实际运行的分支（cond 或 uncond），先判断 timestep cache；只有 timestep cache 失效时，才进入 block cache 判断。
- 只有当 timestep cache 和 block cache 都失效时，才真正执行 transformer blocks。
- 因为 CFG cache 可能跳过 uncond，分支状态必须显式传递为 `cond`/`uncond`，不能用模型调用次数奇偶推断。

## 工作流要求

- `PROGRESS.md` 记录任务目前的进展。每个 session 开始前必须阅读，结束时必须更新。
- 将每个 session 的聊天记录都以一个单独的 log 文件形式保存到 `logs/` 目录下。记录应精炼概括，不要过于冗长。
- 使用 git 版本管理；只有在用户明确结束当前 session 或准备交接时，才 commit 一个版本，不要在中途阶段性修改后急着 commit。
- 最重要的代码、实验结果、agent 聊天概要等需要纳入 git 版本管理。

## 实验归档要求

每组实验必须完整归档，至少包括：

- 生成视频文件，且用 `ffprobe` 验证分辨率、帧数、fps、duration。
- 启动命令脚本或命令记录，必须包含完整参数、prompt、seed、GPU/FSDP/xDiT 参数、cache 参数和输出路径。
- 原始运行日志，保留模型加载、采样、保存、cache 命中/失效日志。
- 整理后的结果表，至少包含 elapsed time、speedup、PSNR（相对 baseline）、reuse/recompute 次数、reuse/recompute step 追踪、视频路径、日志路径。
- PSNR 计算命令和输出记录；默认使用同 prompt/seed/shape 的 no-cache baseline 作为参考。
- 对失败或中断实验单独记录原因，不能混入 completed results 表。
