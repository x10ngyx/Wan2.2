# AGENTS.md

## 项目目标

本项目工作目录为 `/hy-tmp/work/Wan2.2`。当前研究任务是在 Wan2.2 `T2V-14B` 上做推理加速，并形成从 cache threshold 到视频质量/加速效果的数据闭环。

项目分三阶段推进：

1. 实现三个基于 threshold 的 cache 方法，用于 T2V 推理加速：
   - `timestep cache`：按 denoising step 判断是否复用整次 transformer/denoiser 输出。当前主要实现包括 `zeus-threshold`，另有固定 ZEUS cadence 和 SeaCache 作为对照/扩展方法。
   - `block cache`：在 timestep cache 未命中时，按 transformer block 或 block group 判断是否复用中间 residual。当前推荐实验接口是 `block-group`。
   - `cfg cache`：在最外层判断是否复用 CFG delta，从而跳过 uncond 分支。当前实现是 `threshold`。
   - 三种 cache 的组合顺序必须保持为：CFG cache 最外层；每个实际运行的 `cond`/`uncond` 分支先判断 timestep cache；只有 timestep cache 失效时才进入 block cache；只有 timestep 和 block cache 都失效时才真正执行 transformer blocks。cache 状态必须显式区分 `model_stage` 和 `branch`，不能用模型调用次数奇偶推断分支。
   - 推荐统一 CLI：`--timestep_cache zeus-threshold --timestep_threshold <float>`，`--block_cache block-group --block_threshold <float>`，`--cfg_cache threshold --cfg_threshold <float>`。历史参数 `--zeus_threshold`、`--block_group_threshold`、`--bwcache_thresh` 仍兼容；不要为了整理接口改动 cache 核心逻辑。
2. 造数据：系统扫描 threshold 组合，建立 `threshold 组合 -> 视频质量与加速效果` 的关系。质量默认用同 prompt/seed/shape 的 no-cache baseline 计算 FFmpeg PSNR；加速默认用 compute-only inference elapsed time 计算 speedup。
   - 数据表按“一行对应一次完整候选推理”组织，至少包含 prompt/sample 标识、生成参数、cache 方法、三个 threshold、metric/start/end/max_reuse、compute elapsed、speedup、PSNR、reuse/recompute 统计、失败状态、视频路径和日志路径。
3. 实现自适应推理加速：基于已归档实验数据训练一个小规模预测网络，输入目标视频质量与目标加速效果，预测应使用的 threshold 组合，再用该组合做推理。预测网络训练/验证/推理脚本应与实验 runner 分离；runner 只负责按给定 threshold 组合生成和评测。

## 机器与环境约定

- 这台云实例支持 GPU 启动和无卡启动。运行推理、评测、视频生成、PSNR 等 GPU 相关任务前，必须确认当前是 GPU 启动模式。
- GPU 模式下已确认显卡为 `NVIDIA A100 80GB PCIe`，显存约 `81920 MiB`，驱动版本 `570.211.01`，CUDA 版本 `12.8`。
- conda 实验环境为 `/hy-tmp/miniconda3/envs/Wan2.2`，环境名 `Wan2.2`。
- 数据盘挂载在 `/hy-tmp`，容量约 `400G`。项目代码、模型权重、缓存、实验输出、日志、结果表、handoff 包都应放在 `/hy-tmp` 下。
- 系统盘约 `30G`。不要把模型权重、数据集、checkpoint、视频输出或大型缓存放到 `/root`、`/`、`/usr` 等系统盘路径下。
- 用户已说明 `/hy-tmp` 下的数据不需要担心丢失；但重要代码、实验结果表、报告、agent 聊天概要仍必须纳入 git 版本管理。
- Wan2.2 T2V-14B 权重目录为 `/hy-tmp/models/Wan2.2-T2V-A14B`。
- OpenVid-100 prompt zip 路径为 `/hy-tmp/openvid_100_wan22_prompts.zip`。
- Hugging Face / transformers 缓存建议显式指向 `/hy-tmp`：

```bash
export HF_HOME=/hy-tmp/hf-cache
export TRANSFORMERS_CACHE=/hy-tmp/hf-cache
export HF_HUB_CACHE=/hy-tmp/hf-cache/hub
```

## 工作流要求

- 每个 session 开始前必须阅读 `PROGRESS.md`，确认当前阶段、已完成实验、正在运行的 tmux/session 和已知坑。
- 每个 session 结束前必须更新 `PROGRESS.md`，并在 `logs/` 下写一个精炼 session log。日志应记录做了什么、改了哪些文件、删了哪些中间日志、是否运行验证、是否有未完成事项。
- `logs/` 只保留有交接价值的记录：环境/资源、实现说明、关键实验完成结果、报告/数据集/交接包说明、重要 bug 修复与规范变更。可删除被 `PROGRESS.md` 或完整实验归档覆盖的启动监控、半程进度、重复 retry、下载 raw log 等中间噪音。
- 使用 git 版本管理；只有在用户明确结束当前 session 或准备交接时，才 commit，不要在中途阶段性修改后急着 commit。
- 不要回滚或覆盖用户已有改动。遇到脏工作区时先读 diff，再只处理与当前任务相关的文件。

## 实验参数规范

除非用户明确要求变更，T2V cache 实验默认使用：

- task/model：`t2v-A14B`
- checkpoint：`/hy-tmp/models/Wan2.2-T2V-A14B`
- conda：`/hy-tmp/miniconda3/envs/Wan2.2`
- seed：固定 `42`
- size：`832*480`
- frame_num：`45`
- sample_steps：`50`
- sample_solver：`dpm++`
- offload：`--offload_model`
- dtype：`--convert_model_dtype`
- baseline：同 prompt/seed/shape 的 no-cache 输出
- PSNR：FFmpeg `psnr` filter，默认排除 perfect/Infinity frames 后汇总
- timing：优先使用 `inference_compute_elapsed_seconds`，不要用包含模型加载、视频保存和权重搬运的整进程 wall time 计算 speedup

多候选实验必须优先使用单进程 batch runner：每个进程只加载一次 WanT2V pipeline/checkpoint shards，然后在同一进程内顺序运行候选。只有为了隔离崩溃、排查显存泄漏或验证冷启动行为时，才使用逐候选独立进程；这种例外必须在实验记录中说明原因。

## 实验归档要求

每组实验必须完整归档，至少包括：

- 生成视频文件。
- `ffprobe` JSON，验证分辨率、帧数、fps、duration。
- 启动命令脚本或命令记录，包含完整参数、prompt、seed、GPU/FSDP/xDiT 参数、cache 参数和输出路径。
- 原始运行日志，保留模型加载、采样、保存、cache 命中/失效 summary。
- 整理后的结果表，至少包含 elapsed time、speedup、PSNR、reuse/recompute 次数、reuse/recompute step 追踪、视频路径、日志路径。
- PSNR 计算命令和输出记录。
- 对失败或中断实验单独记录原因，不能混入 completed results 表。

推荐归档目录结构：

```text
/hy-tmp/wan22_<experiment_name>_<timestamp>/
  baseline/
  <method_or_combo>/
  commands/
  logs/
  ffprobe/
  psnr/
  results/
  failed/
  experiment_config.json
  launch.env
```

仓库内的 `experiment_results/` 可放结果根目录 symlink，便于查看；大型视频和模型权重不要直接复制进 git。
