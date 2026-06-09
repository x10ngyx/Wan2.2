# 2026-06-09 experiment batch-runner rule

- User asked why the current block-cache-only experiment did not use a batch runner and whether this was already in the experiment spec.
- Checked `AGENTS.md`: it required complete experiment archiving but did not explicitly require single-process batch execution to avoid repeated checkpoint shard loading.
- Root cause for the block-cache-only shell runner: I reused the existing per-candidate shell runner pattern instead of promoting the earlier ZEUS-threshold single-process runner pattern into a general experiment rule.
- Updated `AGENTS.md` to require multi-candidate experiments to prefer a single-process batch runner that loads WanT2V/checkpoint shards once per process and runs candidates sequentially.
- Documented exceptions: independent per-candidate processes are allowed only for crash isolation, memory-leak debugging, or cold-start validation, and must be justified in the experiment record.
