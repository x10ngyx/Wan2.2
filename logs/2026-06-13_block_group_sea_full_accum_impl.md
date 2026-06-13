# 2026-06-13 Block-Group Sea Full Accumulated Cache Implementation

## Summary

- Implemented an experimental block-group cache variant using full group-entry features, SeaCache-style scheduler-aware filtering, and accumulated relative-L1 decisions.
- Preserved legacy block-group behavior by keeping defaults as `--block_group_decision instant` and `--block_group_metric pooled_rel_l1`.

## Files Changed

- `wan/block_group_cache.py`
  - Added `decision`, ret/cutoff, and SEA filter config fields.
  - Added `sea_full_rel_l1` metric support.
  - Added accumulated distance state and summary fields.
  - Added full-feature SEA filtering over the latent token grid.
- `wan/modules/model.py`
  - Allows `sea_full_rel_l1` to return full group-entry features.
  - Passes grid size and scheduler sigmas into block-group cache decisions.
- `wan/text2video.py`
  - Passes scheduler sigmas through the model forward wrapper for block-group cache.
- `generate.py`
  - Added CLI args for block-group decision mode, sea metric, ret/cutoff steps, and SEA filter parameters.
- `PROGRESS.md`
  - Appended this implementation handoff entry.

## Usage

Example experimental invocation fragment:

```bash
--block_cache block-group \
--block_group_decision accumulated \
--block_group_metric sea_full_rel_l1 \
--block_threshold 0.2 \
--block_group_ret_steps 1 \
--block_group_cutoff_steps 1
```

## Validation

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile wan/block_group_cache.py wan/modules/model.py wan/text2video.py generate.py
/hy-tmp/miniconda3/envs/Wan2.2/bin/python generate.py --help | rg -n "block_group_(decision|metric|ret|cutoff|sea)"
```

- A CPU state-machine check also verified first-step protection, accumulated reuse, tail-step recompute, pending filtered feature storage, and summary fields.

## Notes And Risks

- This variant intentionally stores full filtered features per block group, in addition to cached residuals.
- At the default `832x480`, `45` frame shape, this can add multiple GB of GPU memory pressure. Run small pilot tests before large threshold grids.
- No full GPU inference experiment was launched in this session.
