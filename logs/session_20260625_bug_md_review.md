# 2026-06-25 BUG.md Cache Review

## Summary

- Read `BUG.md` and checked its claims against:
  - `wan/timestep_cache.py`
  - `wan/block_group_cache.py`
  - `wan/cfg_cache.py`
  - `wan/modules/model.py`
  - `adaptive_seacache_wan22/cache.py`
  - `adaptive_seacache_wan22/patch.py`
  - `adaptive_threshold_predictor/data.py`
  - `adaptive_threshold_predictor/build_feature_cache.py`
  - `adaptive_threshold_predictor/README.md`
- No code changes were made.
- `BUG.md` is an untracked external review file and was left untouched.

## Findings

- Several listed code snippets are present, but the interpretation is often stronger than the code supports.
- SeaCache `previous_feature` is updated on every decision so the next step compares against the immediately previous filtered feature; this appears consistent with consecutive-step gating, not necessarily a bug.
- SeaCache accumulated distance persists across reuse hits until threshold crossing; this is consistent with an accumulated-threshold design.
- Adaptive predictor training and online inference both use raw latent-derived pooled features by design; this does not mismatch the online wrapper, although it is separate from SeaCache's internal `_modulated_norm1` metric feature.
- `candidate_inverse` is the documented default dataset mode and is intentionally candidate-wise; it may be a weak formulation for target-conditioned control, but it is not an accidental code mismatch.
- Possible cleanup/risk items remain around scheduler sigma anchoring, final-step cutoff conservatism, FFT gain normalization edge cases, and defensive BlockGroup pending-feature clearing.

## Validation

- Static code review only.
- Did not run GPU inference, PSNR evaluation, or unit tests.

## Follow-up: Official SeaCache Alignment

- User clarified that official `https://github.com/jiwoogit/SeaCache` should be the source of truth.
- Cloned the official repo to `/hy-tmp/seacache_official_ref` and compared commit `3b1c688`.
- Official Wan2.1 confirms that most `BUG.md` SeaCache claims are not bugs relative to the reference:
  - `previous_feature`/`previous_e0` is updated every call.
  - accumulated relative L1 persists across reuse decisions and resets on threshold crossing.
  - SEA filter uses `scheduler.sigmas[idx]`.
  - default cutoff forces the final step per branch to recompute.
  - mean gain normalization matches the local implementation.
- Applied one local fix in `wan/timestep_cache.py`: ret/cutoff/history-missing/forced recompute paths now store the unfiltered modulated input and skip SEA filtering, matching official Wan2.1 behavior.
- Validation after the fix:
  - `python -m py_compile wan/timestep_cache.py`
  - CPU-only direct-module behavior check for ret/middle/cutoff SeaCache decisions.
