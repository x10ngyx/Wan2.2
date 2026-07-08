# Session 2026-07-04 SeaCache Ali-10 Prompt-02 Path Lookup

## Request

Locate the Wan2.2 SeaCache Ali-10 second-prompt result folder and experiment folder.

## Findings

- Complete SeaCache Ali-10 UniPC experiment root: `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222`
- Repository experiment scripts: `experiments/seacache_unipc_ali10_50step_45f_480p/`
- Repository symlink: `experiment_results/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222`
- Second prompt sample id: `ali_002`
- Baseline video: `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222/baseline/ali_002.mp4`
- Candidate video folders:
  - `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222/seacache/th_0p10/`
  - `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222/seacache/th_0p20/`
  - `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222/seacache/th_0p30/`
  - `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222/seacache/th_0p50/`
- Summary CSV: `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222/results/summary.csv`

Earlier DPM++ prompt-02 pilot roots also exist:

- `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826`
- `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`
- Repository scripts for those pilots: `experiments/seacache_50step_45f_480p/`

## Changes

- Updated `PROGRESS.md` with a short path lookup note.
- Added this session log.

## Validation

- Read `PROGRESS.md`.
- Listed archive directories and summary rows for `ali_002` / prompt `02`.
- No inference, PSNR, or GPU validation was run.
