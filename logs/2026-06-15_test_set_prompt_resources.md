# 2026-06-15 Test Set Prompt Resources

## Summary

- Created `test_sets/` to hold prompt resources for future Wan2.2 T2V cache experiments.
- Added:
  - `test_sets/ali_10/`: 10 prompts copied from `prompt.txt`.
  - `test_sets/openvid_100/`: 100 prompts extracted from `/hy-tmp/openvid_100_wan22_prompts.zip`.
  - `test_sets/vbench_every20/`: 51 prompts sampled from VBench-2.0 full text by taking source lines `1, 21, 41, ... 1001`.
- Added combined indexes `test_sets/all_prompts.jsonl` and `test_sets/all_prompts.csv`.
- Added `test_sets/manifest.json`, `test_sets/README.md`, and `test_sets/SHA256SUMS`.

## Validation

- `wc -l` results:
  - Ali prompts: `10`
  - OpenVid prompts: `100`
  - VBench full source prompts: `1013`
  - VBench sampled prompts: `51`
  - Combined JSONL rows: `161`
- `python -m json.tool test_sets/manifest.json` passed.
- All JSONL files under `test_sets/` parsed successfully.

## Notes

- Follow-up update: `AGENTS.md` now documents `/hy-tmp/work/Wan2.2/test_sets` as the unified prompt test set directory, with OpenVid-100 under `test_sets/openvid_100/`, replacing the old zip-only resource note.
- No inference, GPU workload, PSNR, or video generation was run.
- Existing dirty worktree entries were left untouched except for appending this session's `PROGRESS.md` entry.
