# Test Sets

This directory keeps the prompt resources used for Wan2.2 T2V cache experiments.

## Layout

- `ali_10/`: 10 Ali prompts copied from repository `prompt.txt`.
- `openvid_100/`: 100 OpenVid prompts extracted from `/hy-tmp/openvid_100_wan22_prompts.zip`.
- `vbench_every20/`: VBench-2.0 prompts sampled from `VBench2_full_text.txt` by taking source lines `1, 21, 41, ...`.
- `all_prompts.jsonl` and `all_prompts.csv`: combined lightweight indexes across all three sets.
- `manifest.json`: source paths, counts, and sampling rule.

Each test-set subdirectory contains:

- `prompts.txt`: one prompt per line for simple runner input.
- `prompts.jsonl`: one JSON object per prompt with stable `sample_id`, source metadata, and `text`.

The VBench source was downloaded from:

https://raw.githubusercontent.com/Vchitect/VBench/master/VBench-2.0/prompts/VBench2_full_text.txt
