# OpenVidHD-100 prompts for Wan2.2 inference

This package contains 100 selected OpenVidHD records for Wan2.2 text-to-video
inference experiments.

The selected captions are intended to be used as Wan2.2 prompts. The original
OpenVid videos are included only as reference/source material and for traceable
sample inspection; they are not required by Wan2.2 `t2v-A14B` inference.

## Files

- `dataset_100.jsonl`
  - Main manifest for experiments.
  - Each line is one JSON object.
  - Required fields:
    - `text`: prompt text for Wan2.2.
    - `video`: relative path to the source/reference mp4.
  - Extra fields include `id`, `part`, `content_group`, `portrait_group`, and
    `motion_group`.
- `prompts.txt`
  - One prompt per line, in the same order as `dataset_100.jsonl`.
  - Use this when a runner only needs plain prompt strings.
- `selected_100.csv`
  - Tabular manifest with grouping labels and OpenVid metadata.
- `selected_100.jsonl`
  - Full manifest with extra metadata.
- `selection_stats.json`
  - Balance statistics.
- `verification_summary.json`
  - Video integrity summary.
- `ffprobe_manifest.json`
  - `ffprobe` metadata for the 100 source videos.
- `video_manifest.json`
  - Extraction manifest.
- `videos/`
  - 100 source/reference mp4 files.
- `scripts/`
  - Selection and extraction scripts used to create this subset.

## Balance

The subset was selected from `nkp37/OpenVid-1M`, `OpenVidHD_part_1`.

- 100 total records.
- Content groups:
  - 39 landscape / scene records.
  - 35 people / portrait records.
  - 26 subject / object records.
- Portrait split:
  - 35 portrait-or-people records.
  - 65 non-portrait records.
- Motion split:
  - 55 dynamic records.
  - 45 calm records.

All 100 source videos were extracted through HTTP Range reads from the Hugging
Face mirror and validated with `ffprobe`.

## Wan2.2 usage

For a single record, pass the `text` field to `generate.py --prompt`.

Example:

```bash
cd /home/huteng/xiongyuxiang/Wan2.2

python generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --offload_model True \
  --convert_model_dtype \
  --prompt "PUT_ONE_TEXT_FIELD_HERE"
```

For batch experiments, read `dataset_100.jsonl` or `prompts.txt` and launch one
Wan2.2 inference per prompt. Keep each output filename tied to the record `id`
so cache-speed and quality metrics can be compared across the same prompt set.

Minimal Python reader:

```python
import json
from pathlib import Path

dataset = Path("dataset_100.jsonl")
for line in dataset.open(encoding="utf-8"):
    record = json.loads(line)
    prompt = record["text"]
    output_id = record["id"]
    print(output_id, prompt)
```

## Notes

- The prompts are descriptive captions from OpenVidHD metadata, not manually
  rewritten prompts.
- The `video` paths are relative to the package root.
- The source videos are provided for inspection and traceability; Wan2.2 T2V
  only needs the prompt text.
- The original large metadata files `OpenVidHD.csv` and `OpenVidHD.json` are not
  included in the zip package.
