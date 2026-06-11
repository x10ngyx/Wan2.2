# 2026-06-11 OpenVid Prompt Comparison

- Read `/hy-tmp/openvid_100_wan22_prompts.zip`.
- Confirmed the main experiment manifest is `openvid_100_wan22_prompts/dataset_100.jsonl`; `text` is the Wan2.2 prompt and `id` is the stable sample identifier.
- The package contains 100 prompt records plus 100 source/reference mp4 files. Source video ffprobe verification reports 100 ok, 0 failed.
- Compared the OpenVid 100 prompts with the original `prompt.txt` 10-prompt set.
- Key difference: OpenVid captions are longer and more dataset-caption-like. Original 10 average 67.9 words, 416.3 chars, and 3.6 sentences; OpenVid 100 average 112.2 words, 620.4 chars, and 6.8 sentences.
- Style difference: original prompts more often contain explicit camera/motion generation instructions; OpenVid prompts more often use repeated caption openings such as `The video features`, `The video shows`, `The video captures`, and `In the video`.
- No code or experiment scripts were changed.
