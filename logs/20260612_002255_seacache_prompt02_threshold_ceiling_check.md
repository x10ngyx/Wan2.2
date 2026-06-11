# 2026-06-12 SeaCache prompt02 threshold ceiling check

- Checked prompt 02 SeaCache high-threshold run at `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`.
- Confirmed threshold `0.80` produced `39/50` unique timestep reuse, `3.499x` speedup, and `18.631 dB` mean PSNR.
- Inspected cache trace and implementation: `39` reuse is not a hard implementation limit. Default SeaCache protects the first and last denoising steps, and the Wan2.2 high/low stage split creates separate cold starts. With very high thresholds the theoretical default ceiling is about `47/50` unique step reuse, but quality would likely degrade further.
- From the logged `0.80` rel-L1 path, rough replay suggests thresholds around `1.0-2.0` may add only a few reuse steps before quality becomes very low; `0.80` is already beyond the useful quality/speed region for prompt 02.
