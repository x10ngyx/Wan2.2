# Vbench10

`Vbench10` is the unified 10-prompt VBench subset for subsequent Wan2.2 acceleration comparisons.

## Source

- Parent set: `test_sets/vbench_every20/`
- Parent source: VBench-2.0 `VBench2_full_text.txt`
- Parent set size: 51 prompts
- Sampling: `random.sample(range(51), 10)` with Python `random.Random(42)`, then sorted by parent index before writing

## Files

- `prompts.txt`: one prompt per line for simple runner input.
- `prompts.jsonl`: one JSON object per prompt with stable `vbench10_XXX` sample IDs and source metadata.
- `selection_manifest.json`: reproducible selection metadata.

## Selected Prompts

| sample_id | parent_sample_id | VBench source line | prompt |
| --- | --- | ---: | --- |
| `vbench10_001` | `vbench_every20_002` | 21 | A woman is playing football. |
| `vbench10_002` | `vbench_every20_007` | 121 | A horse is running along the beach, then it suddenly stops and starts grazing. |
| `vbench10_003` | `vbench_every20_008` | 141 | The race began, and Team A quickly took the lead, securing the front position. Throughout the race, Team B kept pushing, gradually closing the gap on Team A, especially on the downhill segment. In the next uphill section, the Team C runner displayed strong endurance and stamina, overtaking both Team A and Team B in a few short kilometers, taking the lead. As the race entered its final stage, Team B increased their speed, but Team C maintained a steady pace and continued to lead. At the finish line, Team C sprinted to victory with a clear advantage. |
| `vbench10_004` | `vbench_every20_009` | 161 | Snow White was driven into the forest by the evil queen who was jealous of her beauty. In the forest, Snow White met seven kind dwarfs who offered her shelter. However, the evil queen disguised herself as an old woman and gave Snow White a poisoned apple. After biting the apple, Snow White fell into a deep sleep. At that moment, a prince heard of her plight and came to save her. He kissed her, breaking the spell. After Snow White regained consciousness, she decided to take revenge on the queen. She, along with the prince and the dwarfs, returned to the palace and overthrew the evil queen, restoring peace to the kingdom. |
| `vbench10_005` | `vbench_every20_015` | 281 | The camera orbits around in a clockwise direction. Forbidden City. |
| `vbench10_006` | `vbench_every20_016` | 301 | Equal amounts of yellow and blue paint are rapidly combined, with the mixture being vigorously stirred until fully blended. |
| `vbench10_007` | `vbench_every20_018` | 341 | A timelapse captures the reaction as concentrated sulfuric acid is poured onto a rubber balloon. |
| `vbench10_008` | `vbench_every20_035` | 681 | A cat is on the right of a rock, then the cat runs to the left of the rock. |
| `vbench10_009` | `vbench_every20_041` | 801 | people are playing ping-pong. |
| `vbench10_010` | `vbench_every20_048` | 941 | The camera orbits around. Serengeti, the camera circles around. |
