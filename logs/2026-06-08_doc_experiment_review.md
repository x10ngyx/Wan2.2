# 2026-06-08 Doc Experiment Review

- Read local PDF reports in `doc/` using `pypdf` text extraction.
- `Wan2.2 Cache方案.pdf` contains high-level cache plan notes and Yuque embed links, but limited extractable numeric data.
- Step-level hidden-states residual report: output relative L2 is the primary skip decision metric, stability is secondary, cosine similarity is explicitly not used. Middle steps have the lowest residuals; initial and final steps are riskier.
- ZEUS quality/speed report: baseline average time 448.28s for 832x480, 45 frames, 50 steps, 10 prompts. Best quality was ZEUS r=2 40%-60% at 38.08 dB / 1.17x; useful balanced points include r=2 40%-90% at 37.86 dB / 1.48x and r=3 40%-90% at 36.55 dB / 1.57x.
- Specific-block skip report: baseline average time 441.48s. BlockSkip 20%-60% only reached 1.12x-1.44x and PSNR around 19.44 down to 15.45 dB; a step>=20 restriction improved a global v8-style result from 19.44 dB to 26.34 dB but still looked risky.
- Practical conclusion: prioritize timestep/CFG-style cache validation before block cache; if block cache is pursued, gate by timestep and validate with PSNR/SSIM/VMAF before relying on residual-only selection.
