# Extended paired validation evidence

`summary_methods.csv` and per-clip CSVs compare frozen Part 2 metric VDA with
RAW/A/C/D/E_half. The four self-recorded excerpts and two TUM excerpts contain
374 RGB frames in total. Phone clips have no depth GT; use their warp scores as
proxies. Only TUM supplies depth and pose for native metric/geometry evaluation.

The `ibims_*` folders contain 100 paired RAW/C image evaluations per model.
This is the MoGe processed mirror with explicitly masked SI scoring, not the
original iBims MAT protocol. Depth Pro uses GPU FP32; UniDepthV2 uses CPU FP32.
Compare each refinement to its own baseline. Keep the negative Depth Pro
boundary result when citing the positive NYUv2 or UniDepthV2 results.

Files include input/checkpoint identities, resolved settings, software versions,
model timing, paired scores, prediction SHA256, validation counts and source
script hashes. Large RGB/NPY/valid/flow/video files remain in the corresponding
local `outputs/` directories; these numerical CSVs are not substitutes for them.

Prerequisites: the local model repositories/checkpoints, TUM xyz archive, four
phone videos and `python scripts/download_ibims_core.py --processed-mirror`.
See project dependency instructions for the official model environments.

```powershell
./scripts/run_extended_gpu_validation.ps1 -Python python -IncludeCpuReference
```

The optional CPU reference is slower. Its 252 inputs and the 518 GPU inputs
change resolution and device together; their comparison cannot isolate a
hardware or resolution effect. CPU postprocessing timings exclude VDA inference
and are not end-to-end application FPS. The GPU evidence has its own directory
at `../extended_validation_gpu518/`.

SHA256 identifies the original saved files, not a promise that new inference
on another hardware/software stack produces identical floating-point bytes.
Source byte hashes also depend on Git line-ending conversion. Compare decoded
arrays/paired metrics with suitable numerical tolerance when rerunning; retain
new checksums and environment metadata rather than replacing original evidence.
