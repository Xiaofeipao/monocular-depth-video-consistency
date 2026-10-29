# C_cuda acceptance evidence

- `tum16` and `holdout32`: comparison JSON/CSV and CUDA numerical/timing
  checks from serial fixed-thread experiments, one warmup plus five repeats.
  Numeric observations are unchanged; public environment fields use generic
  descriptions and the device alias `CUDA_GPU_A`.
- `holdout_source_manifest.json`: source hashes, timestamps, poses and protocol
  for reconstructing the later clip; relative RGB/depth paths refer to the
  prepared local dataset, not this documentation directory. Rebuild it using
  `prepare_tum.py --start-seconds 10 --frames 32` and the recorded archive.
- `holdout_inference_*`: actual metric VDA GPU source metadata and timing.
- `acceptance.json`: derived checks against the numerical/metric tolerances and
  speedup >1.1; it does not replace or edit the measured observations.

See [scope, commands and limitations](../../part3_cuda_20261029.md).
No accuracy improvement over C is claimed: this is a faster, numerically
equivalent implementation. The later segment was not used to tune parameters,
but shares TUM fr1/xyz's scene and therefore is not an independent dataset.
