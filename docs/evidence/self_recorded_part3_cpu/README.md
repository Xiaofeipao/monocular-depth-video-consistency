# Self-recorded Part 3 CPU evidence

Same 16 RGB frames per clip as the capture review; newly inferred **metric**
VDA-Small baseline, then RAW/A/C/D/E and named conservative/guarded variants.
These are not the earlier relative inverse-depth predictions or scores.

`summary.json` contains both Farneback and independent RAFT common-mask scores.
Each clip includes 15 transitions per method/backend in CSV, source SHA256 and
sampling times, and RAW-referenced structural diagnostics. RAW is not depth GT.
`method_parameters.json` records parameters and source hashes;
`commands.json` records baseline inference arguments. `validation.json` verifies
64 baseline and 512 derived predictions, 120 flow pairs and 4 comparison videos.

Full NPY/RGB/valid, configs/environment/timing, flow/masks and fixed-range videos
remain in `outputs/self_recorded_part3_cpu/`. No raw video or personal environment
is copied into this numerical evidence directory.

Interpretation and reproducible command: [CPU comparison](../../self_recorded_part3.md).
