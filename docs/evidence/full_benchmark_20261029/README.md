# Full benchmark evidence

These are new local runs on the complete Middlebury 15 training pairs and the
standard 654-image NYUv2 test split. Historical teammate evidence is preserved
in its original directories.

- `middlebury`: four experiments, per-scene metrics and the passing 60-file /
  1,200-value saved-prediction reload audit.
- `depth_pro_nyuv2`, `unidepth_v2_nyuv2`: seven mandatory metrics, official SI
  boundary F1, per-image CSV, scene-type results where saved, and independent
  three-sample formula audits (`all_passed: true`).
- `*_C_cuda`: paired full-split RAW/refined scores with source prediction hashes,
  frozen parameters and explicit accuracy/boundary guards. No test tuning.
- `verified_model_assets.json`: exact hashes and sizes of all required model
  files, also recorded in `data/manifests/full_model_assets.json` with source refs.

The raw predictions, runtime environments and full datasets remain in the local
`outputs/full_benchmark_20261029/` and `data/` directories. Public notes use
device aliases. Evaluation reruns used the same saved GPU predictions after
fixing official boundary-source discovery; model inference was not repeated.

All 654 images are scored; the final 32-panel visualization export reserves the
three audit positions and then selects descending AbsRel. Full failure
classification remains available locally. This changes display cost only.
Regression tests: 101 passed.

See [results, interpretation and reproduction](../../full_benchmark_20261029.md).
