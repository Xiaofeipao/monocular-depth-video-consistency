# Part 3: frozen D development/heldout study

The custom project split uses `freiburg1_desk` for development and parameter
selection; `freiburg1_rpy`, `freiburg3_sitting_static`, and
`freiburg3_walking_xyz` are the new heldout sequences. This is not an official
TUM train/test split. Each has 64 synchronized frames. The four phone excerpts
and two previously studied xyz excerpts remain demonstrations/references and
do not enter parameter selection. Sequences can share physical environments.

`selection.json` identifies the frozen candidate and hashes the protocol and
development results. All candidates, alternatives and failures are retained.
`legacy_parity.json` checks default D and pose scoring against pre-study source
on deterministic regression inputs; this does not prove equality for all inputs.

Per-case exports include baseline metadata/config/manifest/timing, actual input
manifest, camera settings, software versions, method summaries, per-frame and
per-pair scores, and region diagnostics. No personal machine paths or hardware
model names are included. Source scripts and saved file checksums are separate.

The public `per_pair.csv` is an audited export with separate `flow_coverage` and
`pose_coverage`. The original local table merged the generic pose `coverage`
field into GT-scored Farneback rows; the audit reconstructs flow coverage from
the saved common mask and labels both explicitly. Depth/warp/pose scores are
unchanged. Original local CSVs remain identifiable in `checksums.csv`.

Compact local predictions are lossless float32 arrays in `predictions.npz`.
Each summary's `prediction_locations` maps methods to an NPZ key or a verified
immutable RunWriter directory; equal methods can share one source. RAW, RGB and
validity masks remain in the corresponding baseline. Every output is checked
positive and finite, so derived prediction validity equals baseline validity.
GT masks are distinct and come from the input manifest. Reconstruct arrays with
`scripts.analyze_d_research.load_predictions(folder, summary)`.

Region scores use fixed GT log-depth jumps >0.08 and a two-pixel boundary band.
`GT_not_rigidly_matched` includes new surfaces, dynamic objects, field-of-view
changes and sensor effects; it is not a moving-object segmentation GT. Phone
RAW-reference scores and photometric mismatch diagnostics cannot establish
metric accuracy or absence of ghosting.

Prerequisites are the VDA metric Small and RAFT Small checkpoints, their project
dependencies, the existing xyz archive, and the four local phone videos.

```powershell
./scripts/run_d_research.ps1 -Python python
```

The script prepares any missing six-clip 518 reference runs. New TUM full
archives are downloaded temporarily inside the project, hashed, and released
after the selected original PNGs/tables and resized subsets are validated.
Required project datasets and PDFs are preserved. No model is trained or
fine-tuned. Timing uses one warm-up and three repeats and distinguishes
fusion-only, flow-inclusive postprocessing, and the warmed offline file
pipeline. Scratch timing outputs are removed because study predictions are
already preserved. Floating-point bytes and timing can vary across systems;
source byte hashes can also vary with Git line endings.

Interpretation and limitations: [study report](../../d_research.md).

The complete test suite passed: 108 tests with pytest 8.4.2. Run
`python -m pytest tests -q --basetemp outputs/test_tmp_d_research` in the project
test environment. Test dependency installation can be isolated in
`.runtime_tests` and added to `PYTHONPATH`; it does not change inference settings.
