# Public smoke data and provenance

These are small **public software-check samples**, not the instructor's NYUv2/KITTI
subsets, mandatory static/moving clips, or a student-recorded phone video.
They do not establish completion of the course's dataset requirements.

- `nyuv2/`: metadata/hashes for four deterministic standard-test images. The
  preparation script downloads RGB and metric GT locally, using range reads from
  the canonical MAT. Raw NYUv2 arrays are excluded from Git because the source
  page requests citation but does not specify a redistribution license.
- `kitti/`: four unmodified RGB/16-bit GT PNG pairs from the official
  `val_selection_cropped` depth-selection archive. This is not the Eigen split.
  Depth scale is 1/256 metres, missing GT is zero, max evaluation depth is 80 m,
  and no second crop is applied to already-cropped input.
- `tum_freiburg1_xyz/`: sixteen selected registered RGB/depth frames and associated
  interpolated camera poses. Depth scale is 1/5000 metres. Original timestamps,
  archive SHA256, source filenames and per-image hashes are preserved. This short
  optional RGB-D subset is not a substitute for the required phone recording.

Data retain upstream terms and are **not** covered by this repository's MIT
code license. See each dataset's notice. Full archives and range caches are local
only under `data/downloads/` and are never staged to Git.

```powershell
python scripts/prepare_public_benchmarks.py --dataset nyuv2 --count 4 --output data/public_samples/nyuv2_local
python scripts/prepare_public_benchmarks.py --dataset kitti --count 4 --output data/public_samples/kitti_local
python scripts/prepare_tum.py --download --range-download --sequence freiburg1_xyz --frames 16 --output data/public_samples/tum_local
```

The public image protocol is recorded in `manifest.json`: NYUv2 full RGB,
half-open evaluation crop [45,471,41,601], exclusive 0.001–10 m GT and score-time
prediction clipping; KITTI uses the official already-cropped validation selection.
These public-sample settings must not override a subsequently supplied instructor
protocol. For teacher data, provide a separate manifest with the supplied crop,
mask, depth encoding, min/max bounds and split.
