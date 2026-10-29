# Local datasets

The Spring 2027 mandatory datasets are Middlebury quarter-resolution and
NYUv2 labeled with the standard split; keep self-recorded videos here as well.
Full datasets stay local. Selected KITTI/TUM smoke samples retain upstream
attribution and licenses; see `public_samples/README.md`. NYUv2 raw samples are
locally downloaded, with only source manifests/hashes shared. Use the provided split/crop/mask protocol rather
than substituting an entire public dataset for a course sample.

Image manifests contain `protocol` and an ordered `samples` list, with unique
`id`, `rgb`, `depth`, optional `mask`, and `depth_scale` (stored value times this
factor gives metres). Paths are relative to the manifest. NPY depth is already
in metres; official KITTI 16-bit depth PNG uses 1/256. Missing depth is zero.
Do not use an 8-bit colour visualization as ground truth.

See `configs/nyuv2.example.json` and `configs/kitti.example.json`. Sample crops
are examples, not a claim that every supplied course sample uses that protocol.
Run `python -m src.prepare_data validate --manifest data/manifest.json` first.

Middlebury scenes must contain `im0.png`, `im1.png`, `calib.txt`, and optionally
`disp0GT.pfm`, `mask0nocc.png`. The images must already be rectified. Baseline
in Middlebury calibration is converted from millimetres to metres; doffs is used.

Optional full NYUv2 conversion (requires scipy and h5py in the active environment):

```powershell
python -m src.prepare_data nyu --mat data/nyu_depth_v2_labeled.mat --split data/splits.mat --output data/nyu654
```

`splits.mat` must supply the official 654 unique 1-based `testNdxs`. This
exporter uses the standard 480x640 labeled MAT axis layout, preserves full
images for focal inference and applies crop [45,471,41,601] only in evaluation.
Source MAT and split SHA256 are recorded. No dataset is downloaded implicitly.
