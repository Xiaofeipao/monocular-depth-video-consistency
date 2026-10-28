# Official source and local checkpoints

Use separate processes for each adapter because the official repositories have
overlapping import names. The project never implicitly downloads weights. It
records the source Git revision and SHA256 of the supplied checkpoint on each run.
Use trusted official sources and respect each model's license.

Prepare sources and dependencies yourself in a suitable Conda environment:

```powershell
git clone https://github.com/DepthAnything/Depth-Anything-V2 third_party/Depth-Anything-V2
git clone https://github.com/DepthAnything/Video-Depth-Anything third_party/Video-Depth-Anything
git clone https://github.com/apple/ml-depth-pro third_party/ml-depth-pro
```

Install each official repository's dependencies, preferably in separate
environments if version requirements conflict. This project's core only needs
NumPy, OpenCV and Pillow. Python 3.11 is preferable to 3.13 for the official
model stacks. Record `git -C third_party/<repo> rev-parse HEAD` and pin that
revision for repeated experiments. Do not claim a tested model version until
its one-image/eight-frame smoke test passes on your machine.

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2):
  download `depth_anything_v2_vits.pth` from the official README. Base relative
  checkpoints predict inverse-depth-like relative values, not metres.
- [Depth Pro](https://github.com/apple/ml-depth-pro): download `depth_pro.pt`
  using the official `get_pretrained_models.sh` instructions. Adapter uses the
  official transform and FOV prediction; no test focal length is supplied.
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything):
  use `video_depth_anything_vits.pth` (relative) or
  `metric_video_depth_anything_vits.pth` (metric, requires `--metric`). Checkpoint
  filenames alone cannot prove training provenance: retain the download source.
  Official window stitching and keyframes are retained. Small and Base are
  supported; no Large model is exposed in the project CLI.
- [Torchvision RAFT Small](https://docs.pytorch.org/vision/stable/models/raft.html):
  pass a local torchvision-compatible state dict via `--flow-checkpoint`.
  Preprocessing uses RGB in [-1,1], images rounded up to a multiple of 8 and
  at least 128x128, then rescales flow vectors back to the saved frame grid.

Minimum model check: run one image (DAV2/Depth Pro), eight video frames (VDA),
confirm shape/finite output, and inspect the source RGB/depth pairing. Real
model checks remain pending until official sources and weights are available.
