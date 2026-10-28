# NYUv2 locally prepared public samples

Authors: Nathan Silberman, Pushmeet Kohli, Derek Hoiem, Rob Fergus.
Source: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
Citation: "Indoor Segmentation and Support Inference from RGBD Images", ECCV 2012.

The first four IDs of the official MATLAB `testNdxs` split were extracted from the
canonical labeled MAT. Full RGB is preserved for focal inference; GT is float32
metres. IDs, source URLs, split SHA256, image/depth hashes and the public example
evaluation protocol are committed in `manifest.json`/`validation.json`.

The source webpage requests citation but does not specify a redistribution
license. **RGB/depth arrays remain local and are not included in Git.** Run
`scripts/prepare_public_benchmarks.py` to obtain matching arrays directly from the
official source. No new license or ownership over NYUv2 data is claimed.
This public smoke sample is not the instructor-provided subset.
