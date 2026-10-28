# TUM RGB-D fr1/xyz selected public sample

Authors: Juergen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, Daniel Cremers.
Source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
Citation: A Benchmark for the Evaluation of RGB-D SLAM Systems, IROS 2012.
Original publication describes CC BY 3.0: https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf
License: https://creativecommons.org/licenses/by/3.0/
Not covered by this repository's MIT code license.

Sixteen registered RGB/depth PNG pairs are selected without changing PNG payloads.
Depth is divided by 5000 for metres; zero is missing. Source timestamp indexes
and groundtruth trajectory are preserved. Selected poses in manifest.json use
translation interpolation and quaternion SLERP at RGB timestamps. ROS default
intrinsics are explicitly assumed for registered PNG geometry as recommended by TUM.
This is an optional public smoke subset, not a mandatory course clip or phone video.
