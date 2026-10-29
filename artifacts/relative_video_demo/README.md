# Saved relative-video reproduction bundle

The neutral demo name is unrelated to a user's address. The provenance record
inside the ZIP identifies the upstream example video, model versions and hashes.
This is an inference reproduction asset, not the final course depth_results.zip.

The archive is split into files smaller than GitHub's individual file limit.
Clone/download **all three parts** and `archive_manifest.json`, then from the
repository root run:

```bash
python scripts/restore_relative_video_bundle.py
```

The script checks each part, the complete ZIP SHA256 and all ZIP CRCs. Extract
`outputs/relative_video_demo_restored.zip`, enter `relative_video_demo`, then:

```bash
python -m pip install -r requirements-replay.txt
python scripts/compare_relative_video.py --dav2 runs/dav2_relative --vda runs/vda_relative --output replay_scoring
```

Includes two complete original 32-frame runs (NPY/RGB/valid/config/environment/
manifest/timing/MP4), executable scoring code, 31-pair CSV, common masks, reference
frame and complete flow/mask settings, and original inference commands.
Machine-specific environment identity and paths are generalized; numerical
observations, saved arrays and videos are preserved. Exact replay requires no GPU.

The scoring script was extracted and verified during export; historical inline
code was not previously saved as a standalone script. Reported means reproduce
exactly: DAV2 0.006369762085077744, VDA 0.0042277539539481365.
The manifest provides the complete archive SHA256 and the hashes of all parts.
