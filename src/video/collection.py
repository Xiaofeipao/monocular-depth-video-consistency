"""Summarize independently verified clips without pooling dev/test or sources."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common import create_output_directory, read_json, sha256_file, write_json


def summarize_evaluations(evaluations: list[Path], output_dir: Path) -> Path:
    from .evaluation import verify_evaluation
    if not evaluations:
        raise ValueError("At least one evaluation is required")
    directories = [Path(path).resolve() for path in evaluations]
    if len(set(directories)) != len(directories):
        raise ValueError("Duplicate evaluation directory")
    rows, records, identities, sessions = [], [], set(), {}
    for directory in directories:
        verification = verify_evaluation(directory, verify_sources=True)
        summary_path = directory / "summary.json"
        result = read_json(summary_path)
        clip = result["clip"]
        manifest = read_json(Path(result["sources"]["manifest_path"]))
        source = manifest["source"]
        identity = source["sha256"]
        if identity in identities:
            raise ValueError("Repeated source recording: do not count adjacent cuts as independent clips")
        identities.add(identity)
        session = source.get("provenance", {}).get("recording_session")
        session_key = (clip["source_kind"], session)
        if session and session_key in sessions and sessions[session_key] != clip["split"]:
            raise ValueError("The same recording session appears in development and test")
        sessions[session_key] = clip["split"]
        for model in ("dav2", "vda"):
            agg = result["aggregate"][model]
            rows.append({"clip_id": clip["clip_id"], "source_kind": clip["source_kind"],
                         "camera_motion": clip["camera_motion"], "split": clip["split"],
                         "model": model, "frames": result["frames_count"],
                         "mean_pair_normalized_mae": agg["mean_pair_normalized_mae"],
                         "mean_pair_normalized_rmse": agg["mean_pair_normalized_rmse"],
                         "mean_coverage": result["coverage"]["mean"],
                         "empty_pairs": result["coverage"]["empty_pairs"],
                         "low_coverage_pairs": result["coverage"]["low_coverage_pairs"]})
        records.append({"summary_path": str(summary_path), "sha256": sha256_file(summary_path),
                        "manifest_sha256": result["sources"]["manifest_sha256"], "verified": verification})
    groups = []
    for kind, split, motion, model in sorted({(r["source_kind"], r["split"], r["camera_motion"], r["model"]) for r in rows}):
        selected = [r for r in rows if (r["source_kind"], r["split"], r["camera_motion"], r["model"]) == (kind, split, motion, model)]
        valid = [r for r in selected if r["mean_pair_normalized_mae"] is not None]
        groups.append({"source_kind": kind, "split": split, "camera_motion": motion, "model": model,
                       "clip_count": len(selected), "evaluable_clips": len(valid),
                       "clip_equal_mean_normalized_mae": float(np.mean([r["mean_pair_normalized_mae"] for r in valid])) if valid else None,
                       "clip_equal_mean_normalized_rmse": float(np.mean([r["mean_pair_normalized_rmse"] for r in valid])) if valid else None})
    output = create_output_directory(output_dir)
    csv_path = output / "clips.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    value = {"schema_version": "relative_video_collection_v1", "status": "complete",
             "aggregation": "equal_clip_within_source_kind_split_camera_motion; no pooled dev/test",
             "groups": groups, "sources": records, "clips_csv_sha256": sha256_file(csv_path),
             "course_completion": "not_automatically_asserted; verify self-recorded fixed/moving, mandatory clips and final submission separately",
             "independent_units": "recording clips; adjacent frame pairs are not independent scenes",
             "statistical_significance_claimed": False}
    write_json(output / "summary.json", value)
    return output / "summary.json"
