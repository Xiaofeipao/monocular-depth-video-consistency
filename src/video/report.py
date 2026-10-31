"""Fixed-scale video comparisons and diagnostic figures from saved relative raw.

The generated assets are presentations of an existing paired evaluation, never
new predictions.  Depth display uses a single range for the entire sequence;
the normalized display uses each model's *fixed reference-frame* unit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .evaluation import (
    MODELS, _artifact, _equivalent, _load_sources, _path_inside, _read_csv, _read_json,
    _safe_warp, _sha, _write_json,
)


def colorize_relative(values: np.ndarray, limits: tuple[float, float] | list[float],
                      valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """TURBO RGB with large inverse-depth-like values near; finite zero is valid."""
    low, high = map(float, limits)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("A fixed finite increasing display range is required")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Colorized depth must be two-dimensional")
    valid = np.isfinite(array)
    if valid_mask is not None:
        if np.asarray(valid_mask).shape != array.shape:
            raise ValueError("Display mask shape mismatch")
        valid &= np.asarray(valid_mask, dtype=bool)
    normalized = np.clip((np.where(valid, array, low) - low) / (high - low), 0, 1)
    byte = np.rint(normalized * 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(byte, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    pixels = int(valid.sum())
    below, above = int((valid & (array < low)).sum()), int((valid & (array > high)).sum())
    return rgb, {"valid_pixels": pixels, "invalid_pixels": int(array.size - pixels),
                 "below_range_pixels": below, "above_range_pixels": above,
                 "below_range_fraction": below / pixels if pixels else None,
                 "above_range_fraction": above / pixels if pixels else None}


def _label(rgb: np.ndarray, text: str) -> np.ndarray:
    result = np.array(rgb, dtype=np.uint8, copy=True)
    height, width = result.shape[:2]
    # Header is drawn into a separate row by _panels. This helper supports
    # diagnostics saved at very small smoke-test resolution as well.
    scale = min(0.55, max(0.18, width / 700))
    cv2.putText(result, text, (5, max(12, min(height - 3, 22))), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _panels(images: list[np.ndarray], labels: list[str], footer: str) -> np.ndarray:
    if not images or len(images) != len(labels):
        raise ValueError("Each comparison panel requires one label")
    height, width = images[0].shape[:2]
    if any(image.shape != (height, width, 3) for image in images):
        raise ValueError("Comparison panels must share RGB dimensions")
    header_height = 30
    columns = []
    for image, label in zip(images, labels):
        header = _label(np.zeros((header_height, width, 3), dtype=np.uint8), label)
        columns.append(np.concatenate((header, image), axis=0))
    joined = np.concatenate(columns, axis=1)
    footer_image = _label(np.zeros((header_height, joined.shape[1], 3), dtype=np.uint8), footer)
    return np.concatenate((joined, footer_image), axis=0)


def _write_video(path: Path, frames, fps: float) -> dict:
    writer = None
    count, shape = 0, None
    try:
        for frame in frames:
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Video frames must be uint8 RGB")
            # Many mp4v encoders silently crop odd dimensions. Pad explicitly
            # and record the resulting dimensions instead of losing a column.
            padded = cv2.copyMakeBorder(frame, 0, frame.shape[0] % 2, 0, frame.shape[1] % 2,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if writer is None:
                shape = padded.shape[:2]
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (shape[1], shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV mp4v encoder could not open the report video")
            elif padded.shape[:2] != shape:
                raise ValueError("Report video frame dimensions changed")
            writer.write(cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
            count += 1
    finally:
        if writer is not None:
            writer.release()
    if not count:
        raise ValueError("Cannot write an empty comparison video")
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            decoded += 1
    finally:
        capture.release()
    if decoded != count:
        raise RuntimeError(f"Report video reload has {decoded} frames, expected {count}")
    return {"frames": count, "decoded_frames_checked": decoded, "fps": float(fps),
            "height": shape[0], "width": shape[1], "codec": "mp4v",
            "playback": "constant_fps_presentation; original per-frame timestamps are overlaid and retained in manifest"}


def _figure(path: Path, figsize=(11, 12)):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize, constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure


def _finite_curve(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([np.nan if row[key] is None else row[key] for row in rows], dtype=float)


def _curves(output: Path, summary: dict, pair_rows: list[dict], frame_rows: list[dict]) -> Path:
    path = output / "curves.png"
    figure = _figure(path)
    axes = figure.subplots(7, 1, sharex=True)
    time = _finite_curve(frame_rows, "timestamp_seconds")
    pair_time = _finite_curve(pair_rows, "timestamp_seconds")
    for name, color, title in zip(MODELS, ("#1f77b4", "#d62728"), ("DA-V2-S", "VDA-S")):
        for ax, suffix in zip(axes[:4], ("raw_median", "normalized_median", "normalized_median_delta", "median_ratio_to_reference")):
            ax.plot(time, _finite_curve(frame_rows, f"{name}_{suffix}"), label=title, color=color)
        axes[4].plot(pair_time, _finite_curve(pair_rows, f"{name}_normalized_mae"), label=title, color=color)
        axes[5].plot(pair_time, _finite_curve(pair_rows, f"{name}_normalized_rmse"), label=title, color=color)
    axes[6].plot(pair_time, _finite_curve(pair_rows, "coverage"), color="#333333", label="shared valid coverage")
    axes[6].axhline(summary["protocol"]["minimum_coverage"], color="red", linestyle="--", label="low-coverage quality flag")
    for ax, ylabel in zip(axes, ("Raw median\n(arbitrary)", "Median in fixed\nreference units", "Median change /\nfixed span", "Raw median /\nreference median", "Flow-aligned MAE /\nfixed span", "Flow-aligned RMSE /\nfixed span", "Shared mask fraction")):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        for boundary in summary["window_boundaries"]["boundaries"]:
            ax.axvline(time[boundary["window_start"]], color="gray", linestyle=":", alpha=0.5)
    axes[-1].set_xlabel("Source-relative timestamp (seconds)")
    axes[-1].set_ylim(-0.02, 1.05)
    figure.suptitle(f"{summary['clip']['clip_id']}: stability diagnostics, no per-frame alignment\nDotted lines: VDA window-start context, not assumed exact output joins")
    figure.savefig(path, dpi=140)
    figure.clear()
    return path


def _source_binding(summary: dict) -> dict:
    files = summary["sources"]["files"]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"manifest_sha256": summary["sources"]["manifest_sha256"], "source_files_sha256": digest,
            "run_json_sha256": {name: next(item["sha256"] for item in files if item["role"] == f"{name}_run") for name in MODELS},
            "reference": summary["reference"],
            "pair_csv_sha256": next(item["sha256"] for item in summary["artifacts"] if item["role"] == "pair_metrics"),
            "frame_csv_sha256": next(item["sha256"] for item in summary["artifacts"] if item["role"] == "frame_metrics")}


def generate_report(summary_path: str | Path, *, config: dict | None = None) -> dict:
    """Generate synchronized fixed-scale views from verified evaluation inputs.

    A pre-existing ``report`` directory is rejected. ``generate_video:false``
    leaves plots/contact sheets/saturation accounting enabled. No model, flow
    estimator or external network is used.
    """
    summary_path = Path(summary_path).resolve()
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("Only complete evaluation summaries can be visualized")
    config = summary.get("config", {}) if config is None else config
    visualization = config.get("visualization", {})
    if visualization.get("colormap", "turbo").lower() != "turbo":
        raise ValueError("The comparison protocol uses one common TURBO colormap")
    raw_range = visualization.get("raw_range", [0.0, 50.0])
    normalized_range = visualization.get("normalized_range", [0.0, 1.0])
    difference_range = visualization.get("difference_range", [0.0, 0.1])
    # Validate all ranges before creating a directory or a video.
    for limits in (raw_range, normalized_range, difference_range):
        if len(limits) != 2:
            raise ValueError("Display ranges must contain [low, high]")
        colorize_relative(np.zeros((1, 1)), limits)
    sources = _load_sources(summary["sources"]["manifest_path"], summary["sources"]["runs"]["dav2"],
                            summary["sources"]["runs"]["vda"], summary["sources"]["flow_cache"])
    if sources["manifest_sha256"] != summary["sources"]["manifest_sha256"]:
        raise ValueError("Report inputs differ from the evaluated manifest")
    _equivalent(sources["source_files"], summary["sources"]["files"], "report raw/RGB/flow source bindings")
    for item in summary["artifacts"]:
        path = _path_inside(summary_path.parent, item["path"])
        if _sha(path) != item["sha256"]:
            raise ValueError(f"Report source artifact changed: {path}")
    output = summary_path.parent / "report"
    output.mkdir(exist_ok=False)
    pair_rows = _read_csv(summary_path.parent / "pair_metrics.csv")
    frame_rows = _read_csv(summary_path.parent / "frame_metrics.csv")
    asset_records = []

    def add(path: Path, role: str, **extra) -> None:
        asset_records.append({**_artifact(summary_path.parent, path), "role": role, **extra})

    add(_curves(output, summary, pair_rows, frame_rows), "curves")
    raw, manifest = sources["raw"], sources["manifest"]
    saturation = {view: {name: {"valid_pixels": 0, "invalid_pixels": 0, "below_range_pixels": 0,
                              "above_range_pixels": 0} for name in MODELS} for view in ("raw", "normalized")}

    def rgb(i: int) -> np.ndarray:
        path = _path_inside(sources["manifest_path"].parent, manifest["frames"][i]["rgb_path"])
        return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def depth_view(name: str, i: int, view: str, *, count: bool = False) -> np.ndarray:
        values = np.asarray(raw[name][i], dtype=np.float64)
        limits = raw_range
        if view == "normalized":
            ref = summary["reference"][name]
            values = (values - ref["percentile_low_value"]) / ref["span"] if ref["span"] else np.full(values.shape, np.nan)
            limits = normalized_range
        image, stats = colorize_relative(values, limits)
        if count:
            for key in saturation[view][name]:
                saturation[view][name][key] += stats[key]
        return image

    def frame_panels(view: str):
        limits = raw_range if view == "raw" else normalized_range
        for i, frame in enumerate(manifest["frames"]):
            label = "raw arbitrary" if view == "raw" else "fixed reference units"
            yield _panels([rgb(i), depth_view("dav2", i, view), depth_view("vda", i, view)],
                          ["RGB", f"DA-V2-S | {label}", f"VDA-S | {label}"],
                          f"frame={i} source={frame.get('source_index')} t={frame['timestamp_seconds']:.4f}s | TURBO near=high | fixed range {limits}")

    def difference_panels(i: int) -> np.ndarray:
        mask_record = next(item for item in summary["artifacts"] if item["role"] == "pair_mask" and item["current_index"] == i)
        mask = np.load(summary_path.parent / mask_record["path"], allow_pickle=False)
        images = [np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)]
        for name in MODELS:
            span = summary["reference"][name]["span"]
            warped, _ = _safe_warp(raw[name][i - 1], sources["flows"][i - 1][1])
            difference = np.abs(np.asarray(raw[name][i], dtype=np.float64) - warped) / span if span else np.full(mask.shape, np.nan)
            images.append(colorize_relative(difference, difference_range, mask)[0])
        row = pair_rows[i - 1]
        return _panels(images, ["Shared mask | white=valid", "DA-V2-S | abs warp difference", "VDA-S | abs warp difference"],
                       f"pair {i - 1}->{i} t={row['timestamp_seconds']:.4f}s | coverage={row['coverage']:.3f} | range={difference_range} fixed reference units")

    for i in range(len(manifest["frames"])):
        for view in ("raw", "normalized"):
            for name in MODELS:
                depth_view(name, i, view, count=True)
    for models in saturation.values():
        for item in models.values():
            pixels = item["valid_pixels"]
            item["below_range_fraction"] = item["below_range_pixels"] / pixels if pixels else None
            item["above_range_fraction"] = item["above_range_pixels"] / pixels if pixels else None
    if visualization.get("generate_video", True):
        for view in ("raw", "normalized"):
            path = output / f"comparison_{view}.mp4"
            add(path, f"comparison_{view}_video", **_write_video(path, frame_panels(view), manifest["fps"]))
        path = output / "flow_difference.mp4"
        add(path, "flow_difference_video", **_write_video(path, (difference_panels(i) for i in range(1, len(manifest["frames"]))), manifest["fps"]))
    count = len(manifest["frames"])
    indices = sorted({0, count // 2, count - 1})
    for view in ("raw", "normalized"):
        panels = [_panels([rgb(i), depth_view("dav2", i, view), depth_view("vda", i, view)],
                          ["RGB", f"DA-V2-S | {view}", f"VDA-S | {view}"],
                          f"frame={i} source={manifest['frames'][i].get('source_index')} t={manifest['frames'][i]['timestamp_seconds']:.4f}s") for i in indices]
        path = output / f"contact_{view}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(panels, axis=0), cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write contact sheet: {path}")
        add(path, f"contact_{view}", frame_indices=indices)
    worst = [item["current_index"] for item in summary["worst_pairs"][:3]]
    if worst:
        path = output / "worst_pairs.png"
        panels = [difference_panels(i) for i in worst]
        if not cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(panels, axis=0), cv2.COLOR_RGB2BGR)):
            raise RuntimeError("Could not write failure contact sheet")
        add(path, "worst_pair_contact", current_indices=worst)
    background_assets = [item for item in summary["artifacts"] if item["role"].endswith("_background_std")]
    if background_assets:
        path = output / "background_std.png"
        figure = _figure(path, (11, 4))
        axes = figure.subplots(1, 2)
        maps = [np.load(summary_path.parent / item["path"], allow_pickle=False) for item in background_assets]
        finite_max = max(float(np.nanmax(value)) for value in maps)
        for ax, item, values in zip(axes, background_assets, maps):
            image = ax.imshow(values, vmin=0, vmax=max(finite_max, 1e-6), cmap="turbo")
            ax.set_title(item["role"].split("_")[0].upper() + " background temporal std")
            ax.axis("off")
            figure.colorbar(image, ax=ax, label="Fixed reference units; population std")
        figure.savefig(path, dpi=140)
        figure.clear()
        add(path, "background_std_figure")
    def number(value, digits=8):
        return "不可评价" if value is None else f"{value:.{digits}f}"

    notes = [f"# {summary['clip']['clip_id']}：Part 1B / 2B 对照", "",
             f"来源类别：`{summary['clip'].get('source_kind')}`；相机条件：`{summary['clip'].get('camera_motion')}`；共 {summary['frames_count']} 帧。", "",
             "这些数值是无深度 GT 的稳定性诊断。相机运动、真实物体变化和光流误差也会影响结果，不能据此声称深度准确率或真实米制尺度改善。", "",
             "| 方法 | 逐对平均归一化 MAE↓ | 逐对平均归一化 RMSE↓ | 按像素加权 MAE↓ | 模型 FPS |", "|---|---:|---:|---:|---:|"]
    for name, title in zip(MODELS, ("DA-V2-S（逐帧）", "VDA-S（离线视频）")):
        metrics = summary["aggregate"][name]
        notes.append(f"| {title} | {number(metrics['mean_pair_normalized_mae'])} | {number(metrics['mean_pair_normalized_rmse'])} | {number(metrics['pixel_weighted_normalized_mae'])} | {number(summary['timing'][name].get('model_fps'), 3)} |")
    coverage = summary["coverage"]
    notes += ["", "两模型使用完全相同的逐对掩码；每个模型的参考帧 P95−P5 数值跨度固定到整段结束，不逐帧重新对齐或拉伸。主指标平均全部非空帧对；低覆盖率只标记质量，不用于挑选更好看的主分数。", "",
              f"共 {coverage['pair_count']} 对，空掩码 {coverage['empty_pairs']} 对，低覆盖率 {coverage['low_coverage_pairs']} 对；平均覆盖率 {coverage['mean']:.2%}，最低 {coverage['minimum']:.2%}。", "",
              f"VDA 的逐对平均归一化 MAE 相对 DA 降低：{number(summary['comparison']['vda_mean_pair_mae_reduction_percent'], 4)}%；负值表示 VDA 更差。", "",
              "`pair_metrics.csv` 保存每对分数及排除像素计数；`frame_metrics.csv` 保存原始中位数、固定参考单位中的中位数变化和可用时的中位数比值。", "",
              "`comparison_raw.mp4` 使用共享原始数值色标；`comparison_normalized.mp4` 使用固定参考单位色标。颜色范围外只影响显示，饱和比例保存在 `report.json`；原始预测与评价没有裁剪。未请求视频时仍生成 PNG。", "",
              "`flow_difference.mp4` 和 `worst_pairs.png` 展示共同有效掩码与归一化对应位置差异。曲线中的灰色竖线是窗口输入起点；`summary.json` 另列实际混合区间、后续输出起点及附近帧对分数。", "",
              "固定背景闪烁仅在相机固定且提供预先声明的背景 ROI 时评价。无 ROI 或相机运动时，不把全画面的波动称作固定背景闪烁。", "",
              "效率来自本次实际运行记录。VDA 的整段吞吐和摊销成本不等同于在线逐帧响应；无预热或重复次数不足的短运行仅作为工程检查。完整精度、参考数值跨度、来源 SHA、模型与光流元数据见上一级 `summary.json`。", ""]
    path = output / "analysis.md"
    path.write_text("\n".join(notes), encoding="utf-8")
    add(path, "plain_language_analysis")
    result = {"binding": _source_binding(summary), "colormap": "turbo", "near_direction": "high",
              "raw_range": list(map(float, raw_range)), "normalized_range": list(map(float, normalized_range)),
              "difference_range": list(map(float, difference_range)), "per_frame_rescaling": False,
              "visualization_only_clipping": True, "invalid_display": "black", "saturation": saturation,
              "video_requested": bool(visualization.get("generate_video", True)), "assets": asset_records}
    metadata = output / "report.json"
    _write_json(metadata, result)
    # Metadata records its assets; the parent summary additionally hashes this
    # metadata file. Avoid a self-hash or rewriting a completed asset manifest.
    result["assets"] = asset_records + [{**_artifact(summary_path.parent, metadata), "role": "report_metadata"}]
    return result
