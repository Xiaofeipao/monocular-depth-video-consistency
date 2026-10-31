"""Single-model fixed-scale plots and videos for Part 1B, from saved raw."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .common import sha256_file, write_json
from .evaluation import _artifact, _path_inside, _safe_warp
from .report import _figure, _panels, _write_video, colorize_relative


def report_binding(summary):
    return {"source_files_sha256": hashlib.sha256(json.dumps(summary["sources"]["files"], sort_keys=True).encode()).hexdigest(),
            "csv_sha256": {a["role"]: a["sha256"] for a in summary["artifacts"] if a["path"].endswith(".csv")},
            "reference": summary["reference"], "visualization": summary["config"]["visualization"]}


def generate_single_report(summary_path: Path, sources: dict, pairs: list[dict], frames: list[dict]):
    root = Path(summary_path).parent
    output = root / "report"
    output.mkdir(exist_ok=False)
    from .common import read_json
    summary = read_json(summary_path)
    settings = summary["config"]["visualization"]
    if settings.get("colormap", "turbo") != "turbo":
        raise ValueError("Only the frozen TURBO display is implemented")
    ranges = {"raw": settings.get("raw_range", [0, 50]), "fixed_reference": settings.get("normalized_range", [0, 1]),
              "difference": settings.get("difference_range", [0, 0.1])}
    for limits in ranges.values():
        if len(limits) != 2:
            raise ValueError("Display range must have two endpoints")
        colorize_relative(np.zeros((1, 1)), limits)
    raw, manifest = sources["raw"], sources["manifest"]
    reference = summary["reference"]
    span, low = reference["span"], reference["percentile_low_value"]
    artifacts = []

    def add(path, role, **extra):
        artifacts.append({**_artifact(root, path), "role": role, **extra})

    times = [r["timestamp_seconds"] for r in frames]
    pair_times = [r["timestamp_seconds"] for r in pairs]
    figure = _figure(output / "curves.png", (11, 13))
    axes = figure.subplots(6, 1, sharex=True)
    for ax, key, label in zip(axes[:3], ("raw_median", "normalized_median_delta", "median_ratio"),
                             ("Raw median (arbitrary)", "Median change / fixed span", "Raw median / reference median")):
        ax.plot(times, [np.nan if r[key] is None else r[key] for r in frames])
        ax.set_ylabel(label)
    for key, label in (("normalized_mae", "MAE / fixed span"), ("normalized_rmse", "RMSE / fixed span")):
        axes[3].plot(pair_times, [np.nan if r[key] is None else r[key] for r in pairs], label=label)
    axes[3].set_ylabel("Flow-aligned change")
    axes[3].legend()
    axes[4].plot(pair_times, [r["coverage"] for r in pairs])
    axes[4].axhline(summary["config"]["evaluation"]["minimum_coverage"], linestyle="--", color="red", label="quality flag only")
    axes[4].set_ylabel("Valid coverage")
    axes[4].set_ylim(-0.02, 1.05)
    axes[4].legend()
    if summary["background_flicker"]["applicable"]:
        axes[5].plot(pair_times, [np.nan if r["background_adjacent_normalized_mae"] is None else r["background_adjacent_normalized_mae"] for r in pairs])
    else:
        axes[5].text(0.05, 0.5, "N/A: fixed camera + predeclared background ROI required", transform=axes[5].transAxes)
    axes[5].set_ylabel("Background change\n/ fixed span")
    axes[-1].set_xlabel("Source timestamp (seconds); scores per selected frame pair")
    for ax in axes:
        ax.grid(alpha=0.25)
    figure.suptitle(f"{manifest['clip_id']}: DA-V2-S framewise diagnostics\nRelative values; no per-frame alignment; no depth GT")
    for extension in ("png", "pdf"):
        path = output / f"curves.{extension}"
        figure.savefig(path, dpi=150)
        add(path, "curves_" + extension)
    figure.clear()

    def rgb(i):
        path = _path_inside(sources["manifest_path"].parent, manifest["frames"][i]["rgb_path"])
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

    saturation = {}

    def depth(i, view):
        value = np.asarray(raw[i], dtype=np.float64)
        if view == "fixed_reference":
            value = (value - low) / span if span else np.full(value.shape, np.nan)
        return colorize_relative(value, ranges[view])

    for view in ("raw", "fixed_reference"):
        totals = {k: 0 for k in ("valid_pixels", "invalid_pixels", "below_range_pixels", "above_range_pixels")}
        for i in range(len(raw)):
            _, stats = depth(i, view)
            for key in totals:
                totals[key] += stats[key]
        saturation[view] = {**totals, "below_range_fraction": totals["below_range_pixels"] / totals["valid_pixels"] if totals["valid_pixels"] else None,
                            "above_range_fraction": totals["above_range_pixels"] / totals["valid_pixels"] if totals["valid_pixels"] else None}

    def frame_panel(i, view):
        f = manifest["frames"][i]
        return _panels([rgb(i), depth(i, view)[0]], ["RGB", "DA-V2-S | " + view],
                       f"frame={i} source={f['source_index']} t={f['timestamp_seconds']:.4f}s | TURBO near=high fixed={ranges[view]}")

    def difference_panel(i):
        warped, _ = _safe_warp(raw[i - 1], sources["flows"][i - 1][1])
        mask = np.load(root / f"masks/pair_{i - 1:06d}_{i:06d}.npy", allow_pickle=False)
        delta = np.abs(np.asarray(raw[i], dtype=np.float64) - warped) / span if span else np.full(mask.shape, np.nan)
        difference = colorize_relative(delta, ranges["difference"], mask)[0]
        return _panels([rgb(i), difference, np.repeat(mask[..., None], 3, axis=2).astype(np.uint8) * 255],
                       ["Current RGB", "Absolute warp difference / span", "Valid mask: white"],
                       f"pair={i-1}->{i} t={pairs[i-1]['timestamp_seconds']:.4f}s coverage={pairs[i-1]['coverage']:.3f} range={ranges['difference']}")

    if settings.get("generate_video", True):
        for view in ("raw", "fixed_reference"):
            path = output / f"depth_{view}.mp4"
            add(path, view + "_video", **_write_video(path, (frame_panel(i, view) for i in range(len(raw))), manifest["fps"]))
        path = output / "temporal_difference.mp4"
        add(path, "difference_video", **_write_video(path, (difference_panel(i) for i in range(1, len(raw))), manifest["fps"]))
    contact = output / "contact_fixed_reference.png"
    indices = sorted({0, len(raw) // 2, len(raw) - 1})
    if not cv2.imwrite(str(contact), cv2.cvtColor(np.concatenate([frame_panel(i, "fixed_reference") for i in indices]), cv2.COLOR_RGB2BGR)):
        raise RuntimeError("Could not write contact sheet")
    add(contact, "contact", frame_indices=indices)
    worst = [r["current_index"] for r in summary["worst_pairs"][:3]]
    if worst:
        path = output / "worst_pairs.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(np.concatenate([difference_panel(i) for i in worst]), cv2.COLOR_RGB2BGR)):
            raise RuntimeError("Could not write worst-pair diagnostics")
        add(path, "worst_pairs", current_indices=worst)
    if (root / "background_normalized_std.npy").exists():
        values = np.load(root / "background_normalized_std.npy", allow_pickle=False)
        fig = _figure(output / "background_std.png", (8, 5))
        ax = fig.subplots()
        # Frozen scale across clips, not an automatic per-image contrast stretch.
        im = ax.imshow(values, cmap="turbo", vmin=0, vmax=0.1)
        ax.set_title("Predeclared background temporal std / fixed span")
        fig.colorbar(im, ax=ax, label="Population std, ddof=0; display [0, 0.1]")
        for ext in ("png", "pdf"):
            path = output / f"background_std.{ext}"
            fig.savefig(path, dpi=150)
            add(path, "background_std_" + ext)
        fig.clear()

    def number(value):
        return "不可评价" if value is None else f"{value:.8f}"

    aggregate, background = summary["aggregate"], summary["background_flicker"]
    notes = [f"# {manifest['clip_id']}：Part 1B逐帧DA-V2-S结果", "",
             f"来源：{manifest['source_kind']}；相机：{manifest['camera_motion']}；划分：{manifest['split']}；{len(raw)}帧。", "",
             "只分析DA-V2-S原生相对预测。以下是无GT的稳定性诊断，不能当作米制准确率或真实尺度漂移。", "",
             "| 项目 | 数值 |", "|---|---:|",
             f"| 逐对平均时序MAE / 固定跨度 | {number(aggregate['mean_pair_normalized_mae'])} |",
             f"| 逐对平均时序RMSE / 固定跨度 | {number(aggregate['mean_pair_normalized_rmse'])} |",
             f"| 平均有效覆盖率 | {summary['coverage']['mean']:.4%} |",
             f"| 中位数比值的平均绝对log变化 | {number(summary['median_summary']['mean_abs_log_ratio'])} |",
             f"| 背景时间标准差均值 / 固定跨度 | {number(background['normalized_std_mean'])} |", "",
             "尺子在参考帧确定后整段固定；合法零值保留，不取倒数，不逐帧对齐或重新归一化。所有非空帧对参与主均值，低覆盖仅标记，空区域记不可评价。", "",
             f"背景诊断适用：{background['applicable']}；不适用/失败原因：{background['unavailable_reason']}。背景ROI只用于背景诊断，不限制全图主时序误差或中位数的取样区域。", "",
             "运动相机会改变真实距离和视野组成，中位数曲线只是proxy。逐帧原生值、变化、比值和不可评价原因在frame_metrics.csv；逐对误差、覆盖和质量在pair_metrics.csv。", "",
             "depth_raw.mp4使用固定原生色标，depth_fixed_reference.mp4使用参考帧常数映射后的固定色标。颜色饱和只影响显示，原始数组与评分不裁剪。时间戳来自源视频；MP4按固定FPS展示，误差不是每秒误差。", "",
             "worst_pairs.png是时序误差较大的诊断帧对，不是有GT的错误标签。请人工检查遮挡、反光、细杆、运动模糊、边缘跳动，并记录代表性失败；自动说明不代替人工结论。", "",
             "效率和GPU显存来自run.json。无预热/单次低分辨率CPU运行只能说明接口可用，不是正式效率。verify会由原生数组与flow重新计算，不重新预测。", ""]
    path = output / "analysis.md"
    path.write_text("\n".join(notes), encoding="utf-8")
    add(path, "plain_language_analysis")
    result = {"binding": report_binding(summary), "saturation": saturation, "ranges": ranges,
              "per_frame_rescaling": False, "color_direction": "near_is_high", "colormap": "turbo",
              "assets": artifacts, "video_requested": settings.get("generate_video", True)}
    write_json(output / "report.json", result)
    return {**result, "assets": artifacts + [{**_artifact(root, output / "report.json"), "role": "report_metadata"}]}
