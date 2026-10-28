"""Report-ready Middlebury stereo visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_stereo_panel(
    output_path: Path,
    *,
    scene_name: str,
    method_name: str,
    rgb: np.ndarray,
    target_disparity_px: np.ndarray,
    prediction_disparity_px: np.ndarray,
    evaluation_mask: np.ndarray,
    raw_prediction_valid: np.ndarray,
    lr_consistent: np.ndarray,
    disparity_max_px: float,
) -> None:
    """Save RGB, GT, prediction, error, invalid, and L-R failure panels."""

    target = np.asarray(target_disparity_px)
    prediction = np.asarray(prediction_disparity_px)
    evaluation = np.asarray(evaluation_mask, dtype=bool)
    common = evaluation & np.isfinite(prediction)
    absolute_error = np.full(target.shape, np.nan, dtype=np.float32)
    absolute_error[common] = np.abs(prediction[common] - target[common])
    invalid = evaluation & ~np.asarray(raw_prediction_valid, dtype=bool)
    lr_failure = evaluation & raw_prediction_valid & ~np.asarray(lr_consistent, dtype=bool)

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = axes.ravel()
    panels[0].imshow(rgb)
    panels[0].set_title("RGB (left)")
    target_image = panels[1].imshow(target, cmap="turbo", vmin=0, vmax=disparity_max_px)
    panels[1].set_title("GT disparity (px)")
    prediction_image = panels[2].imshow(
        prediction, cmap="turbo", vmin=0, vmax=disparity_max_px
    )
    panels[2].set_title(f"{method_name} disparity (px)")
    error_image = panels[3].imshow(absolute_error, cmap="magma", vmin=0, vmax=4)
    panels[3].set_title("Absolute disparity error (0–4 px)")
    panels[4].imshow(invalid, cmap="gray", vmin=0, vmax=1)
    panels[4].set_title("Matcher-invalid in eval mask")
    panels[5].imshow(lr_failure, cmap="gray", vmin=0, vmax=1)
    panels[5].set_title("Left-right inconsistency")
    for axis in panels:
        axis.axis("off")
    figure.colorbar(target_image, ax=[panels[1], panels[2]], shrink=0.75)
    figure.colorbar(error_image, ax=panels[3], shrink=0.75)
    figure.suptitle(f"{scene_name} — {method_name}", fontsize=16)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def save_sensitivity_plot(output_path: Path, rows: list[dict[str, float | int]]) -> None:
    """Plot numerical and first-order depth error against GT distance."""

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    magnitudes = sorted({float(row["disparity_error_px"]) for row in rows})
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(magnitudes)))
    for magnitude, color in zip(magnitudes, colors):
        selected = [row for row in rows if row["disparity_error_px"] == magnitude]
        selected.sort(key=lambda row: float(row["depth_bin_center_m"]))
        distance = [float(row["depth_bin_center_m"]) for row in selected]
        exact = [float(row["mean_exact_abs_depth_error_m"]) for row in selected]
        first = [float(row["mean_first_order_abs_depth_error_m"]) for row in selected]
        axis.plot(distance, exact, color=color, label=f"±{magnitude:g} px exact")
        axis.plot(distance, first, color=color, linestyle="--", alpha=0.8)
    axis.set_xlabel("GT depth (m)")
    axis.set_ylabel("Mean absolute depth error (m)")
    axis.set_title("Disparity error sensitivity: exact (solid) vs first-order (dashed)")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_empirical_depth_error_plot(
    output_path: Path, rows: list[dict[str, float | int | str]]
) -> None:
    """Compare observed depth error and coverage for all stereo experiments."""

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    experiments = sorted({str(row["experiment"]) for row in rows})
    colors = plt.cm.tab10(np.linspace(0.0, 0.8, len(experiments)))
    for experiment, color in zip(experiments, colors):
        selected = [row for row in rows if row["experiment"] == experiment]
        selected.sort(key=lambda row: float(row["depth_bin_center_m"]))
        distance = [float(row["depth_bin_center_m"]) for row in selected]
        depth_error = [float(row["mean_abs_depth_error_m"]) for row in selected]
        coverage = [float(row["valid_coverage"]) for row in selected]
        axes[0].plot(distance, depth_error, color=color, label=experiment)
        axes[1].plot(distance, coverage, color=color, label=experiment)
    axes[0].set_xlabel("GT depth (m)")
    axes[0].set_ylabel("Mean absolute depth error (m)")
    axes[0].set_title("Observed depth error by distance")
    axes[1].set_xlabel("GT depth (m)")
    axes[1].set_ylabel("Valid coverage")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Valid coverage by distance")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_contact_sheet(
    output_path: Path,
    image_paths: list[Path],
    *,
    columns: int = 3,
    tile_width: int = 480,
) -> None:
    """Combine scene panels into a compact visual-inspection sheet."""

    if not image_paths or columns <= 0 or tile_width <= 0:
        raise ValueError("Expected images and positive contact-sheet dimensions")
    tiles: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as image:
            tile = image.convert("RGB")
            tile_height = round(tile.height * tile_width / tile.width)
            tiles.append(tile.resize((tile_width, tile_height), Image.Resampling.LANCZOS))
    tile_height = max(tile.height for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(tile, (x, y))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def save_stereo_failure_zoom(
    output_path: Path,
    *,
    scene_name: str,
    method_name: str,
    rgb: np.ndarray,
    target_disparity_px: np.ndarray,
    prediction_disparity_px: np.ndarray,
    evaluation_mask: np.ndarray,
    crop_xyxy: tuple[int, int, int, int],
    disparity_max_px: float,
) -> None:
    """Save a full-image crop locator and a four-panel local failure zoom."""

    x0, y0, x1, y1 = crop_xyxy
    height, width = target_disparity_px.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid crop {crop_xyxy} for image {(width, height)}")
    target = np.asarray(target_disparity_px, dtype=np.float32).copy()
    prediction = np.asarray(prediction_disparity_px, dtype=np.float32).copy()
    evaluation = np.asarray(evaluation_mask, dtype=bool)
    target[~evaluation] = np.nan
    prediction[~evaluation] = np.nan
    error = np.abs(prediction - target)

    figure, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    axes[0].imshow(rgb)
    rectangle = plt.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, edgecolor="red", facecolor="none", linewidth=2
    )
    axes[0].add_patch(rectangle)
    axes[0].set_title("Full RGB and crop")
    axes[1].imshow(rgb[y0:y1, x0:x1])
    axes[1].set_title("RGB zoom")
    gt_image = axes[2].imshow(
        target[y0:y1, x0:x1], cmap="turbo", vmin=0, vmax=disparity_max_px
    )
    axes[2].set_title("GT disparity")
    axes[3].imshow(
        prediction[y0:y1, x0:x1], cmap="turbo", vmin=0, vmax=disparity_max_px
    )
    axes[3].set_title(f"{method_name} disparity")
    error_image = axes[4].imshow(error[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=4)
    axes[4].set_title("Absolute error (0–4 px)")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(gt_image, ax=[axes[2], axes[3]], shrink=0.7)
    figure.colorbar(error_image, ax=axes[4], shrink=0.7)
    figure.suptitle(f"{scene_name} — thin-structure failure zoom", fontsize=15)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
