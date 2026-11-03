"""Report-ready Middlebury stereo visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_stereo_panel(
    output_path: Path, *, scene_name: str, method_name: str, rgb: np.ndarray,
    target_disparity_px: np.ndarray, prediction_disparity_px: np.ndarray,
    evaluation_mask: np.ndarray, raw_prediction_valid: np.ndarray,
    lr_consistent: np.ndarray, disparity_max_px: float,
    raw_disparity_px: np.ndarray | None = None,
) -> None:
    """Compact six panels, preserving the full input and standalone LR masks."""
    target = np.asarray(target_disparity_px).copy()
    prediction = np.asarray(prediction_disparity_px)
    evaluation = np.asarray(evaluation_mask, dtype=bool)
    target[~evaluation] = np.nan
    raw_valid = np.asarray(raw_prediction_valid, dtype=bool)
    consistent = np.asarray(lr_consistent, dtype=bool)
    final_valid = np.isfinite(prediction) & (prediction > 0)
    common = evaluation & final_valid
    error = np.full(target.shape, np.nan, dtype=np.float32)
    error[common] = np.abs(prediction[common] - target[common])
    lr_status = np.zeros(target.shape, dtype=np.uint8)
    lr_status[raw_valid & ~consistent] = 1
    lr_status[raw_valid & consistent] = 2
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for name, mask in {'matcher_invalid': ~raw_valid, 'invalid': ~final_valid,
                       'lr_inconsistent': raw_valid & ~consistent}.items():
        Image.fromarray(mask.astype(np.uint8) * 255).save(output_path.parent / f'{name}.png')
    Image.fromarray(lr_status).save(output_path.parent / 'lr_status.png')

    # Physical axes sizes preserve aspect ratio and give every image equal space.
    # Colorbars have their own reserved column rather than shrinking image axes.
    panel_width = 4.6
    panel_height = panel_width * target.shape[0] / target.shape[1]
    left, bottom, gap_x, gap_y, top, right = .08, .08, .48, .34, .56, .68
    width = left + 3 * panel_width + 2 * gap_x + right
    height = bottom + 2 * panel_height + gap_y + top
    figure = plt.figure(figsize=(width, height))
    axes = []
    for row in range(2):
        for column in range(3):
            x = left + column * (panel_width + gap_x)
            y = bottom + (1-row) * (panel_height + gap_y)
            axes.append(figure.add_axes([x/width, y/height,
                                        panel_width/width, panel_height/height]))
    axes[0].imshow(rgb)
    axes[0].set_title('RGB (left)', fontsize=10, pad=4)
    for index, values, title in (
        (1, target, 'GT disparity (px)'),
        (2, prediction, f'{method_name} disparity (px)'),
    ):
        disparity_image = axes[index].imshow(values, cmap='turbo', vmin=0, vmax=disparity_max_px)
        axes[index].set_title(title, fontsize=10, pad=4)
    error_image = axes[3].imshow(error, cmap='magma', vmin=0, vmax=4)
    axes[3].set_title('Absolute error (0–4 px)', fontsize=10, pad=4)
    axes[4].imshow(~raw_valid, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[4].set_title('Matcher invalid (white = invalid)', fontsize=10, pad=4)
    axes[5].imshow(~final_valid, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[5].set_title('Final invalid (white = invalid)', fontsize=10, pad=4)
    for axis in axes:
        axis.axis('off')
    colorbar_x = left + 3 * panel_width + 2 * gap_x + .045
    for image, x, y, label in (
        (disparity_image, colorbar_x, bottom + panel_height + gap_y, 'Disparity (px)'),
        (error_image, left + panel_width + .035, bottom, 'Error (px)'),
    ):
        colorbar_axis = figure.add_axes([x/width, y/height, .09/width, panel_height/height])
        bar = figure.colorbar(image, cax=colorbar_axis)
        bar.ax.tick_params(labelsize=8, length=2, pad=2)
        bar.set_label(label, fontsize=8, labelpad=3)
    figure.suptitle(f'{scene_name} — {method_name}', fontsize=13, y=1-.08/height)
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
        depth_error = [float(row["mean_abs_depth_error_m"]) if row["mean_abs_depth_error_m"] is not None else np.nan for row in selected]
        coverage = [float(row["valid_prediction_fraction"]) for row in selected]
        axes[0].plot(distance, depth_error, color=color, label=experiment)
        axes[1].plot(distance, coverage, color=color, label=experiment)
    axes[0].set_xlabel("GT depth (m)")
    axes[0].set_ylabel("Mean absolute depth error (m)")
    axes[0].set_title("Observed depth error by distance")
    axes[1].set_xlabel("GT depth (m)")
    axes[1].set_ylabel("Valid prediction fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Valid prediction fraction by distance")
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
