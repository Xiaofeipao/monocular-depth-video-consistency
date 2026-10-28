"""Report-ready NYUv2 metric-depth visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_nyuv2_panel(
    output_path: Path,
    *,
    rgb: np.ndarray,
    target_depth_m: np.ndarray,
    prediction_depth_m: np.ndarray,
    crop: tuple[slice, slice],
    title: str,
    prediction_title: str = "Depth Pro native metric (m)",
    depth_range_m: tuple[float, float] = (0.5, 10.0),
    error_max_m: float = 2.0,
) -> None:
    """Save RGB, common-crop depth, and error panels with fixed metric scales."""

    target = np.asarray(target_depth_m)[crop]
    prediction = np.asarray(prediction_depth_m)[crop]
    absolute_error = np.abs(prediction - target)
    relative_error = absolute_error / target
    figure, axes = plt.subplots(1, 5, figsize=(20, 4.3), constrained_layout=True)
    axes[0].imshow(rgb[crop])
    axes[0].set_title("RGB (evaluation crop)")
    target_image = axes[1].imshow(
        target,
        cmap="turbo_r",
        vmin=depth_range_m[0],
        vmax=depth_range_m[1],
    )
    axes[1].set_title("GT metric depth (m)")
    axes[2].imshow(
        prediction,
        cmap="turbo_r",
        vmin=depth_range_m[0],
        vmax=depth_range_m[1],
    )
    axes[2].set_title(prediction_title)
    error_image = axes[3].imshow(
        absolute_error, cmap="magma", vmin=0.0, vmax=error_max_m
    )
    axes[3].set_title(f"Absolute error (0–{error_max_m:g} m)")
    relative_image = axes[4].imshow(
        relative_error, cmap="magma", vmin=0.0, vmax=1.0
    )
    axes[4].set_title("Absolute relative error (0–1)")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(target_image, ax=[axes[1], axes[2]], shrink=0.72)
    figure.colorbar(error_image, ax=axes[3], shrink=0.72)
    figure.colorbar(relative_image, ax=axes[4], shrink=0.72)
    figure.suptitle(title, fontsize=14)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_contact_sheet(
    output_path: Path,
    image_paths: list[Path],
    *,
    columns: int = 2,
    tile_width: int = 900,
) -> None:
    """Combine selected full panels without changing their internal color scales."""

    if not image_paths:
        raise ValueError("At least one visualization is required")
    tiles: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as image:
            tile = image.convert("RGB")
            height = round(tile.height * tile_width / tile.width)
            tiles.append(tile.resize((tile_width, height), Image.Resampling.LANCZOS))
    tile_height = max(tile.height for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, tile in enumerate(tiles):
        canvas.paste(
            tile,
            ((index % columns) * tile_width, (index // columns) * tile_height),
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
