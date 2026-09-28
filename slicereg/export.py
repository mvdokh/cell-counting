"""Map clicked cells into atlas space and combine results across slices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .transform import Alignment, SliceTransform

CELL_COLORS = ["#ff5050", "#50ff50", "#50b0ff", "#ffff50", "#ff50ff", "#50ffff"]

CELL_COLUMNS = ["slice_id", "cell_type", "x_px", "y_px", "ap_um", "dv_um", "ml_um",
                "region_id", "region_acronym", "region_name", "hemisphere"]


def map_cells(atlas, al: Alignment, xy: np.ndarray, cell_types: list[str],
              sid: int) -> pd.DataFrame:
    """Full-resolution crop pixels (x, y) -> atlas microns and region labels."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) == 0:
        return pd.DataFrame(columns=CELL_COLUMNS)
    tf = SliceTransform(al)
    pts = atlas.plane_to_3d(tf.image_to_plane(xy), al.ap_um, al.pitch_deg, al.yaw_deg)
    region, hemi = atlas.lookup(pts)
    return pd.DataFrame({
        "slice_id": sid,
        "cell_type": cell_types,
        "x_px": xy[:, 0].round(2),
        "y_px": xy[:, 1].round(2),
        "ap_um": pts[:, 0].round(2),
        "dv_um": pts[:, 1].round(2),
        "ml_um": pts[:, 2].round(2),
        "region_id": region,
        "region_acronym": [atlas.acronym(r) for r in region],
        "region_name": [atlas.region_name(r) for r in region],
        "hemisphere": [atlas.hemisphere_name(h) for h in hemi],
    })[CELL_COLUMNS]


def collect_cells(project) -> pd.DataFrame:
    frames = [df for s in project.slices
              if (df := project.load_cells(s["id"])) is not None and len(df)]
    if not frames:
        return pd.DataFrame(columns=CELL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def export_project(project) -> str:
    cells = collect_cells(project)
    cells.to_csv(project.root / "cells_all.csv", index=False)
    keys = ["slice_id", "cell_type", "region_id", "region_acronym", "region_name", "hemisphere"]
    per_slice = cells.groupby(keys).size().rename("count").reset_index()
    total = (cells.groupby(keys[1:]).size().rename("count").reset_index()
             .assign(slice_id="all"))[keys + ["count"]]
    counts = pd.concat([per_slice, total], ignore_index=True)
    counts.to_csv(project.root / "region_counts.csv", index=False)
    counted = sum(project.status(s["id"])["counted"] for s in project.slices)
    return (f"{len(cells)} cells from {counted}/{len(project.slices)} slices.\n"
            f"Wrote {project.root / 'cells_all.csv'}\n"
            f"and {project.root / 'region_counts.csv'}")
