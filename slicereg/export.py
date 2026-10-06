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


NOT_REGIONS = ("outside", "root")


def top_regions(cells: pd.DataFrame) -> pd.DataFrame:
    """Regions ranked by cell count (all slices), with each hemisphere's and cell
    type's share. Cells outside the atlas or in no specific region are left out of the
    ranking but still count towards ``percent_of_cells``."""
    columns = ["rank", "region_acronym", "region_name", "cells", "percent_of_cells",
               "left", "right"]
    inside = cells[~cells["region_acronym"].isin(NOT_REGIONS)]
    if inside.empty:
        return pd.DataFrame(columns=columns)
    keys = ["region_acronym", "region_name"]
    table = inside.groupby(keys).size().rename("cells").to_frame()
    hemi = inside.groupby(keys + ["hemisphere"]).size().unstack(fill_value=0)
    for side in ("left", "right"):
        table[side] = hemi[side] if side in hemi else 0
    types = inside["cell_type"].astype(str)
    if types.nunique() > 1:
        per_type = inside.groupby(keys + [types]).size().unstack(fill_value=0)
        table = table.join(per_type.add_prefix("cells_"))
    table = table.reset_index().sort_values(["cells", "region_acronym"],
                                            ascending=[False, True], ignore_index=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.insert(4, "percent_of_cells", (100 * table["cells"] / len(cells)).round(2))
    return table


def save_region_bar_plot(top: pd.DataFrame, path) -> None:
    """Bar plot of cells per region, in ``top_regions`` order."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(max(4.0, 0.45 * len(top) + 1.5), 4.0), layout="constrained")
    ax = fig.subplots()
    ax.bar(top["region_acronym"].astype(str), top["cells"], color="0.35")
    ax.set_ylabel("Cells")
    ax.set_xlabel("Region")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.01)
    fig.savefig(path, dpi=300)


def export_project(project) -> str:
    cells = collect_cells(project)
    cells.to_csv(project.root / "cells_all.csv", index=False)
    keys = ["slice_id", "cell_type", "region_id", "region_acronym", "region_name", "hemisphere"]
    per_slice = cells.groupby(keys).size().rename("count").reset_index()
    total = (cells.groupby(keys[1:]).size().rename("count").reset_index()
             .assign(slice_id="all"))[keys + ["count"]]
    counts = pd.concat([per_slice, total], ignore_index=True)
    counts.to_csv(project.root / "region_counts.csv", index=False)
    top = top_regions(cells)
    top.to_csv(project.root / "top_regions.csv", index=False)
    written = ["cells_all.csv", "region_counts.csv", "top_regions.csv"]
    if len(top):
        save_region_bar_plot(top, project.root / "top_regions.png")
        written.append("top_regions.png")
    counted = sum(project.status(s["id"])["counted"] for s in project.slices)
    return (f"{len(cells)} cells from {counted}/{len(project.slices)} slices.\n"
            f"Wrote {', '.join(written)} in\n{project.root}")
