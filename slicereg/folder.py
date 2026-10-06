"""Folders of per-section images (e.g. Zeiss .lsm files named ``c4_r1_..._MIP.lsm``).

A ``slicereg_folder.json`` next to the images says which files to use, how their
names encode the position on the slide, and what the channels are. It is written
with defaults the first time a folder is opened and can be edited afterwards.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import tifffile

CONFIG_NAME = "slicereg_folder.json"
DEFAULT_CONFIG = {
    "pattern": r"c(?P<col>\d+)_r(?P<row>\d+).*_MIP\.lsm$",
    "order": "column",
    "channels": [{"name": "rfp", "color": "red"}, {"name": "nissl", "color": "blue"}],
    "align_channel": "nissl",
    "auto_crop": True,
    "pixel_um": None,
    "atlas": "allen_mouse_25um",
    "section_spacing_um": 80.0,
    "hemispheres": "hemispheres.csv",
}
CONFIG_HELP = {
    "pattern": "regular expression matched against file names; named groups 'col' and "
               "'row' give the section's position on the slide",
    "order": "'column' = sections are numbered down each column (c1_r1, c1_r2, ...), "
             "'row' = along each row (c1_r1, c2_r1, ...); used to guess AP positions",
    "channels": "one entry per image channel, in file order; color is a napari colormap",
    "align_channel": "channel shown to DeepSlice (null = all channels)",
    "auto_crop": "crop each image to its main section and blank bits of neighbouring "
                 "sections (true/false)",
    "pixel_um": "pixel size in microns; null = read it from each file",
    "section_spacing_um": "AP distance between consecutive sections in cutting order "
                          "(section thickness); only used when the project is created",
    "hemispheres": "optional CSV in this folder with columns slice,x1,x2: which "
                   "hemisphere (L/R) is on the image's left (x1) and right (x2). Sections "
                   "with the left hemisphere on the image's left are mirrored so every "
                   "section has the right hemisphere on the left, like the atlas",
}


def load_config(folder: Path) -> tuple[dict, bool]:
    """Return (config, created). Writes a default config if the folder has none."""
    p = Path(folder) / CONFIG_NAME
    if p.exists():
        with open(p) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}, False
    with open(p, "w") as f:
        json.dump({**DEFAULT_CONFIG, "_help": CONFIG_HELP}, f, indent=2)
    return dict(DEFAULT_CONFIG), True


def find_sections(folder: Path, config: dict) -> list[dict]:
    """Matching files as ``{"file", "name", "col", "row"}``, in section order."""
    rx = re.compile(config["pattern"], re.IGNORECASE)
    found = []
    for p in sorted(Path(folder).iterdir()):
        m = rx.search(p.name)
        if not p.is_file() or not m:
            continue
        col, row = int(m.group("col")), int(m.group("row"))
        found.append({"file": p.name, "name": f"c{col}_r{row}", "col": col, "row": row})
    key = (lambda s: (s["row"], s["col"])) if config["order"] == "row" else \
        (lambda s: (s["col"], s["row"]))
    return sorted(found, key=key)


def _name_key(name: str) -> str:
    """'c4_r1', 'C4R1', 'c4-r1' -> 'c4r1'."""
    return re.sub(r"[^0-9a-z]", "", str(name).lower())


def load_hemispheres(folder: Path, config: dict) -> dict[str, str]:
    """{section name key: 'L,R' or 'R,L'} (image left, image right) from the config's
    hemispheres CSV; empty if there is none."""
    name = config.get("hemispheres")
    p = Path(folder) / name if name else None
    if p is None or not p.is_file():
        return {}
    import pandas as pd

    df = pd.read_csv(p, dtype=str)
    cols = {c.strip().lower(): c for c in df.columns}
    missing = [c for c in ("slice", "x1", "x2") if c not in cols]
    if missing:
        raise ValueError(f"{p} needs columns slice, x1, x2 (missing {', '.join(missing)})")
    out = {}
    for _, row in df.iterrows():
        x1 = str(row[cols["x1"]]).strip().upper()[:1]
        x2 = str(row[cols["x2"]]).strip().upper()[:1]
        if {x1, x2} != {"L", "R"}:
            raise ValueError(f"{p}: row {row[cols['slice']]!r} should have L and R in x1/x2, "
                             f"got {x1!r}, {x2!r}")
        out[_name_key(row[cols["slice"]])] = f"{x1},{x2}"
    return out


def grid_layout(sizes: dict[int, tuple[int, int, float, float]],
                gap: float) -> dict[int, tuple[float, float]]:
    """Top-left corners placing each section in its (col, row) cell of a grid.

    ``sizes`` maps id -> (col, row, width, height); each column is as wide as its widest
    section and each row as tall as its tallest, sections centred in their cell."""
    cols = sorted({c for c, _, _, _ in sizes.values()})
    rows = sorted({r for _, r, _, _ in sizes.values()})
    col_w = {c: max(w for cc, _, w, _ in sizes.values() if cc == c) for c in cols}
    row_h = {r: max(h for _, rr, _, h in sizes.values() if rr == r) for r in rows}
    col_x = dict(zip(cols, np.cumsum([0.0] + [col_w[c] + gap for c in cols[:-1]])))
    row_y = dict(zip(rows, np.cumsum([0.0] + [row_h[r] + gap for r in rows[:-1]])))
    return {sid: (float(col_x[c] + (col_w[c] - w) / 2), float(row_y[r] + (row_h[r] - h) / 2))
            for sid, (c, r, w, h) in sizes.items()}


def _stitch(tiles: np.ndarray, positions_px: np.ndarray) -> np.ndarray:
    """(M, C, h, w) tiles at (x, y) pixel offsets -> (C, H, W), max-blended overlaps."""
    th, tw = tiles.shape[-2:]
    off = np.round(positions_px - positions_px.min(axis=0)).astype(int)
    out = np.zeros((tiles.shape[1], off[:, 1].max() + th, off[:, 0].max() + tw), tiles.dtype)
    for tile, (x, y) in zip(tiles, off):
        view = out[:, y:y + th, x:x + tw]
        np.maximum(view, tile, out=view)
    return out


def read_section(path: Path) -> tuple[np.ndarray, float | None]:
    """Return ``(image (H, W, C), pixel size in um or None)``.

    Z-stacks are max-projected and unstitched mosaics (axis ``M``) are stitched from the
    LSM tile positions, so a missing or failed MIP export still gives a usable image.
    """
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        arr = series.asarray()
        axes = series.axes
        md = tif.lsm_metadata if tif.is_lsm else None
    pixel_um = None
    if md and md.get("VoxelSizeX"):
        pixel_um = float(md["VoxelSizeX"]) * 1e6
    for ax in "TZ":
        if ax in axes:
            arr = arr.max(axis=axes.index(ax))
            axes = axes.replace(ax, "")
    if "C" not in axes:
        arr = arr[..., None, :, :] if "M" in axes else arr[None]
        axes = axes.replace("YX", "CYX")
    if "M" in axes:
        if not md or md.get("TilePositions") is None:
            raise ValueError(f"{path.name}: unstitched tiles without tile positions")
        pos = np.asarray(md["TilePositions"], dtype=float)[:, :2]
        vox = np.array([md["VoxelSizeX"], md["VoxelSizeY"]], dtype=float)
        arr = _stitch(arr, pos / vox)
        axes = "CYX"
    if axes != "CYX":
        raise ValueError(f"{path.name}: unsupported image axes {series.axes}")
    return np.ascontiguousarray(np.moveaxis(arr, 0, -1)), pixel_um
