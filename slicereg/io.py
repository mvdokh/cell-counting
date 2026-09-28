"""Slide reading, thumbnails, and the on-disk project layout."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

DEFAULT_ATLAS = "allen_mouse_25um"
THUMB_FACTOR = 16
PREVIEW_WIDTH = 1000


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[-1] > 4:
        return np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported slide shape {arr.shape}")
    return arr


def open_slide(path: str | Path) -> np.ndarray:
    """Return the slide as an (H, W, C) array, memory-mapped when possible."""
    try:
        arr = tifffile.memmap(str(path), mode="r")
    except (ValueError, OSError):
        arr = tifffile.imread(str(path))
    return _to_hwc(arr)


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return np.asarray(img)
    h, w = img.shape[:2]
    out = cv2.resize(np.asarray(img), (max(1, w // factor), max(1, h // factor)),
                     interpolation=cv2.INTER_AREA)
    return out.reshape(out.shape[0], out.shape[1], -1)


def make_thumbnail(slide: np.ndarray, factor: int = THUMB_FACTOR) -> np.ndarray:
    """Area-averaged thumbnail, read in row bands so the slide is never fully loaded."""
    h, w, c = slide.shape
    band = factor * 64
    rows = []
    for r0 in range(0, (h // factor) * factor, band):
        r1 = min(r0 + band, (h // factor) * factor)
        block = np.asarray(slide[r0:r1, : (w // factor) * factor])
        rows.append(downsample(block, factor))
    return np.concatenate(rows, axis=0)


def to_display(img: np.ndarray, lo: float = 0.5, hi: float = 99.8) -> np.ndarray:
    """Per-channel percentile stretch to uint8 RGB for PNG snapshots."""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    out = np.empty_like(img)
    for ch in range(img.shape[2]):
        a, b = np.percentile(img[..., ch], [lo, hi])
        out[..., ch] = np.clip((img[..., ch] - a) / max(b - a, 1e-6), 0, 1)
    out = (out * 255).astype(np.uint8)
    if out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    elif out.shape[2] == 2:
        out = np.concatenate([out, np.zeros_like(out[..., :1])], axis=2)
    return out[..., :3]


def imwrite_png(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def tissue_mask(img: np.ndarray, threshold: float, dark_background: bool,
                blur_px: int) -> np.ndarray:
    """Binary mask of the main piece(s) of tissue in a single section image."""
    g = np.asarray(img).max(axis=2).astype(np.float32)
    k = max(3, int(blur_px) | 1)
    g = cv2.GaussianBlur(g, (k, k), 0)
    m = g > threshold if dark_background else g < threshold
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    areas = ndi.sum(m, lab, index=np.arange(1, n + 1))
    keep = np.flatnonzero(areas >= 0.1 * areas.max()) + 1
    return np.isin(lab, keep)


class Project:
    """A folder next to the slide holding thumbnails, crops, alignments and cells."""

    def __init__(self, root: Path, data: dict):
        self.root = Path(root)
        self.data = data
        self._slide = None

    @classmethod
    def open_or_create(cls, path: str | Path, project_dir: str | Path | None = None) -> "Project":
        path = Path(path)
        if path.is_dir():
            return cls.load(path)
        if path.name == "project.json":
            return cls.load(path.parent)
        root = Path(project_dir) if project_dir else path.with_name(path.stem + "_project")
        if (root / "project.json").exists():
            return cls.load(root)
        root.mkdir(parents=True, exist_ok=True)
        data = {
            "source": str(path.resolve()),
            "atlas": DEFAULT_ATLAS,
            "thumb_factor": THUMB_FACTOR,
            "threshold": None,
            "dark_background": True,
            "section_spacing_um": 100.0,
            "slices": [],
        }
        proj = cls(root, data)
        proj.save()
        return proj

    @classmethod
    def load(cls, root: str | Path) -> "Project":
        root = Path(root)
        with open(root / "project.json") as f:
            return cls(root, json.load(f))

    def save(self) -> None:
        tmp = self.root / "project.json.tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        tmp.replace(self.root / "project.json")

    @property
    def slide(self) -> np.ndarray:
        if self._slide is None:
            self._slide = open_slide(self.data["source"])
        return self._slide

    @property
    def slices(self) -> list[dict]:
        return self.data["slices"]

    def get_slice(self, sid: int) -> dict:
        for s in self.slices:
            if s["id"] == sid:
                return s
        raise KeyError(sid)

    def thumbnail(self) -> np.ndarray:
        p = self.root / "thumbnail.tif"
        if p.exists():
            return _to_hwc(tifffile.imread(p))
        thumb = make_thumbnail(self.slide, self.data["thumb_factor"])
        tifffile.imwrite(p, thumb)
        imwrite_png(self.root / "thumbnail.png", to_display(thumb))
        return thumb

    def slice_dir(self, sid: int) -> Path:
        return self.root / "slices" / f"slice_{sid:02d}"

    def status(self, sid: int) -> dict:
        d = self.slice_dir(sid)
        return {
            "cropped": (d / "image.tif").exists(),
            "aligned": (d / "alignment.json").exists(),
            "counted": (d / "cells.csv").exists(),
        }

    def _used_ids(self) -> set[int]:
        ids = {s["id"] for s in self.slices}
        d = self.root / "slices"
        if d.exists():
            for p in d.glob("slice_*"):
                try:
                    ids.add(int(p.name.split("_")[1]))
                except ValueError:
                    pass
        return ids

    def set_boxes(self, entries: list[tuple[int | None, list[int]]]) -> list[int]:
        """Replace the section list with ``(sid or None, [x0, y0, x1, y1])`` entries.

        New boxes get fresh ids (never reusing an old folder). Moved boxes keep their
        alignment and cells, shifted into the new crop frame. Returns ids needing a crop.
        """
        old = {s["id"]: s for s in self.slices}
        next_id = max(self._used_ids() | {sid for sid, _ in entries if sid is not None} | {0}) + 1
        new_slices, to_crop = [], []
        for sid, bbox in entries:
            bbox = [int(v) for v in bbox]
            if sid is None or sid < 0:
                sid, next_id = next_id, next_id + 1
            prev = old.get(sid)
            entry = dict(prev) if prev else {"id": sid}
            entry["bbox"] = bbox
            if prev is None or prev["bbox"] != bbox or not self.status(sid)["cropped"]:
                if prev is not None and prev["bbox"] != bbox:
                    self._shift_slice_data(sid, prev["bbox"][0] - bbox[0],
                                           prev["bbox"][1] - bbox[1], bbox)
                to_crop.append(sid)
            new_slices.append(entry)
        self.data["slices"] = new_slices
        self.save()
        return to_crop

    def _shift_slice_data(self, sid: int, dx: int, dy: int, bbox: list[int]) -> None:
        al = self.load_alignment(sid)
        if al is not None:
            al["tx"] += dx
            al["ty"] += dy
            al["landmarks"] = [[s, t, x + dx, y + dy] for s, t, x, y in al["landmarks"]]
            al["image_size"] = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
            self.save_alignment(sid, al)
        cells = self.load_cells(sid)
        if cells is not None:
            cells["x_px"] += dx
            cells["y_px"] += dy
            self.save_cells(sid, cells)

    def crop_slice(self, sid: int) -> None:
        s = self.get_slice(sid)
        x0, y0, x1, y1 = s["bbox"]
        d = self.slice_dir(sid)
        d.mkdir(parents=True, exist_ok=True)
        crop = np.ascontiguousarray(self.slide[y0:y1, x0:x1])
        tifffile.imwrite(d / "image.tif", crop, compression="zlib",
                         photometric="rgb" if crop.shape[2] == 3 else "minisblack")
        ds = max(1, int(round((x1 - x0) / PREVIEW_WIDTH)))
        s["ds"] = ds
        preview = downsample(crop, ds)
        tifffile.imwrite(d / "preview.tif", preview)
        thr = self.data["threshold"]
        if thr is None:
            thr = float(cv2.threshold(preview.max(axis=2).astype(np.uint8), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
        blur = 5 * self.data["thumb_factor"] / ds
        mask = tissue_mask(preview, thr, self.data["dark_background"], int(blur))
        cv2.imwrite(str(d / "mask.png"), mask.astype(np.uint8) * 255)

    def load_image(self, sid: int) -> np.ndarray:
        return _to_hwc(tifffile.imread(self.slice_dir(sid) / "image.tif"))

    def load_preview(self, sid: int) -> np.ndarray:
        return _to_hwc(tifffile.imread(self.slice_dir(sid) / "preview.tif"))

    def load_mask(self, sid: int) -> np.ndarray:
        return cv2.imread(str(self.slice_dir(sid) / "mask.png"), cv2.IMREAD_GRAYSCALE) > 0

    def load_alignment(self, sid: int) -> dict | None:
        p = self.slice_dir(sid) / "alignment.json"
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def save_alignment(self, sid: int, alignment: dict) -> None:
        with open(self.slice_dir(sid) / "alignment.json", "w") as f:
            json.dump(alignment, f, indent=2)

    def load_cells(self, sid: int) -> pd.DataFrame | None:
        p = self.slice_dir(sid) / "cells.csv"
        return pd.read_csv(p) if p.exists() else None

    def save_cells(self, sid: int, df: pd.DataFrame) -> None:
        df.to_csv(self.slice_dir(sid) / "cells.csv", index=False)
