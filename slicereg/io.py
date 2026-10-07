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
                blur_px: int, largest_only: bool = False) -> np.ndarray:
    """Binary mask of the main piece(s) of tissue in a single section image.

    ``largest_only`` drops everything but the biggest piece, e.g. bits of neighbouring
    sections at the edge of an image that is known to hold one section."""
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
    if largest_only:
        return lab == int(np.argmax(areas)) + 1
    keep = np.flatnonzero(areas >= 0.1 * areas.max()) + 1
    return np.isin(lab, keep)


def _right_hemisphere_left(al: dict) -> bool:
    """Whether an alignment puts the atlas's right hemisphere on the image's left.

    The atlas plane's first axis runs from the right hemisphere to the left; it points
    to the image's right when the in-plane rotation and mirror don't reverse it."""
    return np.cos(np.radians(al["rotation_deg"])) * (-1.0 if al["flip"] else 1.0) > 0


def _mirror_atlas(al: dict) -> None:
    """Swap the atlas's hemispheres without moving its outline in the image: mirror
    the plane left-right (flip) and the tilt (yaw). Exact for a symmetric atlas."""
    al["flip"] = not al["flip"]
    al["yaw_deg"] = -al["yaw_deg"]
    w = float(al["plane_size_um"][0])
    al["landmarks"] = [[w - s, t, x, y] for s, t, x, y in al["landmarks"]]


class Project:
    """A folder next to the slide holding thumbnails, crops, alignments and cells."""

    def __init__(self, root: Path, data: dict):
        self.root = Path(root)
        self.data = data
        self._slide = None

    @classmethod
    def open_or_create(cls, path: str | Path, project_dir: str | Path | None = None) -> "Project":
        """Open a project folder, or create a project for a slide image or a folder of
        section images (``<name>_project`` next to it unless ``project_dir`` is given)."""
        path = Path(path)
        if path.is_dir() and (path / "project.json").exists():
            return cls.load(path)
        if path.name == "project.json":
            return cls.load(path.parent)
        root = Path(project_dir) if project_dir else path.with_name(path.stem + "_project")
        if (root / "project.json").exists():
            return cls.load(root)
        if path.is_dir():
            return cls._create_folder_project(path, root)
        root.mkdir(parents=True, exist_ok=True)
        data = {
            "source": str(path.resolve()),
            "atlas": DEFAULT_ATLAS,
            "thumb_factor": THUMB_FACTOR,
            "threshold": None,
            "dark_background": True,
            "section_spacing_um": 80.0,
            "slices": [],
        }
        proj = cls(root, data)
        proj.save()
        return proj

    @classmethod
    def _create_folder_project(cls, folder: Path, root: Path) -> "Project":
        from .folder import find_sections, load_config

        config, _ = load_config(folder)
        if not find_sections(folder, config):
            raise FileNotFoundError(
                f"No files in {folder} match the pattern {config['pattern']!r} from "
                f"{folder / 'slicereg_folder.json'}; edit it to match your file names.")
        root.mkdir(parents=True, exist_ok=True)
        data = {
            "source": str(folder.resolve()),
            "source_type": "folder",
            "atlas": config["atlas"],
            "threshold": None,
            "dark_background": True,
            "section_spacing_um": float(config["section_spacing_um"]),
            "slices": [],
        }
        proj = cls(root, data)
        proj.sync_folder()
        return proj

    @property
    def is_folder(self) -> bool:
        return self.data.get("source_type") == "folder"

    def sync_folder(self) -> list[int]:
        """Pick up the folder's config and any new section files.

        Sections keep their id (matched by name) so alignments and cells survive.
        When several files share a slide position, the section keeps the file it
        already uses (new sections take the first); the others are listed in
        ``skipped_files``. Returns the ids whose images still need to be imported."""
        from .folder import _name_key, find_sections, load_config, load_hemispheres

        folder = Path(self.data["source"])
        config, _ = load_config(folder)
        try:
            hemispheres = load_hemispheres(folder, config)
            self.hemisphere_error = None
        except ValueError as e:  # keep the sections' current orientation
            hemispheres, self.hemisphere_error = None, str(e)
        self.data["channels"] = config["channels"]
        self.data["align_channel"] = config["align_channel"]
        self.data["auto_crop"] = bool(config["auto_crop"])
        self.data["order"] = config["order"]
        old = {s["name"]: s for s in self.slices}
        next_id = max(self._used_ids() | {0}) + 1
        candidates: dict[str, list[dict]] = {}
        for found in find_sections(folder, config):
            candidates.setdefault(found["name"], []).append(found)
        self.skipped_files = []  # other files at an already-used slide position
        slices = []
        for name, files in candidates.items():
            current = old.get(name, {}).get("file")
            found = next((f for f in files if f["file"] == current), files[0])
            self.skipped_files += [f["file"] for f in files if f is not found]
            entry = old.get(found["name"])
            if entry is None:
                entry, next_id = {"id": next_id}, next_id + 1
            elif entry.get("file") != found["file"]:  # another image of this section
                for key in ("pixel_um", "crop"):
                    entry.pop(key, None)
            entry.update(found)
            if config["pixel_um"]:
                entry["pixel_um"] = float(config["pixel_um"])
            if hemispheres is not None:
                side = hemispheres.get(_name_key(entry["name"]))
                if side:
                    entry["hemispheres"] = side
                else:
                    entry.pop("hemispheres", None)
                entry["mirror"] = side == "L,R"
            slices.append(entry)
        self.data["slices"] = slices
        self.save()
        return [s["id"] for s in slices if not self.status(s["id"])["cropped"]
                or s.get("auto_crop", False) != self.data["auto_crop"]
                or s.get("imported_file", s["file"]) != s["file"]]

    def import_section(self, sid: int, auto_crop: bool | None = None) -> None:
        """Read one section file of a folder project into the slice folder.

        ``auto_crop`` crops the image to the main piece of tissue and blanks anything
        else (bits of neighbouring sections). The crop box is stored as ``crop`` in
        source-file pixels; existing alignment and cells are moved with it. Sections
        marked ``mirror`` (see ``apply_hemispheres``) are stored mirrored left-right."""
        from .folder import read_section

        s = self.get_slice(sid)
        img, pixel_um = read_section(Path(self.data["source"]) / s["file"])
        if self.status(sid)["cropped"] and s.get("mirrored"):
            self._mirror_slice_data(sid, s["size"][0])  # back to the file's orientation
        s["mirrored"] = False
        if pixel_um and "pixel_um" not in s:
            s["pixel_um"] = pixel_um
        h, w = img.shape[:2]
        crop = [0, 0, w, h]
        if auto_crop is None:
            auto_crop = self.data.get("auto_crop", True)
        if auto_crop:
            img, crop = self._crop_to_tissue(img, s.get("pixel_um"))
        old = s.get("crop", [0, 0, *s.get("size", (w, h))])
        if old[:2] != crop[:2] and self.status(sid)["cropped"]:
            self._shift_slice_data(sid, old[0] - crop[0], old[1] - crop[1], crop)
        s["crop"] = [int(v) for v in crop]
        s["auto_crop"] = bool(auto_crop)
        mirror = bool(s.get("mirror"))
        if mirror:
            img = np.ascontiguousarray(img[:, ::-1])
        self._write_section(sid, img, blur_um=80.0, largest_only=True)
        if mirror:
            self._mirror_slice_data(sid, img.shape[1])
        s["mirrored"] = mirror
        s["imported_file"] = s["file"]

    def apply_hemispheres(self, atlas=None) -> list[int]:
        """Give every section the atlas's orientation: right hemisphere on the image's
        left. Returns the ids that changed.

        Sections marked ``mirror`` (from the folder's hemispheres CSV) have their image,
        preview and mask mirrored left-right, with alignment, landmarks and cells
        mirrored along so the atlas outline stays on the same tissue. Then, where the
        hemispheres are known, an atlas fitted with its hemispheres the wrong way round
        is mirrored (flip toggled, yaw negated): its outline doesn't move, but its left
        and right swap. Cells are re-mapped so their hemisphere follows."""
        changed = []
        for s in self.slices:
            sid = s["id"]
            if not self.status(sid)["cropped"]:
                continue
            moved = False
            want = bool(s.get("mirror", False))
            if bool(s.get("mirrored", False)) != want:
                self._mirror_files(sid)
                self._mirror_slice_data(sid, s["size"][0])
                s["mirrored"] = want
                moved = True
            al = self.load_alignment(sid)
            if s.get("hemispheres") and al is not None and not _right_hemisphere_left(al):
                _mirror_atlas(al)
                self.save_alignment(sid, al)
                moved = True
            if moved:
                changed.append(sid)
                cells = self.load_cells(sid)
                if al is not None and cells is not None and len(cells):
                    from .atlas import Atlas
                    from .export import map_cells
                    from .transform import Alignment

                    atlas = atlas or Atlas(self.data["atlas"])
                    al = self.load_alignment(sid)
                    xy = cells[["x_px", "y_px"]].to_numpy(float)
                    self.save_cells(sid, map_cells(atlas, Alignment.from_dict(al), xy,
                                                   cells["cell_type"].tolist(), sid))
        self.save()
        return changed

    def _mirror_files(self, sid: int) -> None:
        d = self.slice_dir(sid)
        img = np.ascontiguousarray(self.load_image(sid)[:, ::-1])
        tifffile.imwrite(d / "image.tif", img, compression="zlib",
                         photometric="rgb" if img.shape[2] == 3 else "minisblack")
        tifffile.imwrite(d / "preview.tif",
                         np.ascontiguousarray(self.load_preview(sid)[:, ::-1]))
        mask = self.load_mask(sid)[:, ::-1]
        cv2.imwrite(str(d / "mask.png"), mask.astype(np.uint8) * 255)

    def _mirror_slice_data(self, sid: int, width: int) -> None:
        """Mirror a section's alignment, landmarks and cells left-right with its image
        (x -> width - 1 - x), keeping the atlas outline on the same tissue."""
        al = self.load_alignment(sid)
        if al is not None:
            al["rotation_deg"] = -al["rotation_deg"]
            al["flip"] = not al["flip"]
            al["tx"] = width - 1 - al["tx"]
            al["landmarks"] = [[s, t, width - 1 - x, y] for s, t, x, y in al["landmarks"]]
            self.save_alignment(sid, al)
        cells = self.load_cells(sid)
        if cells is not None:
            cells["x_px"] = width - 1 - cells["x_px"]
            self.save_cells(sid, cells)

    def _crop_to_tissue(self, img: np.ndarray, pixel_um: float | None,
                        margin_um: float = 150.0) -> tuple[np.ndarray, list[int]]:
        """Crop to the largest piece of tissue (plus a margin) and zero everything
        outside that piece's mask, dilated by the margin so its edge isn't clipped."""
        h, w = img.shape[:2]
        ds = max(1, int(round(w / PREVIEW_WIDTH)))
        small = downsample(img, ds)
        thr = self.data["threshold"]
        if thr is None:
            thr = float(cv2.threshold(small.max(axis=2).astype(np.uint8), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
        px_um = pixel_um or 1.0
        mask = tissue_mask(small, thr, self.data["dark_background"],
                           int(80.0 / px_um / ds), largest_only=True)
        if not mask.any():
            return img, [0, 0, w, h]
        k = max(1, int(round(margin_um / px_um / ds)))
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        ys, xs = np.nonzero(mask)
        fx, fy = w / small.shape[1], h / small.shape[0]
        x0, x1 = max(0, int(xs.min() * fx)), min(w, int(np.ceil((xs.max() + 1) * fx)))
        y0, y1 = max(0, int(ys.min() * fy)), min(h, int(np.ceil((ys.max() + 1) * fy)))
        full_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1]
        out = img[y0:y1, x0:x1] * full_mask[..., None].astype(img.dtype)
        return np.ascontiguousarray(out), [x0, y0, x1, y1]

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
        x0, y0, x1, y1 = self.get_slice(sid)["bbox"]
        crop = np.ascontiguousarray(self.slide[y0:y1, x0:x1])
        self._write_section(sid, crop, blur_px_full=5 * self.data["thumb_factor"])

    def _write_section(self, sid: int, img: np.ndarray, blur_px_full: float | None = None,
                       blur_um: float | None = None, largest_only: bool = False) -> None:
        """Save image.tif, preview.tif and mask.png for one section."""
        s = self.get_slice(sid)
        d = self.slice_dir(sid)
        d.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(d / "image.tif", img, compression="zlib",
                         photometric="rgb" if img.shape[2] == 3 else "minisblack")
        ds = max(1, int(round(img.shape[1] / PREVIEW_WIDTH)))
        s["ds"] = ds
        s["size"] = [int(img.shape[1]), int(img.shape[0])]
        preview = downsample(img, ds)
        tifffile.imwrite(d / "preview.tif", preview)
        thr = self.data["threshold"]
        if thr is None:
            thr = float(cv2.threshold(preview.max(axis=2).astype(np.uint8), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
        if blur_um is not None and s.get("pixel_um"):
            blur_px_full = blur_um / s["pixel_um"]
        blur = (blur_px_full or 80.0) / ds
        mask = tissue_mask(preview, thr, self.data["dark_background"], int(blur), largest_only)
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
