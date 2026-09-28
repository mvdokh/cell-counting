"""Oblique coronal planes through a BrainGlobe atlas.

Atlas space is BrainGlobe ASR in microns: axis 0 = AP (anterior -> posterior),
axis 1 = DV (dorsal -> ventral), axis 2 = ML (right -> left).

A plane is given by its AP position (at the atlas centre) and two tilt angles.
Plane coordinates ``(s, t)`` are microns along the plane's ML-like and DV-like
axes; with zero tilt, ``(s, t)`` equals ``(ml, dv)`` exactly.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy import ndimage as ndi
from skimage import measure


def rotation(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Pitch tilts the DV axis toward AP; yaw tilts the ML axis toward AP."""
    p, y = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
    r_pitch = np.array([[np.cos(p), np.sin(p), 0], [-np.sin(p), np.cos(p), 0], [0, 0, 1]])
    r_yaw = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return r_yaw @ r_pitch


class Atlas:
    def __init__(self, name: str):
        self.name = name
        self.bg = BrainGlobeAtlas(name, check_latest=False)
        self.res = np.asarray(self.bg.resolution, dtype=float)
        self.shape = np.asarray(self.bg.shape)
        self.extent_um = self.shape * self.res
        self.annotation = np.asarray(self.bg.annotation)
        self.template = np.asarray(self.bg.template).astype(np.float32)
        self.hemispheres = np.asarray(self.bg.hemispheres)
        self.step = float(self.res[2])
        self.plane_size_um = (float(self.extent_um[2]), float(self.extent_um[1]))
        self.structures = {int(s["id"]): s for s in self.bg.structures_list}
        self.structures[0] = {"id": 0, "acronym": "outside", "name": "Outside atlas",
                              "rgb_triplet": [0, 0, 0]}

    @property
    def plane_shape(self) -> tuple[int, int]:
        """(rows, cols) of a sampled plane image, one pixel per ``step`` microns."""
        return int(self.shape[1]), int(self.shape[2])

    def basis(self, ap_um: float, pitch_deg: float = 0.0, yaw_deg: float = 0.0):
        r = rotation(pitch_deg, yaw_deg)
        u, v = r @ np.array([0.0, 0.0, 1.0]), r @ np.array([0.0, 1.0, 0.0])
        w, h = self.plane_size_um
        center = np.array([ap_um, h / 2, w / 2])
        origin = center - (w / 2) * u - (h / 2) * v
        return origin, u, v

    def plane_to_3d(self, st: np.ndarray, ap_um: float, pitch_deg: float = 0.0,
                    yaw_deg: float = 0.0) -> np.ndarray:
        """(N, 2) plane microns -> (N, 3) atlas microns (ap, dv, ml)."""
        origin, u, v = self.basis(ap_um, pitch_deg, yaw_deg)
        st = np.atleast_2d(st)
        return origin + st[:, :1] * u + st[:, 1:2] * v

    def plane_grid_3d(self, ap_um: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        rows, cols = self.plane_shape
        s = (np.arange(cols) + 0.5) * self.step
        t = (np.arange(rows) + 0.5) * self.step
        ss, tt = np.meshgrid(s, t)
        pts = self.plane_to_3d(np.stack([ss.ravel(), tt.ravel()], 1), ap_um, pitch_deg, yaw_deg)
        return pts.reshape(rows, cols, 3)

    def sample_plane(self, ap_um: float, pitch_deg: float = 0.0, yaw_deg: float = 0.0):
        """Return (annotation, template) images of the plane; pixel (j, i) is at
        plane coordinates ((i + 0.5) * step, (j + 0.5) * step)."""
        return _sample_cached(self, round(ap_um, 2), round(pitch_deg, 3), round(yaw_deg, 3))

    def plane_contours(self, ap_um: float, pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                       sigma: float = 1.0) -> tuple[np.ndarray, ...]:
        """Sub-pixel region-boundary polylines for this plane, in plane microns.

        Each polyline is an (N, 2) array of (s, t) points. Cached like ``sample_plane``,
        so dragging sliders that only change colour/width does not retrigger this."""
        return _contours_cached(self, round(ap_um, 2), round(pitch_deg, 3), round(yaw_deg, 3),
                                round(float(sigma), 3))

    def lookup(self, pts_um: np.ndarray):
        """Region id and hemisphere value (0 = outside) for (N, 3) atlas microns."""
        idx = np.floor(np.atleast_2d(pts_um) / self.res).astype(int)
        inside = np.all((idx >= 0) & (idx < self.shape), axis=1)
        region = np.zeros(len(idx), dtype=np.int64)
        hemi = np.zeros(len(idx), dtype=np.int64)
        ii = idx[inside]
        region[inside] = self.annotation[ii[:, 0], ii[:, 1], ii[:, 2]]
        hemi[inside] = self.hemispheres[ii[:, 0], ii[:, 1], ii[:, 2]]
        hemi[region == 0] = 0
        return region, hemi

    def hemisphere_name(self, value: int) -> str:
        if value == self.bg.left_hemisphere_value:
            return "left"
        if value == self.bg.right_hemisphere_value:
            return "right"
        return "outside"

    def acronym(self, rid: int) -> str:
        return self.structures.get(int(rid), self.structures[0])["acronym"]

    def region_name(self, rid: int) -> str:
        return self.structures.get(int(rid), self.structures[0])["name"]

    def color_dict(self) -> dict:
        d = {None: np.zeros(4), 0: np.zeros(4)}
        for rid, s in self.structures.items():
            if rid:
                d[rid] = np.r_[np.asarray(s["rgb_triplet"]) / 255.0, 1.0]
        return d

    def __hash__(self):
        return id(self)


@lru_cache(maxsize=4)
def atlas_structures(name: str) -> list[dict]:
    """Structure list (id, acronym, name, ...) without loading the atlas volumes."""
    return list(BrainGlobeAtlas(name, check_latest=False).structures_list)


@lru_cache(maxsize=8)
def _sample_cached(atlas: Atlas, ap_um: float, pitch_deg: float, yaw_deg: float):
    pts = atlas.plane_grid_3d(ap_um, pitch_deg, yaw_deg)
    rows, cols = atlas.plane_shape
    coords = (pts / atlas.res).reshape(-1, 3).T
    ann = ndi.map_coordinates(atlas.annotation, np.floor(coords), order=0, mode="constant",
                              cval=0).reshape(rows, cols)
    tmpl = ndi.map_coordinates(atlas.template, coords - 0.5, order=1, mode="constant",
                               cval=0).reshape(rows, cols)
    return ann, tmpl


def label_contours(labels: np.ndarray, sigma: float = 1.0,
                   margin: int = 4) -> dict[int, list[np.ndarray]]:
    """Sub-pixel (row, col) contours of every non-zero region in a label image.

    Each region's binary mask is Gaussian-blurred and contoured at the 0.5 level, so
    two touching regions get exactly the same shared border (their blurred masks sum
    to 1 there) instead of two overlapping, jittery lines.
    """
    out: dict[int, list[np.ndarray]] = {}
    if labels.size == 0 or labels.max() == 0:
        return out
    # find_objects allocates one slot per id up to the max; Allen ids reach ~6e8.
    ids, compact = np.unique(labels, return_inverse=True)
    compact = compact.reshape(labels.shape)
    if ids[0] != 0:
        compact += 1
        ids = np.concatenate([[0], ids])
    h, w = labels.shape
    m = margin + (int(round(3 * sigma)) if sigma > 0 else 0)
    for idx, sl in enumerate(ndi.find_objects(compact), start=1):
        if sl is None:
            continue
        rid = int(ids[idx])
        ys, xs = sl
        y0, y1 = max(ys.start - m, 0), min(ys.stop + m, h)
        x0, x1 = max(xs.start - m, 0), min(xs.stop + m, w)
        mask = (compact[y0:y1, x0:x1] == idx).astype(np.float32)
        if sigma > 0:
            mask = ndi.gaussian_filter(mask, sigma)
        for c in measure.find_contours(mask, 0.5):
            out.setdefault(rid, []).append(c + [y0, x0])
    return out


@lru_cache(maxsize=8)
def _contours_cached(atlas: Atlas, ap_um: float, pitch_deg: float, yaw_deg: float,
                     sigma: float) -> tuple[np.ndarray, ...]:
    ann, _ = _sample_cached(atlas, ap_um, pitch_deg, yaw_deg)
    by_region = label_contours(ann, sigma)
    step = atlas.step
    polylines = []
    for cs in by_region.values():
        for c in cs:
            # (row, col) fractional pixel index -> (s, t) plane microns.
            polylines.append((c[:, ::-1] + 0.5) * step)
    return tuple(polylines)


def boundaries(labels: np.ndarray) -> np.ndarray:
    """Pixels where the region label changes (1 px wide)."""
    b = np.zeros(labels.shape, dtype=bool)
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    return b


def label_bbox(labels: np.ndarray):
    """(x0, y0, x1, y1) in pixels of non-zero labels, or None."""
    ys, xs = np.nonzero(labels)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
