"""Mapping between section image pixels and atlas plane microns.

Image coordinates are ``(x, y)`` = (column, row) in full-resolution crop pixels.
Plane coordinates are ``(s, t)`` microns (see ``atlas.py``).

plane -> image:  x = W(A(p)),  A = affine, W = landmark warp in image space
image -> plane:  p = A^-1(W^-1(x)),  W^-1 solved by fixed-point iteration
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.interpolate import RBFInterpolator


@dataclass
class Alignment:
    atlas: str
    plane_size_um: tuple[float, float]
    image_size: tuple[int, int]
    ap_um: float
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    rotation_deg: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    flip: bool = False
    landmarks: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["plane_size_um"] = list(self.plane_size_um)
        d["image_size"] = list(self.image_size)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Alignment":
        keys = cls.__dataclass_fields__.keys()
        d = {k: v for k, v in d.items() if k in keys}
        d["plane_size_um"] = tuple(d["plane_size_um"])
        d["image_size"] = tuple(d["image_size"])
        return cls(**d)


def _anchors(image_size, margin: float = 0.1) -> np.ndarray:
    w, h = image_size
    xs = [-margin * w, w / 2, (1 + margin) * w]
    ys = [-margin * h, h / 2, (1 + margin) * h]
    return np.array([[x, y] for x in xs for y in ys if not (x == w / 2 and y == h / 2)])


class SliceTransform:
    def __init__(self, al: Alignment):
        self.al = al
        th = np.deg2rad(al.rotation_deg)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        flip = np.diag([-1.0 if al.flip else 1.0, 1.0])
        self.m = rot @ np.diag([al.scale_x, al.scale_y]) @ flip
        self.m_inv = np.linalg.inv(self.m)
        self.c = np.asarray(al.plane_size_um, dtype=float) / 2
        self.t = np.array([al.tx, al.ty], dtype=float)
        self._rbf = None
        if al.landmarks:
            lm = np.asarray(al.landmarks, dtype=float)
            src = self.affine(lm[:, :2])
            disp = lm[:, 2:4] - src
            anchors = _anchors(al.image_size)
            pts = np.vstack([src, anchors])
            vals = np.vstack([disp, np.zeros_like(anchors)])
            self._rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline")

    def affine(self, p: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(p) - self.c) @ self.m.T + self.t

    def affine_inv(self, x: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(x) - self.t) @ self.m_inv.T + self.c

    def displacement(self, y: np.ndarray) -> np.ndarray:
        y = np.atleast_2d(y)
        if self._rbf is None:
            return np.zeros_like(y, dtype=float)
        return self._rbf(y)

    def plane_to_image(self, p: np.ndarray) -> np.ndarray:
        y = self.affine(p)
        return y + self.displacement(y)

    def unwarp(self, x: np.ndarray, iters: int = 50, tol: float = 1e-4) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self._rbf is None:
            return x
        y = x - self.displacement(x)
        for _ in range(iters):
            y_new = x - self.displacement(y)
            if np.max(np.abs(y_new - y)) < tol:
                return y_new
            y = y_new
        return y

    def image_to_plane(self, x: np.ndarray) -> np.ndarray:
        return self.affine_inv(self.unwarp(x))


def initial_alignment(atlas, image_size, mask_bbox_px, ap_um: float, pitch_deg: float = 0.0,
                      yaw_deg: float = 0.0, rotation_deg: float = 0.0,
                      flip: bool = False) -> Alignment:
    """Fit scale and translation so the atlas brain outline matches the tissue bbox."""
    from .atlas import label_bbox

    ann, _ = atlas.sample_plane(ap_um, pitch_deg, yaw_deg)
    al = Alignment(atlas=atlas.name, plane_size_um=atlas.plane_size_um,
                   image_size=tuple(int(v) for v in image_size), ap_um=ap_um,
                   pitch_deg=pitch_deg, yaw_deg=yaw_deg, rotation_deg=rotation_deg, flip=flip)
    ab = label_bbox(ann)
    if ab is None or mask_bbox_px is None:
        w, h = image_size
        pw, ph = atlas.plane_size_um
        al.scale_x = al.scale_y = min(w / pw, h / ph)
        al.tx, al.ty = w / 2, h / 2
        return al
    s0, t0, s1, t1 = (np.asarray(ab, dtype=float) * atlas.step).tolist()
    x0, y0, x1, y1 = (float(v) for v in mask_bbox_px)
    al.scale_x = (x1 - x0) / (s1 - s0)
    al.scale_y = (y1 - y0) / (t1 - t0)
    tf = SliceTransform(al)
    centre_img = tf.affine([(s0 + s1) / 2, (t0 + t1) / 2])[0]
    al.tx = float((x0 + x1) / 2 - centre_img[0])
    al.ty = float((y0 + y1) / 2 - centre_img[1])
    return al
