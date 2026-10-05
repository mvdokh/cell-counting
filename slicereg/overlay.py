"""Resample the atlas plane into the (downsampled) section image frame."""

from __future__ import annotations

import cv2
import numpy as np

from .atlas import Atlas, boundaries
from .io import imwrite_png, to_display
from .transform import SliceTransform


def preview_to_full(ij: np.ndarray, ds: int) -> np.ndarray:
    """Preview pixel centres -> full-resolution pixel coordinates."""
    return np.asarray(ij, dtype=float) * ds + (ds - 1) / 2


def plane_maps(tf: SliceTransform, atlas: Atlas, shape: tuple[int, int], ds: int,
               grid_step: int = 8):
    """For every preview pixel, the (col, row) position in the atlas plane image."""
    h, w = shape
    gx = np.arange(0, w + grid_step, grid_step, dtype=float)
    gy = np.arange(0, h + grid_step, grid_step, dtype=float)
    xx, yy = np.meshgrid(gx, gy)
    pts = preview_to_full(np.stack([xx.ravel(), yy.ravel()], 1), ds)
    st = tf.image_to_plane(pts)
    coarse = (st / atlas.step - 0.5).reshape(len(gy), len(gx), 2).astype(np.float32)
    fx, fy = np.meshgrid(np.arange(w, dtype=np.float32) / grid_step,
                         np.arange(h, dtype=np.float32) / grid_step)
    map_x = cv2.remap(coarse[..., 0], fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    map_y = cv2.remap(coarse[..., 1], fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return map_x, map_y


def warp_atlas(tf: SliceTransform, atlas: Atlas, shape: tuple[int, int], ds: int):
    """Return (labels, template) of the aligned atlas plane at preview resolution."""
    al = tf.al
    ann, tmpl = atlas.sample_plane(al.ap_um, al.pitch_deg, al.yaw_deg)
    map_x, map_y = plane_maps(tf, atlas, shape, ds)
    ids, inv = np.unique(ann, return_inverse=True)
    idx = cv2.remap(inv.reshape(ann.shape).astype(np.uint16), map_x, map_y, cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    labels = ids[idx]
    if ids[0] != 0:
        outside = (map_x < -0.5) | (map_y < -0.5) | (map_x > ann.shape[1] - 0.5) | \
                  (map_y > ann.shape[0] - 0.5)
        labels[outside] = 0
    template = cv2.remap(tmpl, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return labels, template


def draw_outlines(contours_px, shape: tuple[int, int], width: float = 2.0,
                  supersample: int = 3) -> np.ndarray:
    """Anti-aliased uint8 (0..255) canvas of the given (x, y) polylines.

    Drawn at ``supersample``x resolution and downsampled, so lines can be smooth and
    thinner than one output pixel wide.
    """
    h, w = shape
    k = max(1, int(supersample))
    canvas = np.zeros((h * k, w * k), np.uint8)
    lw = max(1, int(round(width * k)))
    # A single cv2.polylines call drawing every contour is far faster than one call
    # per contour: a coronal plane can have several hundred region boundaries.
    polys = [np.round(np.asarray(c) * k).astype(np.int32) for c in contours_px
             if len(c) >= 2]
    if polys:
        # Aliased (LINE_8) lines on the oversampled canvas, then an area-average
        # downsample, look just as smooth as cv2.LINE_AA here but render ~5-8x
        # faster (LINE_AA's per-pixel blending is the dominant cost otherwise).
        cv2.polylines(canvas, polys, False, 255, lw, lineType=cv2.LINE_8)
    if k > 1:
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    return canvas


def save_overlay_png(path, preview: np.ndarray, labels: np.ndarray) -> None:
    rgb = to_display(preview)
    rgb[boundaries(labels)] = (255, 255, 255)
    imwrite_png(path, rgb)
