"""Find individual tissue sections on a slide thumbnail."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def detect_sections(thumb: np.ndarray, min_frac: float = 0.2, pad_px: int = 4,
                    close_px: int = 9, blur_px: int = 5):
    """Detect sections on a thumbnail.

    Returns ``(boxes, threshold, dark_background)`` where boxes are
    ``[x0, y0, x1, y1]`` in thumbnail pixels, in reading order (rows, then columns).
    Components smaller than ``min_frac`` of the largest one are treated as debris.
    """
    g = np.asarray(thumb).max(axis=2).astype(np.float32)
    g = cv2.GaussianBlur(g, (blur_px | 1, blur_px | 1), 0)
    g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t8, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(g.min() + t8 / 255.0 * (g.max() - g.min()))
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    dark_background = bool(np.median(border) < threshold)
    m = g > threshold if dark_background else g < threshold
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((close_px, close_px), np.uint8))
    m = ndi.binary_fill_holes(m)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 1:
        return [], threshold, dark_background
    areas = stats[1:, cv2.CC_STAT_AREA]
    h, w = g.shape
    boxes = []
    for x, y, bw, bh, area in stats[1:].tolist():
        if area < min_frac * areas.max():
            continue
        boxes.append([max(0, x - pad_px), max(0, y - pad_px),
                      min(w, x + bw + pad_px), min(h, y + bh + pad_px)])
    return sort_reading_order(boxes), threshold, dark_background


def sort_reading_order(boxes: list[list[int]]) -> list[list[int]]:
    if not boxes:
        return []
    heights = np.array([b[3] - b[1] for b in boxes])
    tol = 0.5 * np.median(heights)
    remaining = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    rows: list[list[list[int]]] = []
    for b in remaining:
        cy = (b[1] + b[3]) / 2
        if rows and abs(cy - np.mean([(r[1] + r[3]) / 2 for r in rows[-1]])) < tol:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: b[0])]
