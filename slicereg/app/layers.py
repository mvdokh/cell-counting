"""Small napari helpers shared by the slide and slice views."""

from __future__ import annotations

import numpy as np
from napari.utils.colormaps import Colormap

from ..export import CELL_COLORS  # noqa: F401

DISPLAY_DEFAULTS = {
    "outlines": True,
    "regions": False,
    "smooth": True,
    "sigma": 1.0,
    "width": 2.0,
    "color": [1.0, 1.0, 1.0],
    "side_by_side": False,
    "cell_size": None,
}


def outline_colormap(rgb) -> Colormap:
    """Transparent -> ``rgb`` colormap for the outline canvas."""
    r, g, b = (float(v) for v in rgb)
    return Colormap([[0.0, 0.0, 0.0, 0.0], [r, g, b, 1.0]])

CHANNEL_COLORMAPS = {
    1: ["gray"],
    2: ["green", "magenta"],
    3: ["red", "green", "blue"],
}


def contrast_limits(channel: np.ndarray) -> tuple[float, float]:
    sub = np.asarray(channel[:: max(1, channel.shape[0] // 512), :: max(1, channel.shape[1] // 512)])
    lo, hi = np.percentile(sub, [0.5, 99.9])
    return float(lo), float(max(hi, lo + 1))


def add_channels(viewer, img: np.ndarray, name: str, channels: list[dict] | None = None,
                 limits: tuple[float, float] | None = None, **kwargs):
    """Add an (H, W, C) image as one additive layer per channel.

    ``channels`` optionally gives ``{"name", "color"}`` per channel (project config);
    otherwise channels are named after default colormaps. ``limits`` fixes the
    contrast limits of every channel instead of estimating them from the data."""
    c = img.shape[2]
    if limits is not None:
        contrast = [tuple(limits)] * c
    else:
        contrast = [contrast_limits(img[..., i]) for i in range(c)]
    cmaps = CHANNEL_COLORMAPS.get(c) or [["red", "green", "blue", "magenta", "cyan", "yellow"][i % 6]
                                         for i in range(c)]
    labels = list(cmaps)
    if channels and len(channels) == c:
        cmaps = [ch.get("color", cm) for ch, cm in zip(channels, cmaps)]
        labels = [ch.get("name", cm) for ch, cm in zip(channels, cmaps)]
    if c == 1:
        return [viewer.add_image(img[..., 0], name=name, colormap=cmaps[0] if channels else "gray",
                                 contrast_limits=contrast[0], **kwargs)]
    return viewer.add_image(
        img, channel_axis=2, name=[f"{name} ({lb})" for lb in labels], colormap=cmaps,
        blending="additive", contrast_limits=contrast, **kwargs)


def channel_index(project, name: str | None) -> int | None:
    """Index of the configured channel called ``name`` (None if not configured)."""
    names = [ch.get("name") for ch in project.data.get("channels") or []]
    return names.index(name) if name in names else None
