"""Small napari helpers shared by the slide and slice views."""

from __future__ import annotations

import numpy as np

from ..export import CELL_COLORS  # noqa: F401

CHANNEL_COLORMAPS = {
    1: ["gray"],
    2: ["green", "magenta"],
    3: ["red", "green", "blue"],
}


def contrast_limits(channel: np.ndarray) -> tuple[float, float]:
    sub = np.asarray(channel[:: max(1, channel.shape[0] // 512), :: max(1, channel.shape[1] // 512)])
    lo, hi = np.percentile(sub, [0.5, 99.9])
    return float(lo), float(max(hi, lo + 1))


def add_channels(viewer, img: np.ndarray, name: str, **kwargs):
    """Add an (H, W, C) image as one additive layer per channel."""
    c = img.shape[2]
    cmaps = CHANNEL_COLORMAPS.get(c) or [["red", "green", "blue", "magenta", "cyan", "yellow"][i % 6]
                                         for i in range(c)]
    if c == 1:
        return [viewer.add_image(img[..., 0], name=name, colormap="gray",
                                 contrast_limits=contrast_limits(img[..., 0]), **kwargs)]
    return viewer.add_image(
        img, channel_axis=2, name=[f"{name} ({cm})" for cm in cmaps], colormap=cmaps,
        blending="additive", contrast_limits=[contrast_limits(img[..., i]) for i in range(c)],
        **kwargs)
