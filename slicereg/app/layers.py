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


def full_range(dtype) -> tuple[float, float]:
    """The values an image of this dtype can hold (0-1 for float images)."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return float(max(0, np.iinfo(dtype).min)), float(np.iinfo(dtype).max)
    return 0.0, 1.0


def get_levels(project, n_channels: int, dtype, sid: int | None = None) -> list[dict]:
    """Brightness levels ``{"min", "max", "gamma"}`` per channel, set by the user.

    A section's own levels (if it has any) win over the project's shared levels;
    without either, each channel shows the dtype's full range unchanged."""
    lo, hi = full_range(dtype)
    out = [{"min": lo, "max": hi, "gamma": 1.0} for _ in range(n_channels)]
    saved = project.get_slice(sid).get("levels") if sid is not None else None
    if saved is None:
        saved = project.data.get("levels")
    for level, s in zip(out, saved or []):
        level.update({k: float(s[k]) for k in ("min", "max", "gamma") if k in s})
    return out


def store_levels(project, levels: list[dict], sid: int | None = None) -> None:
    """Save levels as the project's shared levels, or as one section's own."""
    clean = [{k: round(float(lv[k]), 4) for k in ("min", "max", "gamma")} for lv in levels]
    if sid is None:
        project.data["levels"] = clean
    else:
        project.get_slice(sid)["levels"] = clean
    project.save()


def layer_levels(layers) -> list[dict]:
    return [{"min": float(layer.contrast_limits[0]), "max": float(layer.contrast_limits[1]),
             "gamma": float(layer.gamma)} for layer in layers]


def apply_levels(layers, levels: list[dict]) -> None:
    for layer, lv in zip(layers, levels):
        lo, hi = float(lv["min"]), float(lv["max"])
        if hi <= lo:
            hi = lo + (1.0 if np.issubdtype(layer.data.dtype, np.integer) else 1e-3)
        r0, r1 = layer.contrast_limits_range
        if lo < r0 or hi > r1:
            layer.contrast_limits_range = (min(lo, r0), max(hi, r1))
        layer.contrast_limits = (lo, hi)
        layer.gamma = float(lv.get("gamma", 1.0))


def add_channels(viewer, img: np.ndarray, name: str, channels: list[dict] | None = None,
                 limits: tuple[float, float] | None = None, levels: list[dict] | None = None,
                 **kwargs):
    """Add an (H, W, C) image as one additive layer per channel.

    ``channels`` optionally gives ``{"name", "color"}`` per channel (project config);
    otherwise channels are named after default colormaps. ``levels`` (see
    ``get_levels``) or ``limits`` fix the brightness of every channel instead of
    estimating it from the data."""
    c = img.shape[2]
    if levels is not None:
        contrast = [(lv["min"], max(lv["max"], lv["min"] + 1e-3)) for lv in levels]
    elif limits is not None:
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
        layers = [viewer.add_image(img[..., 0], name=name,
                                   colormap=cmaps[0] if channels else "gray",
                                   contrast_limits=contrast[0], **kwargs)]
    else:
        layers = viewer.add_image(
            img, channel_axis=2, name=[f"{name} ({lb})" for lb in labels], colormap=cmaps,
            blending="additive", contrast_limits=contrast, **kwargs)
    if levels is not None:
        for layer, lv in zip(layers, levels):
            layer.gamma = float(lv.get("gamma", 1.0))
    return layers


def channel_index(project, name: str | None) -> int | None:
    """Index of the configured channel called ``name`` (None if not configured)."""
    names = [ch.get("name") for ch in project.data.get("channels") or []]
    return names.index(name) if name in names else None
