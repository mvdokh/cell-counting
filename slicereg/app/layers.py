"""Small napari helpers shared by the slide and slice views."""

from __future__ import annotations

import numpy as np
from napari.utils.colormaps import Colormap

from ..export import CELL_COLORS  # noqa: F401

DISPLAY_DEFAULTS = {
    "outlines": False,
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


def get_levels(project, n_channels: int, dtype, sid: int | None = None,
               key: str = "levels") -> list[dict]:
    """Brightness levels ``{"min", "max", "gamma"}`` per channel, set by the user.

    A section's own levels (stored under ``key`` on the slice) win when ``sid`` is
    set; for section view (``key="levels"``) that falls back to the project's shared
    levels. Overview per-section preview levels (``key="overview_levels"`` with a
    ``sid``) fall back to the dtype full range, not the shared overview contrast.
    Without either, each channel shows the dtype's full range unchanged."""
    lo, hi = full_range(dtype)
    out = [{"min": lo, "max": hi, "gamma": 1.0} for _ in range(n_channels)]
    saved = None
    if sid is not None:
        saved = project.get_slice(sid).get(key)
        if saved is None and key == "levels":
            saved = project.data.get(key)
    else:
        saved = project.data.get(key)
    for level, s in zip(out, saved or []):
        level.update({k: float(s[k]) for k in ("min", "max", "gamma") if k in s})
    return out


def store_levels(project, levels: list[dict], sid: int | None = None,
                 key: str = "levels") -> None:
    """Save levels under ``key`` on the project, or on one section when ``sid`` is set."""
    clean = [{k: round(float(lv[k]), 4) for k in ("min", "max", "gamma")} for lv in levels]
    if sid is None:
        project.data[key] = clean
    else:
        project.get_slice(sid)[key] = clean
    project.save()


def remap_levels(img: np.ndarray, levels: list[dict]) -> np.ndarray:
    """Bake per-channel min/max/gamma into pixel values (dtype full range out).

    Used for per-section overview preview brightness on a shared canvas layer."""
    img = np.asarray(img)
    out = np.zeros_like(img, dtype=np.float32)
    lo_out, hi_out = full_range(img.dtype)
    n_ch = img.shape[2] if img.ndim == 3 else 1
    for c in range(n_ch):
        channel = img[..., c] if img.ndim == 3 else img
        lv = levels[c] if c < len(levels) else {"min": lo_out, "max": hi_out, "gamma": 1.0}
        lo, hi = float(lv["min"]), float(lv["max"])
        gamma = float(lv.get("gamma", 1.0))
        if hi <= lo:
            hi = lo + (1.0 if np.issubdtype(img.dtype, np.integer) else 1e-3)
        t = np.clip((channel.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0) ** gamma
        mapped = t * (hi_out - lo_out) + lo_out
        if img.ndim == 3:
            out[..., c] = mapped
        else:
            out = mapped
    return np.ascontiguousarray(out.astype(img.dtype, copy=False))


def equalize_channels(img: np.ndarray, mask: np.ndarray | None = None,
                      clip: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Stretch each channel's tissue pixels to the dtype's full range.

    Used for the slide overview so sections imaged with different laser settings
    look about the same before the shared overview brightness controls are applied.
    ``clip`` is the low/high percentile of tissue values mapped to black / white."""
    img = np.asarray(img)
    out = np.zeros_like(img)
    if mask is None:
        mask = img.any(axis=-1) if img.ndim == 3 else img > 0
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != img.shape[:2] or not mask.any():
        mask = np.ones(img.shape[:2], bool)
    lo_out, hi_out = full_range(img.dtype)
    for c in range(img.shape[-1] if img.ndim == 3 else 1):
        channel = img[..., c] if img.ndim == 3 else img
        vals = channel[mask]
        lo, hi = (float(v) for v in np.percentile(vals, list(clip)))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            # Flat channel (e.g. saturated or empty): show tissue at full brightness.
            filled = np.zeros_like(channel)
            filled[mask] = hi_out
            if img.ndim == 3:
                out[..., c] = filled.astype(img.dtype, copy=False)
            else:
                out = filled.astype(img.dtype, copy=False)
            continue
        stretched = (channel.astype(np.float32) - lo) / (hi - lo)
        stretched = np.clip(stretched, 0.0, 1.0) * (hi_out - lo_out) + lo_out
        if img.ndim == 3:
            out[..., c] = stretched.astype(img.dtype, copy=False)
        else:
            out = stretched.astype(img.dtype, copy=False)
    if img.ndim == 3:
        out[~mask] = 0
    else:
        out = np.where(mask, out, 0)
    return np.ascontiguousarray(out)


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
