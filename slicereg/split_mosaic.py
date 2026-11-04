"""Split a multi-section Zeiss LSM mosaic into one cropped LSM per section.

A slide scan that tiles several sections in a regular grid (for example 3 rows
by 2 columns) is cropped into separate ``c{col}r{row}.lsm`` files. The Zeiss
``CZ_LSMINFO`` tag is copied from the source and patched so the crops stay
readable as LSM (pixel size, ZCYX axes) by ImageJ, Fiji and ``read_section``.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np
import tifffile

_CZ = np.dtype(tifffile.tifffile.TIFF.CZ_LSMINFO)


def _lsminfo_raw(path: Path) -> bytes:
    with tifffile.TiffFile(str(path)) as tif:
        if not tif.is_lsm:
            raise ValueError(f"{path.name} is not a Zeiss LSM file")
        tag = tif.pages[0].tags[34412]
        fh = tif.filehandle
        fh.seek(tag.valueoffset)
        return fh.read(int(tag.count))


def _patch_lsminfo(raw: bytes, width: int, height: int, z: int,
                   thumb_w: int, thumb_h: int) -> bytes:
    """Update dimensions and clear file-absolute Offset* pointers."""
    out = bytearray(raw)
    struct.pack_into("<iii", out, 8, int(width), int(height), int(z))
    struct.pack_into("<ii", out, 32, int(thumb_w), int(thumb_h))
    for name in _CZ.names:
        if name.startswith("Offset"):
            off = _CZ.fields[name][1]
            if off + 4 <= len(out):
                struct.pack_into("<I", out, off, 0)
    return bytes(out)


def _thumbnail(plane: np.ndarray, thumb_h: int, thumb_w: int) -> np.ndarray:
    """Downsample a ``(C, Y, X)`` plane to ``(C, thumb_h, thumb_w)``."""
    _, h, w = plane.shape
    ys = (np.linspace(0, h, thumb_h, endpoint=False)).astype(np.int64)
    xs = (np.linspace(0, w, thumb_w, endpoint=False)).astype(np.int64)
    return np.ascontiguousarray(plane[:, ys][:, :, xs])


def _thumb_size(height: int, width: int) -> tuple[int, int]:
    scale = max(height, width) / 128.0
    return max(1, int(round(height / scale))), max(1, int(round(width / scale)))


def _write_lsm_plane(tif: tifffile.TiffWriter, plane: np.ndarray, *,
                     lsminfo: bytes | None, thumb_h: int, thumb_w: int) -> None:
    """Append one ``(C,Y,X)`` plane (plus its thumbnail) to an open LSM writer."""
    plane = np.ascontiguousarray(plane)
    kwargs: dict = {
        "photometric": "rgb",
        "planarconfig": "separate",
        "compression": None,
        "metadata": None,
        "subfiletype": 0,
    }
    if lsminfo is not None:
        kwargs["extratags"] = [(34412, "B", len(lsminfo), lsminfo, True)]
    tif.write(plane, **kwargs)
    tif.write(_thumbnail(plane, thumb_h, thumb_w),
              photometric="rgb", planarconfig="separate",
              compression=None, metadata=None, subfiletype=1)


def write_lsm(path: Path, data: np.ndarray, lsminfo: bytes) -> None:
    """Write ``data`` as a Zeiss-style LSM. ``data`` is ``(C,Y,X)`` or ``(Z,C,Y,X)``."""
    path = Path(path)
    if data.ndim == 3:
        data = data[None]
    if data.ndim != 4:
        raise ValueError(f"expected (C,Y,X) or (Z,C,Y,X), got shape {data.shape}")
    z, _c, h, w = data.shape
    thumb_h, thumb_w = _thumb_size(h, w)
    info = _patch_lsminfo(lsminfo, w, h, z, thumb_w, thumb_h)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(str(path), bigtiff=True) as tif:
        for zi in range(z):
            _write_lsm_plane(tif, data[zi],
                             lsminfo=info if zi == 0 else None,
                             thumb_h=thumb_h, thumb_w=thumb_w)


def _default_names(rows: int, cols: int, first_col: int, first_row: int,
                   mip: bool) -> list[str]:
    suffix = "_mip" if mip else ""
    names = []
    for r in range(rows):
        for c in range(cols):
            names.append(f"c{first_col + c}r{first_row + r}{suffix}")
    return names


def split_mosaic(path: Path | str, rows: int, cols: int, *,
                 first_col: int = 1, first_row: int = 1,
                 names: list[str] | None = None,
                 out_dir: Path | str | None = None) -> list[Path]:
    """Crop a regular ``rows`` × ``cols`` LSM mosaic into one file per tile.

    Tiles are named in row-major order (left-to-right, top-to-bottom). By
    default that is ``c{col}r{row}.lsm`` starting at ``first_col`` /
    ``first_row``; a ``_mip`` suffix is kept when the source name contains
    ``_mip``. The source file is left unchanged.
    """
    path = Path(path)
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    out_dir = Path(out_dir) if out_dir is not None else path.parent
    mip = bool(re.search(r"_mip", path.stem, flags=re.IGNORECASE))
    if names is None:
        names = _default_names(rows, cols, first_col, first_row, mip)
    if len(names) != rows * cols:
        raise ValueError(f"need {rows * cols} names, got {len(names)}")

    lsminfo = _lsminfo_raw(path)
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes = series.axes
        shape = series.shape
        if axes not in ("CYX", "ZCYX"):
            raise ValueError(f"{path.name}: expected CYX or ZCYX axes, got {axes}")
        height, width = shape[-2], shape[-1]
        if height % rows or width % cols:
            raise ValueError(
                f"{path.name}: shape {height}x{width} does not divide evenly into "
                f"{rows} rows x {cols} cols")
        tile_h, tile_w = height // rows, width // cols

        boxes = []
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            y0, x0 = r * tile_h, c * tile_w
            boxes.append((y0, y0 + tile_h, x0, x0 + tile_w))

        # Prefer a memmap when the pixels are contiguous (typical for MIPs).
        try:
            arr = tifffile.memmap(str(path), mode="r")
        except ValueError:
            arr = None

        written: list[Path] = []
        if arr is not None:
            for name, (y0, y1, x0, x1) in zip(names, boxes):
                tile = np.asarray(arr[..., y0:y1, x0:x1])
                out = out_dir / f"{name}.lsm"
                write_lsm(out, tile, lsminfo)
                written.append(out)
                print(f"wrote {out.name}  ({tile.shape})")
        else:
            # Stream Z planes once: crop every tile from each plane and append.
            z = shape[0] if axes == "ZCYX" else 1
            thumb_h, thumb_w = _thumb_size(tile_h, tile_w)
            infos = [_patch_lsminfo(lsminfo, tile_w, tile_h, z, thumb_w, thumb_h)
                     for _ in names]
            out_dir.mkdir(parents=True, exist_ok=True)
            paths = [out_dir / f"{name}.lsm" for name in names]
            writers = [tifffile.TiffWriter(str(p), bigtiff=True) for p in paths]
            try:
                for zi, page in enumerate(series.pages):
                    plane = page.asarray()
                    for wri, info, (y0, y1, x0, x1) in zip(writers, infos, boxes):
                        _write_lsm_plane(wri, plane[:, y0:y1, x0:x1],
                                         lsminfo=info if zi == 0 else None,
                                         thumb_h=thumb_h, thumb_w=thumb_w)
                    print(f"  plane {zi + 1}/{z}")
            finally:
                for wri in writers:
                    wri.close()
            for p in paths:
                written.append(p)
                print(f"wrote {p.name}")
    return written
