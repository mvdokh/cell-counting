"""Split a regular-grid LSM mosaic into one file per section."""

from pathlib import Path

import numpy as np
import tifffile

from slicereg.folder import read_section
from slicereg.split_mosaic import split_mosaic, write_lsm


def _fake_lsminfo(width: int, height: int, z: int = 1, channels: int = 3) -> bytes:
    """Minimal CZ_LSMINFO blob (ScanType 0 = ZCYX) with a 1.25 um pixel."""
    raw = bytearray(464)
    import struct

    struct.pack_into("<II", raw, 0, 67127628, 464)  # magic, structure size
    struct.pack_into("<iiii", raw, 8, width, height, z, channels)
    struct.pack_into("<i", raw, 24, 1)  # DimensionTime
    struct.pack_into("<ii", raw, 32, 16, 16)  # ThumbnailX/Y
    vox = 1.25e-6
    struct.pack_into("<ddd", raw, 40, vox, vox, 1e-5)  # VoxelSize X/Y/Z
    return bytes(raw)


def test_split_mosaic_writes_named_lsm_crops(tmp_path: Path):
    rows, cols = 2, 2
    tile_h, tile_w, c = 40, 60, 3
    mosaic = np.zeros((c, rows * tile_h, cols * tile_w), dtype=np.uint8)
    # paint each tile with a unique value so we can check the crop
    values = {}
    for r in range(rows):
        for col in range(cols):
            val = 10 * (r + 1) + (col + 1)
            values[(r, col)] = val
            mosaic[:, r * tile_h:(r + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = val
    src = tmp_path / "c1r1-c2r2_mip.lsm"
    write_lsm(src, mosaic, _fake_lsminfo(cols * tile_w, rows * tile_h))

    written = split_mosaic(src, rows, cols, first_col=1, first_row=1)
    assert [p.name for p in written] == [
        "c1r1_mip.lsm", "c2r1_mip.lsm", "c1r2_mip.lsm", "c2r2_mip.lsm"]
    assert src.exists()  # master kept

    for r in range(rows):
        for col in range(cols):
            img, px = read_section(tmp_path / f"c{col + 1}r{r + 1}_mip.lsm")
            assert img.shape == (tile_h, tile_w, c)
            assert abs(px - 1.25) < 1e-6
            assert img[0, 0, 0] == values[(r, col)]


def test_write_lsm_z_stack_roundtrip(tmp_path: Path):
    data = np.arange(2 * 3 * 20 * 30, dtype=np.uint8).reshape(2, 3, 20, 30)
    path = tmp_path / "stack.lsm"
    write_lsm(path, data, _fake_lsminfo(30, 20, z=2))
    with tifffile.TiffFile(str(path)) as tif:
        assert tif.is_lsm
        assert tif.series[0].axes == "ZCYX"
        assert tif.series[0].shape == (2, 3, 20, 30)
    img, px = read_section(path)  # max-projects Z
    assert img.shape == (20, 30, 3) and px is not None
