import json

import numpy as np
import pandas as pd
import pytest
import tifffile

from slicereg.folder import (CONFIG_NAME, DEFAULT_CONFIG, _stitch, find_sections, grid_layout,
                             load_config, read_section)
from slicereg.io import Project


def touch(folder, *names):
    for n in names:
        (folder / n).write_bytes(b"")


def test_find_sections_uses_mip_files_in_column_order(tmp_path):
    touch(tmp_path, "c5_r1_10x_stitch_MIP.lsm", "c4_r2_10x_stitch_MIP.lsm",
          "c4_r1_10x_stitch_MIP.lsm", "c4_r1_10x_stitch.lsm", "notes.txt")
    names = [s["name"] for s in find_sections(tmp_path, DEFAULT_CONFIG)]
    assert names == ["c4_r1", "c4_r2", "c5_r1"]
    by_row = [s["name"] for s in find_sections(tmp_path, {**DEFAULT_CONFIG, "order": "row"})]
    assert by_row == ["c4_r1", "c5_r1", "c4_r2"]


def test_load_config_writes_editable_default(tmp_path):
    config, created = load_config(tmp_path)
    assert created and (tmp_path / CONFIG_NAME).exists()
    saved = json.loads((tmp_path / CONFIG_NAME).read_text())
    saved["order"] = "row"
    (tmp_path / CONFIG_NAME).write_text(json.dumps(saved))
    config, created = load_config(tmp_path)
    assert not created and config["order"] == "row" and config["channels"]


def test_grid_layout_places_by_column_and_row():
    sizes = {1: (4, 1, 100.0, 50.0), 2: (4, 2, 80.0, 60.0), 3: (6, 1, 120.0, 40.0)}
    pos = grid_layout(sizes, gap=10.0)
    assert pos[1] == (0.0, 0.0)
    assert pos[2] == (10.0, 60.0)          # centred in the 100-wide column, below row 1
    assert pos[3] == (110.0, 5.0)          # next column (not a gap for missing c5)


def test_stitch_places_tiles_with_overlap():
    rng = np.random.default_rng(0)
    full = rng.integers(0, 255, size=(2, 30, 50)).astype(np.uint8)
    offsets = np.array([[0, 0], [30, 0], [0, 15], [30, 15]], float)
    tiles = np.stack([full[:, y:y + 15, x:x + 20] for x, y in offsets.astype(int)])
    out = _stitch(tiles, offsets + 7.0)
    assert out.shape == (2, 30, 50)
    mask = np.zeros((30, 50), bool)
    for x, y in offsets.astype(int):
        mask[y:y + 15, x:x + 20] = True
    assert np.array_equal(out[:, mask], full[:, mask])


def test_read_section_max_projects_z(tmp_path):
    stack = np.zeros((3, 2, 40, 60), np.uint8)
    stack[1, 0, 5, 5] = 200
    stack[2, 1, 6, 6] = 100
    p = tmp_path / "c1_r1_MIP.tif"
    tifffile.imwrite(p, stack, metadata={"axes": "ZCYX"})
    img, pixel_um = read_section(p)
    assert img.shape == (40, 60, 2) and pixel_um is None
    assert img[5, 5, 0] == 200 and img[6, 6, 1] == 100


def test_folder_project_imports_sections(tmp_path):
    folder = tmp_path / "scans"
    folder.mkdir()
    (folder / CONFIG_NAME).write_text(json.dumps(
        {"pattern": r"c(?P<col>\d+)_r(?P<row>\d+)_MIP\.tif$", "pixel_um": 2.5}))
    for name in ("c1_r1", "c1_r2"):
        img = np.zeros((2, 300, 400), np.uint8)
        img[1, 110:290, 120:390] = 120
        img[1, 0:20, 0:20] = 120          # a bit of the neighbouring section
        tifffile.imwrite(folder / f"{name}_MIP.tif", img, metadata={"axes": "CYX"})
    project = Project.open_or_create(folder)
    assert project.is_folder and project.root.name == "scans_project"
    assert [s["name"] for s in project.slices] == ["c1_r1", "c1_r2"]
    assert project.sync_folder() == [1, 2]
    project.import_section(1)
    s = project.get_slice(1)
    x0, y0, x1, y1 = s["crop"]
    assert s["pixel_um"] == 2.5 and 20 < x0 < 120 and 20 < y0 < 110 and x1 == 400 and y1 == 300
    img = project.load_image(1)
    assert img.shape == (y1 - y0, x1 - x0, 2) and s["size"] == [x1 - x0, y1 - y0]
    assert img[150 - y0, 200 - x0, 1] == 120
    assert project.load_mask(1)[150 - y0, 200 - x0]

    (folder / "c2_r1_MIP.tif").write_bytes((folder / "c1_r1_MIP.tif").read_bytes())
    assert project.sync_folder() == [2, 3]
    assert {s["name"]: s["id"] for s in project.slices} == {"c1_r1": 1, "c1_r2": 2, "c2_r1": 3}
    assert Project.open_or_create(folder).root == project.root


def test_turning_auto_crop_off_reimports_and_moves_cells(tmp_path):
    folder = tmp_path / "scans"
    folder.mkdir()
    config = {"pattern": r"c(?P<col>\d+)_r(?P<row>\d+)_MIP\.tif$", "pixel_um": 2.5}
    (folder / CONFIG_NAME).write_text(json.dumps(config))
    img = np.zeros((2, 300, 400), np.uint8)
    img[1, 110:290, 120:390] = 120
    tifffile.imwrite(folder / "c1_r1_MIP.tif", img, metadata={"axes": "CYX"})
    project = Project.open_or_create(folder)
    project.import_section(1)
    x0, y0 = project.get_slice(1)["crop"][:2]
    project.save_cells(1, pd.DataFrame({"x_px": [200.0 - x0], "y_px": [150.0 - y0]}))

    (folder / CONFIG_NAME).write_text(json.dumps({**config, "auto_crop": False}))
    assert project.sync_folder() == [1]
    project.import_section(1)
    assert project.get_slice(1)["crop"] == [0, 0, 400, 300]
    cells = project.load_cells(1)
    assert (cells.loc[0, "x_px"], cells.loc[0, "y_px"]) == (200.0, 150.0)
    assert project.sync_folder() == []


def test_folder_without_matching_files_explains(tmp_path):
    with pytest.raises(FileNotFoundError, match="pattern"):
        Project.open_or_create(tmp_path)
