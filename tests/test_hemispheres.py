import json

import numpy as np
import pandas as pd
import pytest
import tifffile

from slicereg.app.positions_plot import section_positions
from slicereg.atlas import Atlas
from slicereg.export import map_cells
from slicereg.folder import CONFIG_NAME, DEFAULT_CONFIG, load_hemispheres
from slicereg.io import Project, _right_hemisphere_left
from slicereg.transform import Alignment, SliceTransform


@pytest.fixture(scope="module")
def atlas():
    return Atlas("allen_mouse_25um")


def test_load_hemispheres_matches_names_loosely(tmp_path):
    (tmp_path / "hemispheres.csv").write_text("Slice,X1,X2\nc4_r1,L,R\nC4-R2,right,left\n")
    sides = load_hemispheres(tmp_path, DEFAULT_CONFIG)
    assert sides == {"c4r1": "L,R", "c4r2": "R,L"}
    assert load_hemispheres(tmp_path, {**DEFAULT_CONFIG, "hemispheres": "none.csv"}) == {}


def test_load_hemispheres_rejects_bad_values(tmp_path):
    (tmp_path / "hemispheres.csv").write_text("slice,x1,x2\nc4_r1,L,L\n")
    with pytest.raises(ValueError):
        load_hemispheres(tmp_path, DEFAULT_CONFIG)
    (tmp_path / "hemispheres.csv").write_text("slice,left\nc4_r1,L\n")
    with pytest.raises(ValueError):
        load_hemispheres(tmp_path, DEFAULT_CONFIG)


def _project(tmp_path):
    folder = tmp_path / "scans"
    folder.mkdir()
    (folder / CONFIG_NAME).write_text(json.dumps(
        {"pattern": r"c(?P<col>\d+)_r(?P<row>\d+)_MIP\.tif$", "pixel_um": 25.0,
         "auto_crop": False}))
    for name in ("c1_r1", "c1_r2"):
        img = np.zeros((2, 320, 456), np.uint8)
        img[1, 40:280, 30:430] = 100
        img[1, 100:120, 50:80] = 250          # marks the image's left side
        tifffile.imwrite(folder / f"{name}_MIP.tif", img, metadata={"axes": "CYX"})
    project = Project.open_or_create(folder)
    for sid in project.sync_folder():
        project.import_section(sid)
    return folder, project


def _align(atlas, project, sid, **kw):
    w, h = project.get_slice(sid)["size"]
    al = Alignment(atlas=atlas.name, plane_size_um=atlas.plane_size_um, image_size=(w, h),
                   ap_um=7000.0, pitch_deg=2.0, yaw_deg=3.0, rotation_deg=5.0,
                   scale_x=0.04, scale_y=0.04, tx=w / 2 + 6, ty=h / 2 - 4, **kw)
    al.landmarks = [[4000.0, 3000.0, *(SliceTransform(al).plane_to_image([[4000.0, 3000.0]])[0]
                                       + [7.0, -5.0])]]
    project.save_alignment(sid, al.to_dict())
    xy = np.array([[120.0, 160.0], [340.0, 160.0], [228.0, 100.0]])
    project.save_cells(sid, map_cells(atlas, al, xy, ["rfp"] * 3, sid))
    return al


def test_apply_hemispheres_mirrors_section_and_atlas(tmp_path, atlas):
    folder, project = _project(tmp_path)
    before = _align(atlas, project, 1)
    _align(atlas, project, 2, flip=True)
    cells_before = project.load_cells(1)
    img_before = project.load_image(1)
    assert project.apply_hemispheres(atlas) == []    # no hemispheres file yet

    (folder / "hemispheres.csv").write_text("slice,x1,x2\nc1_r1,L,R\nc1_r2,R,L\n")
    project.sync_folder()
    assert project.apply_hemispheres(atlas) == [1, 2]
    w = project.get_slice(1)["size"][0]
    assert np.array_equal(project.load_image(1), img_before[:, ::-1])
    assert project.load_mask(1).shape == img_before.shape[:2]

    after = project.load_alignment(1)
    assert after["flip"] is False and _right_hemisphere_left(after)
    assert np.isclose(after["rotation_deg"], -5.0) and np.isclose(after["yaw_deg"], -3.0)
    # the outline is on the same tissue: mirrored plane points land on mirrored pixels
    pw = atlas.plane_size_um[0]
    p = np.array([[3000.0, 2000.0], [6000.0, 4500.0], [4000.0, 3000.0]])
    old = SliceTransform(before).plane_to_image(p)
    new = SliceTransform(Alignment.from_dict(after)).plane_to_image(
        np.c_[pw - p[:, 0], p[:, 1]])
    assert np.allclose(new, np.c_[w - 1 - old[:, 0], old[:, 1]], atol=0.05)

    cells = project.load_cells(1)
    assert np.allclose(cells["x_px"], w - 1 - cells_before["x_px"])
    swap = {"left": "right", "right": "left"}
    assert list(cells["hemisphere"][:2]) == [swap[h] for h in cells_before["hemisphere"][:2]]

    other = project.load_alignment(2)       # image already right way round; atlas mirrored
    assert other["flip"] is False and np.isclose(other["rotation_deg"], 5.0)
    assert np.isclose(other["yaw_deg"], -3.0)
    assert not project.get_slice(2).get("mirrored")

    assert project.apply_hemispheres(atlas) == []
    project.import_section(1)               # re-reading the file keeps it mirrored
    assert np.array_equal(project.load_image(1), img_before[:, ::-1])
    again = project.load_alignment(1)
    assert np.isclose(again["tx"], after["tx"]) and again["flip"] is False
    assert np.allclose(project.load_cells(1)["x_px"], cells["x_px"])


def test_section_positions_in_slide_order(tmp_path, atlas):
    _, project = _project(tmp_path)
    _align(atlas, project, 2)
    names, sids, ap, pitch = section_positions(project)
    assert names == ["c1_r1", "c1_r2"] and sids == [1, 2]
    assert np.isnan(ap[0]) and ap[1] == 7000.0 and pitch[1] == 2.0
