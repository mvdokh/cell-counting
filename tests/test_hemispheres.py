import json

import numpy as np
import pytest
import tifffile

from slicereg.app.positions_plot import section_positions
from slicereg.atlas import Atlas
from slicereg.export import map_cells
from slicereg.folder import CONFIG_NAME, DEFAULT_CONFIG, load_hemispheres
from slicereg.io import (Project, _left_hemisphere_left, _mirror_atlas,
                         ensure_bigger_ml_on_image_left)
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
        img[1, 100:120, 50:80] = 250
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
    al = Alignment.from_dict(project.load_alignment(sid))
    xy = np.array([[120.0, 160.0], [340.0, 160.0], [228.0, 100.0]])
    project.save_cells(sid, map_cells(atlas, al, xy, ["rfp"] * 3, sid))
    return al


def test_ensure_bigger_ml_on_image_left():
    al = {"flip": False, "yaw_deg": 3.0, "rotation_deg": 0.0,
          "plane_size_um": [11400.0, 8000.0], "landmarks": [[100.0, 200.0, 1.0, 2.0]]}
    assert not _left_hemisphere_left(al)
    assert ensure_bigger_ml_on_image_left(al) is True
    assert _left_hemisphere_left(al) and al["flip"] is True
    assert np.isclose(al["yaw_deg"], -3.0)
    assert ensure_bigger_ml_on_image_left(al) is False


def test_save_alignment_enforces_bigger_ml_on_left(tmp_path, atlas):
    _, project = _project(tmp_path)
    w, h = project.get_slice(1)["size"]
    al = Alignment(atlas=atlas.name, plane_size_um=atlas.plane_size_um, image_size=(w, h),
                   ap_um=7000.0, flip=False, yaw_deg=4.0, rotation_deg=0.0,
                   scale_x=0.04, scale_y=0.04, tx=w / 2, ty=h / 2)
    project.save_alignment(1, al.to_dict())
    saved = project.load_alignment(1)
    assert _left_hemisphere_left(saved) and saved["flip"] is True
    assert np.isclose(saved["yaw_deg"], -4.0)


def test_fix_atlas_ml_does_not_touch_images(tmp_path, atlas):
    _, project = _project(tmp_path)
    w, h = project.get_slice(1)["size"]
    # Write a wrong-way alignment without going through enforce.
    al = Alignment(atlas=atlas.name, plane_size_um=atlas.plane_size_um, image_size=(w, h),
                   ap_um=7000.0, flip=False, yaw_deg=2.0, rotation_deg=0.0,
                   scale_x=0.04, scale_y=0.04, tx=w / 2, ty=h / 2)
    project.save_alignment(1, al.to_dict(), enforce_ml=False)
    img_before = project.load_image(1).copy()
    assert not _left_hemisphere_left(project.load_alignment(1))

    assert project.fix_atlas_ml_orientation(atlas) == [1]
    assert _left_hemisphere_left(project.load_alignment(1))
    assert np.array_equal(project.load_image(1), img_before)
    assert project.get_slice(1).get("mirrored") is False


def test_apply_hemispheres_mirrors_rl_image_only(tmp_path, atlas):
    folder, project = _project(tmp_path)
    _align(atlas, project, 1, flip=False)
    before2 = _align(atlas, project, 2, flip=True)
    cells_before = project.load_cells(2)
    img1_before = project.load_image(1)
    img2_before = project.load_image(2)

    (folder / "hemispheres.csv").write_text("slice,x1,x2\nc1_r1,L,R\nc1_r2,R,L\n")
    project.sync_folder()
    assert project.apply_hemispheres(atlas) == [2]

    assert np.array_equal(project.load_image(1), img1_before)
    assert _left_hemisphere_left(project.load_alignment(1))

    w = project.get_slice(2)["size"][0]
    assert np.array_equal(project.load_image(2), img2_before[:, ::-1])
    after2 = project.load_alignment(2)
    p = np.array([[3000.0, 2000.0], [6000.0, 4500.0], [4000.0, 3000.0]])
    old = SliceTransform(before2).plane_to_image(p)
    new = SliceTransform(Alignment.from_dict(after2)).plane_to_image(p)
    assert np.allclose(new, np.c_[w - 1 - old[:, 0], old[:, 1]], atol=0.05)
    cells = project.load_cells(2)
    assert np.allclose(cells["x_px"], w - 1 - cells_before["x_px"])
    assert list(cells["hemisphere"][:2]) == list(cells_before["hemisphere"][:2])


def test_section_positions_in_slide_order(tmp_path, atlas):
    _, project = _project(tmp_path)
    _align(atlas, project, 2)
    names, sids, ap, pitch = section_positions(project)
    assert names == ["c1_r1", "c1_r2"] and sids == [1, 2]
    assert np.isnan(ap[0]) and ap[1] == 7000.0 and pitch[1] == 2.0


def test_flip_section_mirrors_alignment_or_keeps_it(tmp_path, atlas):
    _, project = _project(tmp_path)
    before = _align(atlas, project, 1)
    img_before = project.load_image(1)
    cells_before = project.load_cells(1)
    w = project.get_slice(1)["size"][0]

    project.flip_section_horizontal(1, mirror_alignment=True)
    assert np.array_equal(project.load_image(1), img_before[:, ::-1])
    assert project.get_slice(1).get("user_flipped") is True
    assert np.allclose(project.load_cells(1)["x_px"], w - 1 - cells_before["x_px"])
    flipped_img = project.load_image(1).copy()
    project.apply_hemispheres(atlas)  # may fix atlas ML; must not re-flip the image
    assert np.array_equal(project.load_image(1), flipped_img)
    assert _left_hemisphere_left(project.load_alignment(1))

    al_kept = dict(project.load_alignment(1))
    project.flip_section_horizontal(1, mirror_alignment=False)
    assert np.array_equal(project.load_image(1), img_before)
    assert project.get_slice(1).get("user_flipped") is False
    kept = project.load_alignment(1)
    assert kept["flip"] is al_kept["flip"] and np.isclose(kept["tx"], al_kept["tx"])
    assert np.allclose(project.load_cells(1)["x_px"], cells_before["x_px"])
    assert before.ap_um == 7000.0


def test_user_flip_stacks_with_hemispheres_mirror(tmp_path, atlas):
    folder, project = _project(tmp_path)
    img0 = project.load_image(2)
    (folder / "hemispheres.csv").write_text("slice,x1,x2\nc1_r1,L,R\nc1_r2,R,L\n")
    project.sync_folder()
    assert 2 in project.apply_hemispheres(atlas)
    assert np.array_equal(project.load_image(2), img0[:, ::-1])

    project.flip_section_horizontal(2, mirror_alignment=True)
    assert project.get_slice(2).get("user_flipped") is True
    assert project.get_slice(2).get("mirrored") is False
    assert np.array_equal(project.load_image(2), img0)
    assert project.apply_hemispheres(atlas) == []
