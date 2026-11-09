"""Numbered slide folders under one experiment are chained for plots and render."""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from slicereg.experiment import (collect_experiment_cells, experiment_positions,
                                 experiment_root, experiment_sections, is_experiment_dir,
                                 list_slides, open_experiment, sibling_slides, slide_number)
from slicereg.io import Project


def _folder_project(tmp_path: Path, slide: int, sections: list[tuple[int, int]],
                    *, ap_base: float = 7000.0):
    """Create image folder ``slide`` with MIP-like names and a project with fake alignments."""
    folder = tmp_path / str(slide)
    folder.mkdir()
    for col, row in sections:
        img = np.zeros((20, 30, 3), np.uint8)
        img[5:15, 5:25] = 80
        tifffile.imwrite(folder / f"c{col}r{row}_mip.lsm", np.moveaxis(img, -1, 0))
    (folder / "slicereg_folder.json").write_text(
        '{\n  "pattern": "^c(?P<col>\\\\d+)r(?P<row>\\\\d+)_mip\\\\.lsm$",\n'
        '  "channels": [{"name": "gfp", "color": "green"},'
        '{"name": "rfp", "color": "red"},{"name": "nissl", "color": "blue"}],\n'
        '  "align_channel": "nissl", "auto_crop": false, "atlas": "allen_mouse_25um",\n'
        '  "section_spacing_um": 80.0\n}\n',
        encoding="utf-8")
    proj = Project.open_or_create(folder)
    for i, s in enumerate(proj.slices):
        # minimal written image so status looks cropped
        d = proj.slice_dir(s["id"])
        d.mkdir(parents=True, exist_ok=True)
        img = np.zeros((10, 10, 3), np.uint8)
        img[:, :3] = 40   # left edge mark so horizontal flips are detectable
        img[2:8, 2:8] = 80
        tifffile.imwrite(d / "image.tif", img)
        tifffile.imwrite(d / "preview.tif", img)
        import cv2
        cv2.imwrite(str(d / "mask.png"), np.ones((10, 10), np.uint8) * 255)
        s["size"] = [10, 10]
        proj.save_alignment(s["id"], {
            "ap_um": ap_base + 80 * i, "pitch_deg": 1.0, "yaw_deg": 0.0,
            "rotation_deg": 0.0, "flip": False,
            "scale_x": 0.1, "scale_y": 0.1, "tx": 5.0, "ty": 5.0,
            "plane_size_um": [1000.0, 800.0], "plane_centre_um": [0.0, 0.0],
            "landmarks": [], "sigma": 1.0, "atlas": "allen_mouse_25um",
        })
        cells = pd.DataFrame({
            "slice_id": [s["id"]], "cell_type": ["green"], "x_px": [5], "y_px": [5],
            "ap_um": [ap_base], "dv_um": [0], "ml_um": [0],
            "region_id": [1], "region_acronym": ["IRN"], "region_name": ["IRt"],
            "hemisphere": ["right"],
        })
        proj.save_cells(s["id"], cells)
    proj.save()
    return proj


def test_list_slides_and_section_order_across_experiment(tmp_path):
    p7 = _folder_project(tmp_path, 7, [(5, 2), (5, 3)], ap_base=8000)
    p8 = _folder_project(tmp_path, 8, [(1, 1), (1, 2)], ap_base=8160)
    assert experiment_root(p7) == tmp_path.resolve()
    assert slide_number(p7) == 7
    slides = list_slides(tmp_path)
    assert [s["slide"] for s in slides] == [7, 8]
    assert sibling_slides(p7)[0]["slide"] == 7

    secs = experiment_sections(p7)
    assert [s["label"] for s in secs] == ["7/c5_r2", "7/c5_r3", "8/c1_r1", "8/c1_r2"]
    # last of 7 then first of 8
    assert secs[1]["name"] == "c5_r3" and secs[2]["name"] == "c1_r1"

    labels, refs, ap, pitch = experiment_positions(p7)
    assert labels == ["7/c5_r2", "7/c5_r3", "8/c1_r1", "8/c1_r2"]
    assert list(ap) == [8000.0, 8080.0, 8160.0, 8240.0]
    assert refs[2]["project"].root == p8.root

    cells = collect_experiment_cells(p7)
    assert len(cells) == 4
    assert set(cells["slide"]) == {7, 8}
    assert list(cells["region_acronym"]) == ["IRN"] * 4


def test_open_experiment_folder_opens_first_slide(tmp_path):
    _folder_project(tmp_path, 6, [(1, 1)], ap_base=5000)
    _folder_project(tmp_path, 7, [(1, 1)], ap_base=5080)
    (tmp_path / "5").mkdir()  # empty slide folder is skipped
    assert is_experiment_dir(tmp_path)
    proj = open_experiment(tmp_path)
    assert slide_number(proj) == 6
    again = Project.open_or_create(tmp_path)
    assert again.root == proj.root


def test_cell_types_carry_across_experiment_slides(tmp_path):
    p6 = _folder_project(tmp_path, 6, [(1, 1)], ap_base=5000)
    p7 = _folder_project(tmp_path, 7, [(1, 1)], ap_base=5080)
    p6.add_cell_type("green")
    p6.add_cell_type("red")
    p7 = Project.load(p7.root)
    assert "green" in p7.cell_types() and "red" in p7.cell_types()
    assert (tmp_path / "slicereg_experiment.json").is_file()


def test_flip_manifest_lists_sections_per_animal(tmp_path):
    import json

    from slicereg.experiment import (collect_flipped_labels, save_experiment_flips,
                                    sync_experiment_flips)

    p7 = _folder_project(tmp_path, 7, [(5, 2), (5, 3)], ap_base=8000)
    _folder_project(tmp_path, 8, [(1, 1)], ap_base=8160)
    sid = p7.slices[0]["id"]
    before = p7.load_image(sid)
    p7.flip_section_horizontal(sid, mirror_alignment=False)
    assert np.array_equal(p7.load_image(sid), before[:, ::-1])
    assert p7.get_slice(sid).get("user_flipped") is True
    cfg = json.loads((tmp_path / "slicereg_experiment.json").read_text(encoding="utf-8"))
    assert "7/c5_r2" in cfg["flipped"]
    assert collect_flipped_labels(p7) == ["7/c5_r2"]
    # Flag cleared in project.json is restored from the animal manifest; disk already
    # matches, so apply_hemispheres is a no-op.
    p7.get_slice(sid)["user_flipped"] = False
    p7.save()
    sync_experiment_flips(p7)
    assert p7.get_slice(sid).get("user_flipped") is True
    assert p7.apply_hemispheres() == []
    assert np.array_equal(p7.load_image(sid), before[:, ::-1])

    # A manifest-only entry restores the flag on another section (file flip itself is
    # covered in test_hemispheres; apply_hemispheres needs a full alignment to remap).
    other = p7.slices[1]["id"]
    save_experiment_flips(tmp_path, ["7/c5_r2", "7/c5_r3"])
    sync_experiment_flips(p7)
    assert p7.get_slice(other).get("user_flipped") is True
    assert Project._want_mirrored(p7.get_slice(other)) is True
    assert collect_flipped_labels(p7) == ["7/c5_r2", "7/c5_r3"]
