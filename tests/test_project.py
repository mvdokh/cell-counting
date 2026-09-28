import numpy as np
import pandas as pd
import pytest
import tifffile

from slicereg.atlas import Atlas
from slicereg.detect import detect_sections
from slicereg.export import export_project, map_cells
from slicereg.io import Project
from slicereg.transform import Alignment, initial_alignment


@pytest.fixture(scope="module")
def atlas():
    return Atlas("allen_mouse_25um")


@pytest.fixture
def project(tmp_path):
    slide = np.full((1600, 3200, 3), 8, np.uint8)
    slide[200:1400, 200:1500] = 120
    slide[200:1400, 1700:3000] = 120
    path = tmp_path / "slide.tif"
    tifffile.imwrite(path, slide, photometric="rgb")
    return Project.open_or_create(path)


def test_detect_crop_align_count_move_export(project, atlas):
    thumb = project.thumbnail()
    boxes, thr, dark = detect_sections(thumb)
    assert len(boxes) == 2
    project.data.update(threshold=thr, dark_background=dark)
    f = project.data["thumb_factor"]
    to_crop = project.set_boxes([(None, [v * f for v in b]) for b in boxes])
    assert to_crop == [1, 2]
    for sid in to_crop:
        project.crop_slice(sid)
    assert project.status(1)["cropped"] and project.load_mask(1).mean() > 0.5

    s = project.get_slice(1)
    size = (s["bbox"][2] - s["bbox"][0], s["bbox"][3] - s["bbox"][1])
    al = initial_alignment(atlas, size, [0, 0, *size], ap_um=7000.0)
    project.save_alignment(1, al.to_dict())
    cells = map_cells(atlas, al, [[size[0] / 2, size[1] / 2]], ["cell"], 1)
    project.save_cells(1, cells)
    assert cells.loc[0, "region_acronym"] != "outside"

    moved = [s["bbox"][0] - 32, s["bbox"][1] - 16, s["bbox"][2], s["bbox"][3]]
    assert project.set_boxes([(1, moved), (2, project.get_slice(2)["bbox"])]) == [1]
    project.crop_slice(1)
    al2 = Alignment.from_dict(project.load_alignment(1))
    after = project.load_cells(1)
    assert after.loc[0, "x_px"] == pytest.approx(cells.loc[0, "x_px"] + 32)
    remapped = map_cells(atlas, al2, after[["x_px", "y_px"]].to_numpy(), ["cell"], 1)
    assert np.allclose(remapped[["ap_um", "dv_um", "ml_um"]], cells[["ap_um", "dv_um", "ml_um"]])

    export_project(project)
    all_cells = pd.read_csv(project.root / "cells_all.csv")
    assert len(all_cells) == 1
    assert (project.root / "region_counts.csv").exists()
