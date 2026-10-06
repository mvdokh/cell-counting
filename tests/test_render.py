import numpy as np

from slicereg.render import (VIEWS, parse_color, region_target, resolve_regions, split_color,
                             split_region_text, suggest_regions, view_camera)

STRUCTURES = [
    {"acronym": "IRN", "name": "Intermediate reticular nucleus"},
    {"acronym": "PARN", "name": "Parvicellular reticular nucleus"},
    {"acronym": "XII", "name": "Hypoglossal nucleus"},
    {"acronym": "V", "name": "Motor nucleus of trigeminal"},
    {"acronym": "P5", "name": "Peritrigeminal zone"},
    {"acronym": "PERI5", "name": "Perirhinal area, layer 5"},
    {"acronym": "Acs5", "name": "Accessory trigeminal nucleus"},
    {"acronym": "VII", "name": "Facial motor nucleus"},
]


def test_split_region_text():
    assert split_region_text("irt/pcrt xii, mo5;peri5  ") == ["irt", "pcrt", "xii", "mo5",
                                                              "peri5"]


def test_paxinos_names_resolve_to_allen():
    found, missing = resolve_regions(split_region_text("irt/pcrt xii mo5 peri5 acc5 fmn"),
                                     STRUCTURES)
    assert [acr for _, acr, _ in found] == ["IRN", "PARN", "XII", "V", "P5", "Acs5", "VII"]
    assert missing == []


def test_allen_acronyms_case_insensitive_and_deduplicated():
    found, missing = resolve_regions(["irn", "IRt", "vii", "nope"], STRUCTURES)
    assert [acr for _, acr, _ in found] == ["IRN", "VII"]
    assert missing == ["nope"]


def test_suggestions():
    assert "IRN" in suggest_regions("reticular", STRUCTURES)


def test_structure_colours():
    assert split_region_text("IRt:red, PCRt=#3080ff xii") == ["IRt:red", "PCRt=#3080ff", "xii"]
    assert split_color("IRt:red") == ("IRt", "red")
    assert split_color("PCRt=#3080ff") == ("PCRt", "#3080ff")
    assert split_color("xii") == ("xii", None)
    found, missing = resolve_regions(["IRt:red", "pcrt=#3080ff", "nope:blue"], STRUCTURES)
    assert [(q, acr) for q, acr, _ in found] == [("IRt:red", "IRN"), ("pcrt=#3080ff", "PARN")]
    assert missing == ["nope:blue"]


def test_parse_color():
    assert parse_color("#ff0000") == (1.0, 0.0, 0.0)
    assert parse_color("00ff00") == (0.0, 1.0, 0.0)
    assert parse_color("black") == (0.0, 0.0, 0.0)
    assert parse_color("steelblue") is not None
    assert parse_color("notacolor") is None


class _Mesh:
    def bounds(self):
        return [0, 13200, 0, 8000, -11400, 0]


class _Scene:
    class root:
        _mesh = _Mesh()


def test_view_camera_frames_the_brain_from_each_side():
    centre = np.array([6600, 4000, -5700])
    for view, (direction, up) in VIEWS.items():
        cam = view_camera(_Scene, view)
        assert np.allclose(cam["focal_point"], centre)
        offset = np.array(cam["pos"]) - centre
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        assert np.allclose(offset / np.linalg.norm(offset), d)
        assert abs(np.dot(cam["viewup"], d)) < 1e-9
        near, far = cam["clipping_range"]
        assert near < np.linalg.norm(offset) < far
    # the brain is longest along AP, so side views need more distance than front views
    side = np.linalg.norm(np.array(view_camera(_Scene, "left")["pos"]) - centre)
    front = np.linalg.norm(np.array(view_camera(_Scene, "front")["pos"]) - centre)
    assert side > front


class _Atlas:
    atlas_name = "fake"
    resolution = (100.0, 100.0, 100.0)
    structures = {"IRN": {"id": 7}, "PARN": {"id": 8}}
    annotation = np.zeros((20, 10, 12), np.int32)
    annotation[5:15, 6:8, 2:4] = 7
    annotation[5:12, 6:8, 8:10] = 8

    def get_structure_descendants(self, acronym):
        return []


def test_target_views_look_at_the_caudal_tip():
    tip, bounds = region_target(_Atlas(), tip_um=0.0)
    assert np.allclose(tip, [1400, 650, -250])          # last IRN slab, ML flipped to -z
    assert np.allclose(bounds, [[500, 1400], [600, 700], [-900, -200]])

    class Scene(_Scene):
        atlas = _Atlas()

    whole, zoomed = view_camera(Scene, "back_low"), view_camera(Scene, "back_low_zoom")
    offset = np.array(whole["pos"]) - whole["focal_point"]
    d = offset / np.linalg.norm(offset)
    shift = np.array(whole["focal_point"]) - region_target(Scene.atlas)[0]
    assert abs(shift @ d) < 1e-6                         # only slid sideways off the tip
    assert offset[0] > 0 and offset[1] > 0               # behind and below, looking up
    assert np.linalg.norm(np.array(zoomed["pos"]) - zoomed["focal_point"]) < \
        np.linalg.norm(offset)
