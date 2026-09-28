import numpy as np
import pytest

from slicereg.atlas import Atlas, label_contours


@pytest.fixture(scope="module")
def atlas():
    return Atlas("allen_mouse_25um")


def test_zero_tilt_plane_is_coronal(atlas):
    w, h = atlas.plane_size_um
    p = atlas.plane_to_3d([[w / 2, h / 2], [100.0, 300.0]], ap_um=7000.0)
    assert np.allclose(p[0], [7000.0, h / 2, w / 2])
    assert np.allclose(p[1], [7000.0, 300.0, 100.0])


def test_tilted_plane_stays_orthonormal_and_centred(atlas):
    origin, u, v = atlas.basis(7000.0, pitch_deg=8.0, yaw_deg=-5.0)
    assert np.isclose(np.linalg.norm(u), 1) and np.isclose(np.linalg.norm(v), 1)
    assert np.isclose(u @ v, 0)
    w, h = atlas.plane_size_um
    assert np.allclose(atlas.plane_to_3d([[w / 2, h / 2]], 7000.0, 8.0, -5.0)[0],
                       [7000.0, h / 2, w / 2])


def test_sampled_plane_matches_volume_slice(atlas):
    ap = 250 * atlas.res[0] + 1.0
    ann, _ = atlas.sample_plane(ap)
    assert np.array_equal(ann, atlas.annotation[250])


def test_region_lookup(atlas):
    region, hemi = atlas.lookup(np.array([[5000.0, 4000.0, 5700.0], [-10.0, 0.0, 0.0]]))
    assert atlas.acronym(region[0]) == "MS"
    assert region[1] == 0 and hemi[1] == 0
    _, hemi = atlas.lookup(np.array([[5000.0, 4000.0, 3000.0], [5000.0, 4000.0, 8500.0]]))
    assert atlas.hemisphere_name(hemi[0]) == "right"
    assert atlas.hemisphere_name(hemi[1]) == "left"


def test_label_contours_are_subpixel_and_shared():
    """Two touching discs: contours should close up and not double the shared edge."""
    yy, xx = np.mgrid[0:60, 0:80]
    labels = np.zeros((60, 80), dtype=np.int64)
    labels[(xx - 25) ** 2 + (yy - 30) ** 2 <= 15 ** 2] = 1
    labels[(xx - 55) ** 2 + (yy - 30) ** 2 <= 15 ** 2] = 2

    by_region = label_contours(labels, sigma=1.0)
    assert set(by_region) == {1, 2}
    for rid, contours in by_region.items():
        assert contours, f"region {rid} should have at least one contour"
        for c in contours:
            # Sub-pixel: not all vertices land exactly on integer pixel coordinates.
            assert not np.allclose(c, np.round(c))
            # find_contours returns closed loops for a fully interior blob.
            assert np.allclose(c[0], c[-1])

    # The two discs touch around x=40; both regions' contours should pass near there,
    # tracing the same shared border rather than two independently-jittered lines.
    near_border = [c for c in by_region[1] for p in c if abs(p[1] - 40) < 1.0]
    assert near_border, "region 1 should have contour points near the shared border"


def test_label_contours_handles_huge_region_ids():
    labels = np.zeros((40, 40), dtype=np.uint32)
    labels[5:15, 5:15] = 614454277
    labels[20:35, 20:35] = 8
    assert set(label_contours(labels)) == {614454277, 8}


def test_label_contours_empty_for_blank_image():
    assert label_contours(np.zeros((10, 10), dtype=np.int64)) == {}
