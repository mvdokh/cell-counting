import numpy as np
import pytest

from slicereg.atlas import Atlas


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
