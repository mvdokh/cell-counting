import numpy as np

from slicereg.app.layers import equalize_channels, full_range, get_levels, store_levels


class _Project:
    def __init__(self):
        self.data = {"slices": [{"id": 1}, {"id": 2}]}
        self.saved = 0

    def get_slice(self, sid):
        return next(s for s in self.data["slices"] if s["id"] == sid)

    def save(self):
        self.saved += 1


def test_levels_default_to_full_range_without_auto_adjusting():
    assert full_range(np.uint8) == (0.0, 255.0)
    assert full_range(np.uint16) == (0.0, 65535.0)
    p = _Project()
    assert get_levels(p, 2, np.uint8, sid=1) == [{"min": 0.0, "max": 255.0, "gamma": 1.0}] * 2


def test_shared_levels_apply_to_every_section_unless_it_has_its_own():
    p = _Project()
    store_levels(p, [{"min": 5, "max": 120, "gamma": 0.8}, {"min": 0, "max": 90, "gamma": 1}])
    assert get_levels(p, 2, np.uint8, sid=1)[0] == {"min": 5.0, "max": 120.0, "gamma": 0.8}
    assert get_levels(p, 2, np.uint8) == get_levels(p, 2, np.uint8, sid=2)
    store_levels(p, [{"min": 0, "max": 60, "gamma": 1}, {"min": 0, "max": 90, "gamma": 1}], sid=2)
    assert get_levels(p, 2, np.uint8, sid=2)[0]["max"] == 60.0
    assert get_levels(p, 2, np.uint8, sid=1)[0]["max"] == 120.0
    assert get_levels(p, 2, np.uint8)[0]["max"] == 120.0


def test_overview_levels_are_independent_of_section_levels():
    p = _Project()
    store_levels(p, [{"min": 0, "max": 50, "gamma": 1}] * 2)
    store_levels(p, [{"min": 10, "max": 200, "gamma": 0.9}] * 2, key="overview_levels")
    assert get_levels(p, 2, np.uint8)[0]["max"] == 50.0
    assert get_levels(p, 2, np.uint8, key="overview_levels")[0]["max"] == 200.0


def test_equalize_channels_matches_dim_and_bright_sections():
    yy, xx = np.mgrid[0:40, 0:40]
    ramp = (xx + yy) % 20
    dim = np.zeros((40, 40, 2), np.uint8)
    bright = np.zeros((40, 40, 2), np.uint8)
    dim[..., 0] = 10 + ramp
    dim[..., 1] = 15 + ramp
    bright[..., 0] = 100 + 5 * ramp
    bright[..., 1] = 120 + 5 * ramp
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    eq_dim = equalize_channels(dim, mask)
    eq_bright = equalize_channels(bright, mask)
    # After stretching, dim and bright tissue share the same display range.
    assert np.percentile(eq_dim[..., 0][mask], 99) > 240
    assert np.percentile(eq_bright[..., 0][mask], 99) > 240
    assert abs(np.median(eq_dim[..., 0][mask]) - np.median(eq_bright[..., 0][mask])) < 25
    assert eq_dim[0, 0, 0] == 0
