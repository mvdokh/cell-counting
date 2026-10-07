import numpy as np

from slicereg.app.layers import full_range, get_levels, store_levels


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
    assert get_levels(p, 2, np.uint8)[0]["max"] == 120.0      # the slide overview's levels
