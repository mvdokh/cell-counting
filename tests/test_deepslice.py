import cv2
import numpy as np
import pytest

from slicereg.deepslice import (ANCHOR_KEYS, _asr_to_quicknii, _quicknii_to_asr,
                                alignment_to_anchoring, anchoring_to_alignment, input_image,
                                uncrop_anchoring, write_input_image)
from slicereg.transform import Alignment, SliceTransform

PLANE = (11400.0, 8000.0)
SIZE = (4448, 2576)
FIELDS = ("ap_um", "pitch_deg", "yaw_deg", "rotation_deg", "scale_x", "scale_y", "tx", "ty")


def make(**kw):
    base = dict(atlas="allen_mouse_25um", plane_size_um=PLANE, image_size=SIZE, ap_um=10950.0,
                pitch_deg=-2.5, yaw_deg=1.5, rotation_deg=-3.5, scale_x=0.46, scale_y=0.35,
                tx=2248.6, ty=1315.1, flip=False)
    return Alignment(**{**base, **kw})


def test_quicknii_axes_are_reversed_asr():
    # QuickNII origin is the left/posterior/inferior corner = ASR (13200, 8000, 11400).
    assert np.allclose(_quicknii_to_asr([0, 0, 0]), [13200, 8000, 11400])
    assert np.allclose(_quicknii_to_asr([456, 528, 320]), [0, 0, 0])
    p = np.array([[5000.0, 3000.0, 2000.0], [100.0, 7000.0, 11000.0]])
    assert np.allclose(_quicknii_to_asr(_asr_to_quicknii(p)), p)


@pytest.mark.parametrize("kw", [{}, {"flip": True}, {"rotation_deg": 150.0, "yaw_deg": -12.0},
                                {"pitch_deg": 20.0, "scale_y": 0.5, "flip": True}])
def test_anchoring_round_trip(kw):
    al = make(**kw)
    back = anchoring_to_alignment(alignment_to_anchoring(al), al.atlas, PLANE, SIZE)
    assert back.flip == al.flip
    for f in FIELDS:
        assert getattr(back, f) == pytest.approx(getattr(al, f), abs=1e-6), f


def test_steep_tilt_is_clipped_and_stays_close():
    al = make(pitch_deg=34.0, tx=SIZE[0] / 2, ty=SIZE[1] / 2)
    back = anchoring_to_alignment(alignment_to_anchoring(al), al.atlas, PLANE, SIZE)
    assert back.pitch_deg == pytest.approx(30.0)
    centre = np.array([[SIZE[0] / 2, SIZE[1] / 2]])
    p0 = to_3d(al, SliceTransform(al).image_to_plane(centre))
    p1 = to_3d(back, SliceTransform(back).image_to_plane(centre))
    assert np.linalg.norm(p0 - p1) < 50.0


def to_3d(al, st):
    from slicereg.deepslice import _plane_frame

    origin, e_s, e_t, _ = _plane_frame(PLANE, al.ap_um, al.pitch_deg, al.yaw_deg)
    return origin + st[:, :1] * e_s + st[:, 1:2] * e_t


def test_uncrop_anchoring_recovers_full_image_anchoring():
    full = alignment_to_anchoring(make())
    o = np.array([full[k] for k in ("ox", "oy", "oz")])
    u = np.array([full[k] for k in ("ux", "uy", "uz")])
    v = np.array([full[k] for k in ("vx", "vy", "vz")])
    x0, y0, x1, y1 = 400.0, 250.0, 3900.0, 2300.0
    w, h = SIZE
    o_c = o + x0 / w * u + y0 / h * v
    crop = dict(zip(ANCHOR_KEYS, (*o_c, *(u * (x1 - x0) / w), *(v * (y1 - y0) / h))))
    back = uncrop_anchoring(crop, (x0, y0, x1, y1), SIZE)
    assert all(back[k] == pytest.approx(full[k]) for k in ANCHOR_KEYS)


def test_write_input_image_blanks_and_crops_to_mask(tmp_path):
    img = np.full((100, 200, 1), 50, np.uint8)
    mask = np.zeros((100, 200), bool)
    mask[20:60, 50:150] = True
    box = write_input_image(img, tmp_path / "in.png", mask=mask, pad=0.0)
    assert box == (50, 20, 150, 60)
    assert cv2.imread(str(tmp_path / "in.png"), cv2.IMREAD_GRAYSCALE).shape == (40, 100)


def test_input_image_is_uint8_and_inverts():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 4000, size=(50, 80, 2)).astype(np.uint16)
    g = input_image(img)
    assert g.dtype == np.uint8 and g.shape == (50, 80)
    assert np.abs(input_image(img, invert=True).astype(int) - (255 - g)).max() <= 1
