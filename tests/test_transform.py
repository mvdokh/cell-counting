import numpy as np

from slicereg.transform import Alignment, SliceTransform


def make_alignment(landmarks=()):
    return Alignment(atlas="test", plane_size_um=(11400.0, 8000.0), image_size=(4300, 2500),
                     ap_um=9000.0, rotation_deg=7.0, scale_x=0.41, scale_y=0.37,
                     tx=2100.0, ty=1300.0, flip=True, landmarks=list(landmarks))


def random_image_points(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([0, 0], [4300, 2500], size=(n, 2))


def test_affine_round_trip():
    tf = SliceTransform(make_alignment())
    x = random_image_points()
    assert np.abs(tf.plane_to_image(tf.image_to_plane(x)) - x).max() < 1e-6


def test_warp_round_trip_and_landmarks_hit():
    base = SliceTransform(make_alignment())
    src_img = np.array([[1000, 800], [3000, 900], [2200, 2000], [1500, 1500]], float)
    dst_img = src_img + np.array([[40, -25], [-30, 20], [15, 35], [-20, -10]])
    src_plane = base.image_to_plane(src_img)
    lm = np.hstack([src_plane, dst_img]).tolist()
    tf = SliceTransform(make_alignment(lm))

    assert np.abs(tf.plane_to_image(src_plane) - dst_img).max() < 1e-3
    x = random_image_points()
    assert np.abs(tf.plane_to_image(tf.image_to_plane(x)) - x).max() < 1.0


def test_single_landmark_is_supported():
    base = SliceTransform(make_alignment())
    p = base.image_to_plane([[2000, 1200]])
    tf = SliceTransform(make_alignment([[*p[0], 2030, 1190]]))
    assert np.abs(tf.plane_to_image(p) - [[2030, 1190]]).max() < 1e-3


def test_alignment_serialisation():
    al = make_alignment([[1, 2, 3, 4]])
    assert Alignment.from_dict(al.to_dict()) == al
