"""Initial alignment of a section with DeepSlice (https://github.com/PolarBean/DeepSlice).

DeepSlice needs TensorFlow and pins old dependencies, so it lives in its own virtual
environment (``slicereg deepslice-setup``) and is run as a subprocess on
``deepslice_worker.py``.

DeepSlice predicts a QuickNII "anchoring" ``(o, u, v)`` in Allen CCFv3 25 um voxels:
image pixel ``(x, y)`` of a ``W x H`` image lies at ``o + x/W * u + y/H * v``.
QuickNII axes are x = left -> right, y = posterior -> anterior, z = inferior ->
superior, i.e. every axis is reversed relative to BrainGlobe ASR.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .atlas import rotation
from .transform import Alignment, SliceTransform

VOXEL_UM = 25.0
CCF_SHAPE_ASR = np.array([528, 320, 456])
ANCHOR_KEYS = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")
ENV_DIR = Path(__file__).resolve().parent.parent / ".venv-deepslice"
WORKER = Path(__file__).resolve().parent / "deepslice_worker.py"
MAX_TILT_DEG = 30.0


def supports_atlas(name: str) -> bool:
    return name.startswith("allen_mouse")


def python_path() -> Path:
    override = os.environ.get("SLICEREG_DEEPSLICE_PYTHON")
    if override:
        return Path(override)
    exe = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    return ENV_DIR / exe


def is_installed() -> bool:
    return python_path().exists()


def setup(env_dir: Path = ENV_DIR) -> None:
    """Create the DeepSlice environment and download the model weights."""
    if not python_path().exists():
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
    py = str(python_path())
    subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([py, "-m", "pip", "install", "DeepSlice", "tensorflow"], check=True)
    subprocess.run([py, str(WORKER), "--download"], check=True)


def worker_command(image_png: Path, out_json: Path, ensemble: bool = True) -> list[str]:
    cmd = [str(python_path()), "-u", str(WORKER), str(image_png), str(out_json)]
    return cmd if ensemble else cmd + ["--fast"]


def input_image(preview: np.ndarray, invert: bool = False) -> np.ndarray:
    """8-bit grey image for DeepSlice: channels contrast-stretched, then max-projected."""
    img = np.asarray(preview, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    grey = np.zeros(img.shape[:2], np.float32)
    for c in range(img.shape[2]):
        ch = img[..., c]
        lo, hi = np.percentile(ch, [0.5, 99.9])
        grey = np.maximum(grey, np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1))
    if invert:
        grey = 1 - grey
    return (grey * 255).astype(np.uint8)


def write_input_image(preview: np.ndarray, path: Path, invert: bool = False) -> None:
    cv2.imwrite(str(path), input_image(preview, invert))


def _quicknii_to_asr(q: np.ndarray) -> np.ndarray:
    return (CCF_SHAPE_ASR - np.asarray(q, dtype=float)[..., [1, 2, 0]]) * VOXEL_UM


def _asr_to_quicknii(p: np.ndarray) -> np.ndarray:
    q = CCF_SHAPE_ASR - np.asarray(p, dtype=float) / VOXEL_UM
    return q[..., [2, 0, 1]]


def _plane_frame(plane_size_um, ap_um: float, pitch_deg: float, yaw_deg: float):
    r = rotation(pitch_deg, yaw_deg)
    e_s, e_t, n = r[:, 2], r[:, 1], r[:, 0]
    w, h = plane_size_um
    origin = np.array([ap_um, h / 2, w / 2]) - (w / 2) * e_s - (h / 2) * e_t
    return origin, e_s, e_t, n


def alignment_to_anchoring(al: Alignment) -> dict:
    """QuickNII anchoring of an (unwarped) alignment; landmarks are ignored."""
    tf = SliceTransform(replace(al, landmarks=[]))
    origin, e_s, e_t, _ = _plane_frame(al.plane_size_um, al.ap_um, al.pitch_deg, al.yaw_deg)
    w, h = al.image_size
    st = tf.affine_inv(np.array([[0.0, 0.0], [w, 0.0], [0.0, h]]))
    p = origin + st[:, :1] * e_s + st[:, 1:2] * e_t
    q = _asr_to_quicknii(p)
    o, u, v = q[0], q[1] - q[0], q[2] - q[0]
    return dict(zip(ANCHOR_KEYS, (*o, *u, *v)))


def anchoring_to_alignment(anchoring: dict, atlas_name: str, plane_size_um,
                           image_size) -> Alignment:
    """Closest ``Alignment`` (plane + rotation/scale/flip/offset) to a QuickNII anchoring.

    Tilts beyond the +-30 deg the GUI allows are clipped; the section is then fitted
    to the projection of the DeepSlice plane onto the clipped one.
    """
    a = np.array([float(anchoring[k]) for k in ANCHOR_KEYS])
    o_q, u_q, v_q = a[:3], a[3:6], a[6:]
    w_img, h_img = (float(v) for v in image_size)
    o = _quicknii_to_asr(o_q)
    du = _quicknii_to_asr(o_q + u_q) - o
    dv = _quicknii_to_asr(o_q + v_q) - o
    ax, ay = du / w_img, dv / h_img  # atlas microns per image pixel along x and y

    n = np.cross(ax, ay)
    n /= np.linalg.norm(n)
    if n[0] < 0:
        n = -n
    pitch = float(np.clip(np.degrees(np.arcsin(-n[1])), -MAX_TILT_DEG, MAX_TILT_DEG))
    yaw = float(np.clip(np.degrees(np.arctan2(-n[2], n[0])), -MAX_TILT_DEG, MAX_TILT_DEG))

    w, h = plane_size_um
    ap = float(o[0] - (n[1] * (h / 2 - o[1]) + n[2] * (w / 2 - o[2])) / n[0])
    origin, e_s, e_t, _ = _plane_frame(plane_size_um, ap, pitch, yaw)

    # image pixel -> plane microns: p = A x + d
    proj = np.stack([e_s, e_t])
    A = proj @ np.stack([ax, ay], axis=1)
    d = proj @ (o - origin)
    m = np.linalg.inv(A)
    t = m @ (np.array([w, h]) / 2 - d)

    flip = bool(np.linalg.det(m) < 0)
    mm = m @ np.diag([-1.0, 1.0]) if flip else m
    theta = np.arctan2(mm[1, 0] - mm[0, 1], mm[0, 0] + mm[1, 1])
    c, s = np.cos(theta), np.sin(theta)
    scale_x = float(c * mm[0, 0] + s * mm[1, 0])
    scale_y = float(-s * mm[0, 1] + c * mm[1, 1])
    return Alignment(atlas=atlas_name, plane_size_um=tuple(plane_size_um),
                     image_size=tuple(int(v) for v in image_size), ap_um=ap,
                     pitch_deg=pitch, yaw_deg=yaw, rotation_deg=float(np.degrees(theta)),
                     scale_x=scale_x, scale_y=scale_y, tx=float(t[0]), ty=float(t[1]),
                     flip=flip)
