"""3D view of aligned sections and counted cells in brainrender."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .export import CELL_COLORS, NOT_REGIONS, collect_cells
from .transform import Alignment, SliceTransform

# Common Paxinos & Franklin abbreviations -> Allen CCF acronyms. Checked before the
# Allen acronyms themselves, so e.g. "peri5" means peritrigeminal zone (Allen P5),
# not perirhinal layer 5 (Allen PERI5).
PAXINOS_ALIASES = {
    "irt": "IRN",
    "pcrt": "PARN",
    "12n": "XII",
    "mo5": "V",
    "5n": "V",
    "pe5": "P5",
    "peri5": "P5",
    "acs5": "Acs5",
    "acc5": "Acs5",
    "7n": "VII",
    "fmn": "VII",
}

HEMISPHERES = ("both", "left", "right")
STYLES = ("plastic", "shiny", "glossy", "metallic", "cartoon")

# Camera direction (from the brain's centre towards the camera) and screen-up vector,
# in brainrender's rendered axes: x = posterior, y = ventral, z = right.
VIEWS = {
    "three_quarter": ((-0.74, -0.32, 0.59), (0, -1, 0)),
    "front": ((-1, 0, 0), (0, -1, 0)),
    "back": ((1, 0, 0), (0, -1, 0)),
    "left": ((0, 0, -1), (0, -1, 0)),
    "right": ((0, 0, 1), (0, -1, 0)),
    "top": ((0, -1, 0), (-1, 0, 0)),
    "bottom": ((0, 1, 0), (-1, 0, 0)),
    "back_low": ((1, 0.21, 0), (0, -1, 0)),
    "back_low_zoom": ((1, 0.21, 0), (0, -1, 0)),
}
VIEW_LABELS = {
    "three_quarter": "3/4 view",
    "front": "Front (coronal)",
    "back": "Back (coronal)",
    "left": "Left (sagittal)",
    "right": "Right (sagittal)",
    "top": "Top (dorsal)",
    "bottom": "Bottom (ventral)",
    "back_low": "Back, low (IRt/PCRt)",
    "back_low_zoom": "Back, low, zoomed",
}
# Views aimed at the caudal end of these regions (Allen IRt, PCRt) rather than the
# brain's centre; True = frame the regions instead of the whole brain.
TARGET_REGIONS = ("IRN", "PARN")
TARGET_VIEWS = {"back_low": False, "back_low_zoom": True}
TARGET_TIP_UM = 300.0
DEFAULT_CHANNEL_COLORS = {1: ["gray"], 2: ["green", "magenta"], 3: ["red", "green", "blue"]}
WINDOW_SIZE = (1600, 1200)
SCREENSHOT_SCALE = 2


def split_region_text(text: str) -> list[str]:
    """'IRt/PCRt, XII:red mo5' -> ['IRt', 'PCRt', 'XII:red', 'mo5']."""
    return [t for t in re.split(r"[\s,;/]+", text or "") if t]


def split_color(token: str) -> tuple[str, str | None]:
    """'IRt:red' or 'IRt=#ff0000' -> ('IRt', 'red'); 'IRt' -> ('IRt', None)."""
    m = re.match(r"^([^:=]+)[:=](.+)$", token.strip())
    return (m.group(1), m.group(2)) if m else (token.strip(), None)


def parse_color(text: str) -> tuple[float, float, float] | None:
    """A colour name ('red', 'steelblue', 'blue5') or hex code as RGB 0-1, or None."""
    from vedo.colors import get_color

    t = text.strip()
    if re.fullmatch(r"[0-9a-fA-F]{6}", t):
        t = "#" + t
    rgb = tuple(float(v) for v in get_color(t))
    # vedo quietly turns unknown names into black
    if rgb == (0.0, 0.0, 0.0) and t.lower() not in ("black", "k", "#000000", "#000"):
        return None
    return rgb


def resolve_regions(queries, structures) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Match region names to atlas structures, case-insensitively.

    Returns ``([(query, acronym, full name), ...], [unmatched queries])`` with
    duplicate acronyms dropped. A ``:colour`` suffix on a query is ignored here.
    """
    by_acronym = {s["acronym"].lower(): s for s in structures}
    found, missing, seen = [], [], set()
    for q in queries:
        key = split_color(q)[0].strip().lower()
        if not key:
            continue
        alias = PAXINOS_ALIASES.get(key)
        s = by_acronym.get(alias.lower()) if alias else None
        s = s or by_acronym.get(key)
        if s is None:
            missing.append(q)
        elif s["acronym"] not in seen:
            seen.add(s["acronym"])
            found.append((q, s["acronym"], s["name"]))
    return found, missing


def suggest_regions(query: str, structures, limit: int = 4) -> list[str]:
    """Acronyms whose acronym starts with, or whose name contains, ``query``."""
    q = split_color(query)[0].strip().lower()
    hits = [s["acronym"] for s in structures if s["acronym"].lower().startswith(q)]
    hits += [s["acronym"] for s in structures
             if q in s["name"].lower() and s["acronym"] not in hits]
    return hits[:limit]


def section_texture(project, sid: int) -> np.ndarray:
    """RGBA texture of a section: channels blended in their configured colours (as in
    the 2D view), stretched within the tissue, transparent outside it."""
    img = project.load_preview(sid).astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]
    mask = project.load_mask(sid).astype(bool)
    if mask.shape != img.shape[:2] or not mask.any():
        mask = np.ones(img.shape[:2], bool)
    n = img.shape[2]
    config = project.data.get("channels") or []
    names = [ch.get("color", "gray") for ch in config] if len(config) == n else \
        DEFAULT_CHANNEL_COLORS.get(n, ["red", "green", "blue"] * n)
    rgb = np.zeros(img.shape[:2] + (3,), np.float32)
    for c in range(n):
        color = parse_color("white" if names[c] in ("gray", "grey") else names[c]) or (1, 1, 1)
        lo, hi = np.percentile(img[..., c][mask], [0.5, 99.7])
        rgb += np.clip((img[..., c] - lo) / max(hi - lo, 1e-6), 0, 1)[..., None] * color
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return np.concatenate([rgb, (mask * 255).astype(np.uint8)[..., None]], axis=2)


def slice_mesh(atlas, project, sid: int, grid: int = 24):
    """Section preview as a textured mesh placed (with its warp) in atlas space."""
    import vedo

    al = Alignment.from_dict(project.load_alignment(sid))
    tf = SliceTransform(al)
    w, h = al.image_size
    nx, ny = grid, max(2, int(round(grid * h / w)))
    u, v = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    xy = np.stack([u.ravel() * w, v.ravel() * h], 1)
    verts = atlas.plane_to_3d(tf.image_to_plane(xy), al.ap_um, al.pitch_deg, al.yaw_deg)
    faces = [[j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i]
             for j in range(ny - 1) for i in range(nx - 1)]
    mesh = vedo.Mesh([verts, faces])
    mesh.texture(section_texture(project, sid), tcoords=np.stack([u.ravel(), 1 - v.ravel()], 1))
    return mesh


_target_cache: dict = {}


def region_target(atlas, acronyms=TARGET_REGIONS,
                  tip_um: float = TARGET_TIP_UM) -> tuple[np.ndarray, np.ndarray] | None:
    """(caudal tip centre, (3, 2) bounds) of the regions, in brainrender's rendered
    axes (x = posterior, y = ventral, z = -ML). The tip is the centroid of the
    voxels within ``tip_um`` of the regions' most posterior point."""
    key = (atlas.atlas_name, tuple(acronyms), tip_um)
    if key not in _target_cache:
        ids = set()
        for acr in acronyms:
            try:  # StructuresDict looks up acronyms but `in` only sees ids
                ids.add(atlas.structures[acr]["id"])
                ids |= {atlas.structures[d]["id"] for d in atlas.get_structure_descendants(acr)}
            except KeyError:
                continue
        ann = np.asarray(atlas.annotation)
        idx = np.argwhere(np.isin(ann, list(ids))) if ids else np.empty((0, 3))
        if not len(idx):
            _target_cache[key] = None
        else:
            pts = idx * np.asarray(atlas.resolution, float) * [1, 1, -1]
            tip = pts[pts[:, 0] >= pts[:, 0].max() - tip_um].mean(axis=0)
            _target_cache[key] = (tip, np.stack([pts.min(axis=0), pts.max(axis=0)], 1))
    return _target_cache[key]


def view_camera(scene, view: str, aspect: float = WINDOW_SIZE[0] / WINDOW_SIZE[1],
                view_angle: float = 30.0, margin: float = 1.08) -> dict:
    """Camera looking at the whole brain from one of ``VIEWS``, zoomed to fit it.

    ``TARGET_VIEWS`` look at the caudal tip of ``TARGET_REGIONS`` instead of the
    brain's centre, framing either the whole brain or just those regions."""
    def corners(b):
        return np.array([[x, y, z] for x in b[0] for y in b[1] for z in b[2]])

    brain = np.asarray(scene.root._mesh.bounds(), float).reshape(3, 2)
    bounds, centre = brain, brain.mean(axis=1)
    target = None
    if view in TARGET_VIEWS and getattr(scene, "atlas", None) is not None:
        target = region_target(scene.atlas)
        if target is not None:
            centre = target[0]
            if TARGET_VIEWS[view]:
                bounds = target[1]
    direction, up = (np.asarray(v, float) for v in VIEWS[view])
    direction /= np.linalg.norm(direction)
    up = up - up.dot(direction) * direction
    up /= np.linalg.norm(up)
    right = np.cross(up, direction)

    def fit(focal):
        """Closest distance at which every corner of the box projects inside the frame."""
        rel = corners(bounds) - focal
        off_axis = np.maximum(np.abs(rel @ up), np.abs(rel @ right) / aspect) * margin
        return rel, (rel @ direction + off_axis / np.tan(np.radians(view_angle) / 2)).max()

    rel, distance = fit(centre)
    if target is not None:  # same angle, but slide sideways to centre the box in frame
        depth = distance - rel @ direction
        for axis in (up, right):
            a = (rel @ axis) / depth
            centre = centre + axis * (a.max() + a.min()) / 2 * distance
        rel, distance = fit(centre)
    radius = np.linalg.norm(corners(brain) - centre, axis=1).max()
    return {"pos": tuple(centre + direction * distance), "focal_point": tuple(centre),
            "viewup": tuple(up), "clipping_range": (max(1.0, distance - 2 * radius),
                                                    distance + 2 * radius)}


def build_scene(project, show_slices: bool = True, top_regions: int = 5,
                structures: list[str] | None = None, show_cells: bool = True,
                cell_types: list[str] | None = None, show_brain: bool = True,
                region_alpha: float = 0.3, hemisphere: str = "both", radius: float = 25,
                brain_alpha: float = 0.3, style: str = "plastic", show_axes: bool = False,
                show_inset: bool = True, show_title: bool = True, offscreen: bool = False):
    import brainrender
    from brainrender import Scene
    from brainrender.actors import Points

    from .atlas import Atlas

    brainrender.settings.OFFSCREEN = offscreen
    brainrender.settings.SHADER_STYLE = style
    brainrender.settings.ROOT_ALPHA = float(brain_alpha)
    brainrender.settings.SHOW_AXES = show_axes
    atlas_name = project.data["atlas"]
    scene = Scene(root=show_brain, atlas_name=atlas_name, check_latest=False,
                  inset=show_inset, title=Path(project.data["source"]).stem)
    if not show_title:
        set_title_visible(scene, False)
    cells = collect_cells(project)
    if cell_types:
        cells = cells[cells["cell_type"].isin(cell_types)]
    if show_cells:
        for i, (ctype, df) in enumerate(cells.groupby("cell_type", sort=False)):
            scene.add(Points(df[["ap_um", "dv_um", "ml_um"]].to_numpy(float),
                             name=str(ctype), colors=CELL_COLORS[i % len(CELL_COLORS)],
                             radius=radius))

    regions = []  # (acronym, colour or None for the atlas colour)
    if structures:
        found, missing = resolve_regions(structures, scene.atlas.structures_list)
        for query, acr, _ in found:
            color_text = split_color(query)[1]
            color = parse_color(color_text) if color_text else None
            if color_text and color is None:
                print(f"Unknown colour {color_text!r} for {acr}; using the atlas colour.")
            regions.append((acr, color))
        if missing:
            print(f"Unknown structures (skipped): {', '.join(missing)}")
    if top_regions and len(cells):
        inside = cells[~cells["region_acronym"].isin(NOT_REGIONS)]
        shown = {acr for acr, _ in regions}
        regions += [(a, None) for a in inside["region_acronym"].value_counts().index[:top_regions]
                    if a not in shown]
    default = [acr for acr, color in regions if color is None]
    if default:
        scene.add_brain_region(*default, alpha=region_alpha, hemisphere=hemisphere)
    for acr, color in regions:
        if color is not None:
            scene.add_brain_region(acr, alpha=region_alpha, color=color, hemisphere=hemisphere)

    if show_slices:
        atlas = Atlas(atlas_name)
        for s in project.slices:
            if project.status(s["id"])["aligned"]:
                scene.add(slice_mesh(atlas, project, s["id"]), names=f"slice_{s['id']:02d}",
                          classes="section")
    _restyle(scene)
    return scene


def title_actors(scene) -> list:
    return [a.mesh for a in scene.get_actors(br_class="title")]


def title_visible(scene) -> bool:
    return any(t.actor.GetVisibility() for t in title_actors(scene))


def set_title_visible(scene, visible: bool) -> None:
    for t in title_actors(scene):
        t.on() if visible else t.off()


def _restyle(scene) -> None:
    """After brainrender's shader style: section images unlit (true colours) and cells
    matte, without the specular highlight the brain's style gives them."""
    apply_style = scene._apply_style

    def styled():
        apply_style()
        for actor in scene.get_actors(br_class="section"):
            actor._mesh.lighting("off")
        for actor in scene.get_actors(br_class="Points"):
            actor._mesh.lighting(ambient=0.5, diffuse=0.6, specular=0.0)

    scene._apply_style = styled


class WindowControls:
    """Snap-to-view and save-image buttons inside the brainrender window."""

    def __init__(self, scene, save_dir: Path):
        self.scene = scene
        self.save_dir = Path(save_dir)
        plt = scene.plotter
        self.buttons = []
        self.title_button = None
        entries = [([VIEW_LABELS[v]], lambda *_, v=v: self.show_view(v)) for v in VIEWS]
        if title_actors(scene):
            entries.append((["Hide title", "Show title"] if title_visible(scene) else
                            ["Show title", "Hide title"], self.toggle_title))
        entries.append((["Save image"], lambda *_: self.save_image()))
        for i, (states, fn) in enumerate(entries):
            pos = (0.08, 0.95 - 0.05 * i - (0.025 if i >= len(VIEWS) else 0))
            b = plt.add_button(fn, states=states, c=["k"] * len(states),
                               bc=["#e0e0e0"] * len(states), pos=pos, size=16,
                               font="Arial", bold=False)
            if b is not None:
                self.buttons.append(b)
                if fn == self.toggle_title:
                    self.title_button = b
        import vedo

        self.message = vedo.Text2D("", pos="bottom-left", s=0.8, c="k")
        plt.add(self.message)

    def show_view(self, view: str) -> None:
        plt = self.scene.plotter
        w, h = plt.window.GetSize()
        cam = view_camera(self.scene, view, aspect=w / max(h, 1),
                          view_angle=plt.camera.GetViewAngle())
        plt.camera.SetFocalPoint(cam["focal_point"])
        plt.camera.SetPosition(cam["pos"])
        plt.camera.SetViewUp(cam["viewup"])
        plt.renderer.ResetCameraClippingRange()
        plt.render()

    def toggle_title(self, *_) -> None:
        set_title_visible(self.scene, not title_visible(self.scene))
        if self.title_button is not None:
            self.title_button.switch()
        self.scene.plotter.render()

    def _ask_path(self) -> str | None:
        default = f"brainrender_{datetime.now():%Y%m%d_%H%M%S}.png"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            return str(self.save_dir / default)
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            parent=root, title="Save brainrender image", initialdir=str(self.save_dir),
            initialfile=default, defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")])
        root.destroy()
        return path or None

    def save_image(self) -> None:
        path = self._ask_path()
        if not path:
            return
        plt = self.scene.plotter
        for b in self.buttons:
            b.off()
        self.message.text("")
        plt.render()
        plt.screenshot(path, scale=SCREENSHOT_SCALE)
        for b in self.buttons:
            b.on()
        self.message.text(f"Saved {path}")
        plt.render()
        print(f"Saved {path}")


def render(project, screenshot: str | None = None, view: str = "three_quarter",
           **options) -> None:
    scene = build_scene(project, offscreen=screenshot is not None, **options)
    camera = view_camera(scene, view)
    if screenshot:
        scene.render(interactive=False, camera=camera, zoom=1.0)
        scene.plotter.screenshot(screenshot, scale=SCREENSHOT_SCALE)
        scene.close()
        print(f"Saved {screenshot}")
    else:
        WindowControls(scene, Path(project.root) / "renders")
        scene.render(camera=camera, zoom=1.0)
