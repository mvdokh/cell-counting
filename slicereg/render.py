"""3D view of aligned sections and counted cells in brainrender."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .export import CELL_COLORS, collect_cells
from .io import to_display
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


def split_region_text(text: str) -> list[str]:
    """'IRt/PCRt, XII mo5' -> ['IRt', 'PCRt', 'XII', 'mo5']."""
    return [t for t in re.split(r"[\s,;/]+", text or "") if t]


def resolve_regions(queries, structures) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Match region names to atlas structures, case-insensitively.

    Returns ``([(query, acronym, full name), ...], [unmatched queries])`` with
    duplicate acronyms dropped.
    """
    by_acronym = {s["acronym"].lower(): s for s in structures}
    found, missing, seen = [], [], set()
    for q in queries:
        key = q.strip().lower()
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
    q = query.strip().lower()
    hits = [s["acronym"] for s in structures if s["acronym"].lower().startswith(q)]
    hits += [s["acronym"] for s in structures
             if q in s["name"].lower() and s["acronym"] not in hits]
    return hits[:limit]


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
    rgb = to_display(project.load_preview(sid))
    alpha = (project.load_mask(sid) * 255).astype(np.uint8)[..., None]
    tex = np.concatenate([rgb, alpha], axis=2)
    mesh = vedo.Mesh([verts, faces])
    mesh.texture(tex, tcoords=np.stack([u.ravel(), 1 - v.ravel()], 1))
    return mesh


def build_scene(project, show_slices: bool = True, top_regions: int = 5,
                structures: list[str] | None = None, show_cells: bool = True,
                cell_types: list[str] | None = None, show_brain: bool = True,
                region_alpha: float = 0.3, hemisphere: str = "both", radius: float = 25,
                offscreen: bool = False):
    import brainrender
    from brainrender import Scene
    from brainrender.actors import Points

    from .atlas import Atlas

    if offscreen:
        brainrender.settings.OFFSCREEN = True
    atlas_name = project.data["atlas"]
    scene = Scene(root=show_brain, atlas_name=atlas_name, check_latest=False,
                  title=Path(project.data["source"]).stem)
    cells = collect_cells(project)
    if cell_types:
        cells = cells[cells["cell_type"].isin(cell_types)]
    if show_cells:
        for i, (ctype, df) in enumerate(cells.groupby("cell_type", sort=False)):
            scene.add(Points(df[["ap_um", "dv_um", "ml_um"]].to_numpy(float),
                             name=str(ctype), colors=CELL_COLORS[i % len(CELL_COLORS)],
                             radius=radius))

    acronyms = []
    if structures:
        found, missing = resolve_regions(structures, scene.atlas.structures_list)
        acronyms += [acr for _, acr, _ in found]
        if missing:
            print(f"Unknown structures (skipped): {', '.join(missing)}")
    if top_regions and len(cells):
        inside = cells[~cells["region_acronym"].isin(["outside", "root"])]
        acronyms += [a for a in inside["region_acronym"].value_counts().index[:top_regions]
                     if a not in acronyms]
    if acronyms:
        scene.add_brain_region(*acronyms, alpha=region_alpha, hemisphere=hemisphere)

    if show_slices:
        atlas = Atlas(atlas_name)
        for s in project.slices:
            if project.status(s["id"])["aligned"]:
                scene.add(slice_mesh(atlas, project, s["id"]), names=f"slice_{s['id']:02d}",
                          classes="section")
    return scene


def render(project, screenshot: str | None = None, **options) -> None:
    scene = build_scene(project, offscreen=screenshot is not None, **options)
    if screenshot:
        scene.render(interactive=False)
        scene.screenshot(name=screenshot)
        scene.close()
    else:
        scene.render()
