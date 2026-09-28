"""3D view of aligned sections and counted cells in brainrender."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .export import CELL_COLORS, collect_cells
from .io import to_display
from .transform import Alignment, SliceTransform


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


def build_scene(project, show_slices: bool = True, top_regions: int = 5, radius: float = 25,
                offscreen: bool = False):
    import brainrender
    from brainrender import Scene
    from brainrender.actors import Points

    from .atlas import Atlas

    if offscreen:
        brainrender.settings.OFFSCREEN = True
    atlas_name = project.data["atlas"]
    scene = Scene(atlas_name=atlas_name, title=Path(project.data["source"]).stem)
    cells = collect_cells(project)
    for i, (ctype, df) in enumerate(cells.groupby("cell_type", sort=False)):
        scene.add(Points(df[["ap_um", "dv_um", "ml_um"]].to_numpy(float), name=str(ctype),
                         colors=CELL_COLORS[i % len(CELL_COLORS)], radius=radius))
    if top_regions and len(cells):
        inside = cells[~cells["region_acronym"].isin(["outside", "root"])]
        acronyms = inside["region_acronym"].value_counts().index[:top_regions].tolist()
        if acronyms:
            scene.add_brain_region(*acronyms, alpha=0.2)
    if show_slices:
        atlas = Atlas(atlas_name)
        for s in project.slices:
            if project.status(s["id"])["aligned"]:
                scene.add(slice_mesh(atlas, project, s["id"]), names=f"slice_{s['id']:02d}",
                          classes="section")
    return scene


def render(project, show_slices: bool = True, top_regions: int = 5,
           screenshot: str | None = None) -> None:
    scene = build_scene(project, show_slices, top_regions, offscreen=screenshot is not None)
    if screenshot:
        scene.render(interactive=False)
        scene.screenshot(name=screenshot)
        scene.close()
    else:
        scene.render()
