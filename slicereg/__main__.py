"""Command line entry point.

    python -m slicereg open  C:\\path\\to\\slide.tif     # detect, crop, align, count
    python -m slicereg open  C:\\path\\to\\image_folder  # one image file per section
    python -m slicereg export C:\\path\\to\\slide_project
    python -m slicereg render C:\\path\\to\\slide_project
    python -m slicereg deepslice-setup                  # one-off, for DeepSlice alignment
"""

from __future__ import annotations

import argparse


def main(argv=None) -> None:
    from .render import STYLES, VIEWS

    parser = argparse.ArgumentParser(prog="slicereg", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_open = sub.add_parser("open", help="open a slide TIFF, a folder of section images "
                                         "(see slicereg_folder.json), or a project folder")
    p_open.add_argument("path")
    p_open.add_argument("--project", help="project folder (default: <slide>_project next to it)")
    p_open.add_argument("--atlas", help="BrainGlobe atlas for a new project, "
                                        "e.g. allen_mouse_10um (default allen_mouse_25um)")
    p_open.add_argument("--spacing", type=float,
                        help="section spacing in um used to guess the next slice's AP position")

    p_export = sub.add_parser("export", help="write cells_all.csv, region_counts.csv and top_regions.csv")
    p_export.add_argument("project")

    p_render = sub.add_parser("render", help="show aligned slices and cells in brainrender")
    p_render.add_argument("project")
    p_render.add_argument("--structures", nargs="+", metavar="NAME",
                          help="brain structures to show, Allen or Paxinos acronyms, "
                               "optionally with a colour: e.g. IRt:red PCRt:#3080ff XII Mo5")
    p_render.add_argument("--regions", type=int, default=5,
                          help="also show the N regions with most cells (0 = none)")
    p_render.add_argument("--alpha", type=float, default=0.3, help="structure opacity (0-1)")
    p_render.add_argument("--hemisphere", choices=["both", "left", "right"], default="both",
                          help="which side of the structures to show")
    p_render.add_argument("--no-slices", action="store_true", help="hide section images")
    p_render.add_argument("--no-cells", action="store_true", help="hide counted cells")
    p_render.add_argument("--cell-types", nargs="+", metavar="TYPE",
                          help="only show these cell types")
    p_render.add_argument("--no-brain", action="store_true",
                          help="hide the transparent whole brain")
    p_render.add_argument("--brain-alpha", type=float, default=0.3,
                          help="whole-brain opacity (0-1)")
    p_render.add_argument("--style", choices=list(STYLES), default="plastic",
                          help="surface shading of brain and structures")
    p_render.add_argument("--view", choices=list(VIEWS), default="three_quarter",
                          help="starting camera view")
    p_render.add_argument("--axes", action="store_true", help="show axes / scale bars")
    p_render.add_argument("--no-inset", action="store_true",
                          help="hide the small orientation brain in the corner")
    p_render.add_argument("--cell-size", type=float, default=25.0,
                          help="radius of the cell spheres in um")
    p_render.add_argument("--no-title", action="store_true",
                          help="hide the title at the top of the window")
    p_render.add_argument("--screenshot", help="save a PNG instead of opening a window")

    sub.add_parser("deepslice-setup",
                   help="install DeepSlice (and TensorFlow) in its own environment, "
                        "for the 'Predict alignment with DeepSlice' button")

    args = parser.parse_args(argv)
    if args.cmd == "deepslice-setup":
        from .deepslice import ENV_DIR, setup

        setup()
        print(f"DeepSlice is ready ({ENV_DIR}).")
        return
    from .io import Project

    if args.cmd == "open":
        project = Project.open_or_create(args.path, args.project)
        changed = False
        if args.atlas and not any(project.status(s["id"])["aligned"] for s in project.slices):
            project.data["atlas"] = args.atlas
            changed = True
        if args.spacing is not None:
            project.data["section_spacing_um"] = args.spacing
            changed = True
        if changed:
            project.save()
        from .app import run

        run(str(project.root))
    elif args.cmd == "export":
        from .export import export_project

        print(export_project(Project.load(args.project)))
    elif args.cmd == "render":
        from .render import render, split_region_text

        structures = [t for s in args.structures or [] for t in split_region_text(s)]
        render(Project.load(args.project), screenshot=args.screenshot, view=args.view,
               show_slices=not args.no_slices, top_regions=args.regions,
               structures=structures, show_cells=not args.no_cells,
               cell_types=args.cell_types, show_brain=not args.no_brain,
               region_alpha=args.alpha, hemisphere=args.hemisphere,
               brain_alpha=args.brain_alpha, style=args.style, show_axes=args.axes,
               show_inset=not args.no_inset, radius=args.cell_size,
               show_title=not args.no_title)


if __name__ == "__main__":
    main()
