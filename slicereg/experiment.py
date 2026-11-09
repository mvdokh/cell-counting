"""Group numbered slide folders under one experiment (e.g. PCRt_Rabies_2/7, /8).

Each slide keeps its own project. Section order runs through slides in numeric order,
so after slide 7's last section (e.g. c5_r3) comes slide 8's first (c1_r1).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .export import CELL_COLUMNS, collect_cells
from .io import Project

_SLIDE_NAME = re.compile(r"^\d+$")
_CNR_MIP = re.compile(r"^c\d+r\d+_mip\.lsm$", re.IGNORECASE)
_IMAGE_GLOB = ("*.lsm", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png", "*.bmp")
EXPERIMENT_CONFIG = "slicereg_experiment.json"


def experiment_root(project: Project) -> Path | None:
    """Parent folder of a numbered slide image folder, or None if not in an experiment."""
    if not project.is_folder:
        return None
    source = Path(project.data["source"])
    if not _SLIDE_NAME.fullmatch(source.name):
        return None
    return source.parent.resolve()


def slide_number(project: Project) -> int | None:
    root = experiment_root(project)
    if root is None:
        return None
    return int(Path(project.data["source"]).name)


def is_experiment_dir(path: Path | str) -> bool:
    """True if ``path`` holds numbered slide folders (e.g. ``PCRt_Rabies_2/7``)."""
    path = Path(path)
    return path.is_dir() and any(
        p.is_dir() and _SLIDE_NAME.fullmatch(p.name) for p in path.iterdir())


def slide_has_images(folder: Path) -> bool:
    folder = Path(folder)
    return any(next(folder.glob(pat), None) is not None for pat in _IMAGE_GLOB)


def list_slides(experiment: Path, *, usable_only: bool = False) -> list[dict]:
    """Numbered slide folders under ``experiment``, sorted: ``{slide, folder, project}``.

    With ``usable_only``, empty image folders that have no project yet are omitted."""
    experiment = Path(experiment)
    out = []
    for p in experiment.iterdir():
        if not (p.is_dir() and _SLIDE_NAME.fullmatch(p.name)):
            continue
        n = int(p.name)
        proj = experiment / f"{n}_project"
        if not (proj / "project.json").exists():
            proj = p / "project"  # uncommon layout
            if not (proj / "project.json").exists():
                proj = None
        if usable_only and proj is None and not slide_has_images(p):
            continue
        out.append({"slide": n, "folder": p, "project": proj})
    return sorted(out, key=lambda s: s["slide"])


def sibling_slides(project: Project) -> list[dict]:
    """Slide list for this project's experiment, or just this slide if it is alone."""
    root = experiment_root(project)
    if root is None:
        source = Path(project.data["source"])
        return [{"slide": None, "folder": source, "project": project.root}]
    return list_slides(root, usable_only=True)


def _ensure_slide_config(folder: Path) -> None:
    """Write a folder config if missing, using the cNrN_mip naming when that fits."""
    from .folder import CONFIG_HELP, CONFIG_NAME, DEFAULT_CONFIG
    import json

    folder = Path(folder)
    path = folder / CONFIG_NAME
    if path.exists():
        return
    config = dict(DEFAULT_CONFIG)
    names = [p.name for p in folder.iterdir() if p.is_file()]
    if any(_CNR_MIP.match(n) for n in names):
        config["pattern"] = r"^c(?P<col>\d+)r(?P<row>\d+)_mip\.lsm$"
        config["channels"] = [
            {"name": "gfp", "color": "green"},
            {"name": "rfp", "color": "red"},
            {"name": "nissl", "color": "blue"},
        ]
        config["align_channel"] = "nissl"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**config, "_help": CONFIG_HELP}, f, indent=2)


def open_slide_project(entry: dict) -> Project:
    """Load or create the project for one slide entry from ``list_slides``."""
    if entry["project"] is not None and (Path(entry["project"]) / "project.json").exists():
        return Project.load(entry["project"])
    _ensure_slide_config(entry["folder"])
    return Project.open_or_create(entry["folder"])


def open_experiment(path: Path | str) -> Project:
    """Open an experiment folder: ensure each slide with images has a project, return
    the first one (lowest slide number)."""
    from .folder import find_sections, load_config

    path = Path(path).resolve()
    if not is_experiment_dir(path):
        raise FileNotFoundError(f"{path} is not an experiment (no numbered slide folders)")
    first: Project | None = None
    opened = []
    for entry in list_slides(path, usable_only=True):
        if entry["project"] is None and not slide_has_images(entry["folder"]):
            continue
        if entry["project"] is None:
            _ensure_slide_config(entry["folder"])
            config, _ = load_config(entry["folder"])
            if not find_sections(entry["folder"], config):
                continue
        proj = open_slide_project(entry)
        opened.append(entry["slide"])
        if first is None:
            first = proj
    if first is None:
        raise FileNotFoundError(
            f"No slide folders with section images under {path}. "
            "Expected numbered folders like 6/, 7/ with .lsm files.")
    print(f"Experiment {path.name}: slides {', '.join(str(s) for s in opened)}")
    sync_experiment_cell_types(first)
    return first


def load_experiment_projects(project: Project) -> list[tuple[int | None, Project]]:
    """``(slide_number, Project)`` for every sibling that already has a project.

    The current project is always included. Slides without a project yet are skipped
    (open them from the slide switcher to create one)."""
    current_n = slide_number(project)
    found: list[tuple[int | None, Project]] = []
    seen = {project.root.resolve()}
    for entry in sibling_slides(project):
        if entry["project"] is None:
            continue
        root = Path(entry["project"]).resolve()
        if root == project.root.resolve():
            found.append((entry["slide"], project))
            continue
        if root in seen or not (root / "project.json").exists():
            continue
        seen.add(root)
        found.append((entry["slide"], Project.load(root)))
    if not any(p.root.resolve() == project.root.resolve() for _, p in found):
        found.insert(0, (current_n, project))
    return sorted(found, key=lambda t: (t[0] is None, t[0] or 0))


def section_label(slide: int | None, name: str) -> str:
    return f"{slide}/{name}" if slide is not None else name


def experiment_sections(project: Project) -> list[dict]:
    """Every section across the experiment, in cutting order.

    Each dict has ``slide``, ``project``, ``sid``, ``name``, ``label``.
    """
    out = []
    for slide, proj in load_experiment_projects(project):
        for s in proj.slices:
            name = s.get("name") or f"{s['id']:02d}"
            out.append({
                "slide": slide,
                "project": proj,
                "sid": s["id"],
                "name": name,
                "label": section_label(slide, name),
            })
    return out


def experiment_positions(project: Project) -> tuple[list[str], list[dict], np.ndarray, np.ndarray]:
    """(labels, section refs, AP um, pitch deg) for the whole experiment."""
    labels, refs, ap, pitch = [], [], [], []
    for sec in experiment_sections(project):
        al = sec["project"].load_alignment(sec["sid"])
        labels.append(sec["label"])
        refs.append(sec)
        ap.append(np.nan if al is None else float(al["ap_um"]))
        pitch.append(np.nan if al is None else float(al.get("pitch_deg", 0.0)))
    return labels, refs, np.array(ap), np.array(pitch)


def collect_experiment_cells(project: Project) -> pd.DataFrame:
    """All counted cells from every slide project, with a ``slide`` column."""
    frames = []
    for slide, proj in load_experiment_projects(project):
        df = collect_cells(proj)
        if df is None or not len(df):
            continue
        df = df.copy()
        df.insert(0, "slide", slide if slide is not None else "")
        # Keep slice_id unique across slides for display: "7:3"
        if slide is not None:
            df["slice_id"] = [f"{slide}:{sid}" for sid in df["slice_id"]]
        frames.append(df)
    if not frames:
        cols = ["slide", *CELL_COLUMNS]
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def experiment_config_path(root: Path | str) -> Path:
    return Path(root) / EXPERIMENT_CONFIG


def load_experiment_config(root: Path | str) -> dict:
    path = experiment_config_path(root)
    if not path.is_file():
        return {}
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def experiment_cell_types_stored(root: Path | str) -> list[str]:
    """Cell types saved for the whole experiment, or []."""
    raw = load_experiment_config(root).get("cell_types") or []
    return list(dict.fromkeys(str(t).strip() for t in raw if str(t).strip()))


def save_experiment_cell_types(root: Path | str, types: list[str]) -> None:
    import json

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_experiment_config(root)
    cfg["cell_types"] = list(dict.fromkeys(str(t).strip() for t in types if str(t).strip()))
    with open(experiment_config_path(root), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def sync_experiment_cell_types(project: Project) -> list[str]:
    """Union types from the experiment file and every slide; write them everywhere."""
    root = experiment_root(project)
    if root is None:
        return project.cell_types()
    types: list[str] = []
    for name in experiment_cell_types_stored(root):
        if name not in types:
            types.append(name)
    for _, proj in load_experiment_projects(project):
        for name in list(proj.data.get("cell_types") or []):
            name = str(name).strip()
            if name and name not in types:
                types.append(name)
        for s in proj.slices:
            df = proj.load_cells(s["id"])
            if df is None or not len(df) or "cell_type" not in df.columns:
                continue
            for name in df["cell_type"].astype(str):
                if name not in types:
                    types.append(name)
    if not types:
        types = ["cell"]
    save_experiment_cell_types(root, types)
    for _, proj in load_experiment_projects(project):
        if list(proj.data.get("cell_types") or []) != types:
            proj.data["cell_types"] = types
            proj.save()
    return types


def experiment_cell_types(project: Project) -> list[str]:
    root = experiment_root(project)
    if root is not None:
        return sync_experiment_cell_types(project)
    types: list[str] = []
    for name in project.cell_types():
        if name not in types:
            types.append(name)
    cells = collect_experiment_cells(project)
    if len(cells) and "cell_type" in cells.columns:
        for name in cells["cell_type"].astype(str):
            if name not in types:
                types.append(name)
    return types or ["cell"]


def _section_flip_label(slide: int | None, name: str) -> str:
    return section_label(slide, name)


def _slice_flip_label(project: Project, sid: int) -> str:
    s = project.get_slice(sid)
    name = s.get("name") or f"{sid:02d}"
    return _section_flip_label(slide_number(project), name)


def save_experiment_flips(root: Path | str, labels: list[str]) -> None:
    """Write the animal-level list of manually flipped sections."""
    import json

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_experiment_config(root)
    cfg["flipped"] = sorted(dict.fromkeys(str(x) for x in labels if str(x).strip()))
    with open(experiment_config_path(root), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def collect_flipped_labels(project: Project) -> list[str]:
    """Labels of sections with ``user_flipped`` across the experiment (or this project)."""
    out = []
    for slide, proj in load_experiment_projects(project):
        for s in proj.slices:
            if not s.get("user_flipped"):
                continue
            name = s.get("name") or f"{s['id']:02d}"
            out.append(_section_flip_label(slide, name))
    return sorted(dict.fromkeys(out))


def record_section_flip(project: Project, sid: int) -> None:
    """After toggling ``user_flipped``, refresh the animal (or project) flip manifest."""
    root = experiment_root(project)
    labels = collect_flipped_labels(project)
    if root is not None:
        save_experiment_flips(root, labels)
        return
    project.data["flipped"] = labels
    project.save()


def sync_experiment_flips(project: Project) -> list[str]:
    """Merge the flip manifest with each slice's ``user_flipped`` flag.

    Manifest entries set the flag; existing flags are kept and written back so the
    animal-level list stays complete. Returns the flipped labels."""
    root = experiment_root(project)
    wanted: set[str] = set()
    if root is not None:
        wanted.update(str(x) for x in (load_experiment_config(root).get("flipped") or [])
                      if str(x).strip())
    else:
        wanted.update(str(x) for x in (project.data.get("flipped") or []) if str(x).strip())

    for slide, proj in load_experiment_projects(project):
        dirty = False
        for s in proj.slices:
            name = s.get("name") or f"{s['id']:02d}"
            label = _section_flip_label(slide, name)
            if s.get("user_flipped"):
                wanted.add(label)
            elif label in wanted:
                s["user_flipped"] = True
                dirty = True
        if dirty:
            proj.save()

    labels = sorted(wanted)
    if root is not None:
        save_experiment_flips(root, labels)
    else:
        if list(project.data.get("flipped") or []) != labels:
            project.data["flipped"] = labels
            project.save()
    return labels
