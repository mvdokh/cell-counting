"""Group Allen atlas regions into major divisions and sensory / motor classes."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .export import NOT_REGIONS

# Major divisions matched by deepest Allen parent id on a structure's path.
# Brainstem here means hindbrain (pons + medulla), separate from midbrain.
MAJOR_DIVISIONS = (
    ("Cortex", ("CTX",)),
    ("Cerebral nuclei", ("STR", "PAL")),
    ("Thalamus", ("TH",)),
    ("Hypothalamus", ("HY",)),
    ("Midbrain", ("MB",)),
    ("Brainstem", ("HB",)),
    ("Cerebellum", ("CB",)),
)
MAJOR_OTHER = "Other"
MAJOR_ORDER = [label for label, _ in MAJOR_DIVISIONS] + [MAJOR_OTHER]

# Parent acronyms whose descendants count as sensory or motor. Overlaps are rare;
# if both match, sensory wins. Everything else is Other (including IRt/PCRt).
SENSORY_PARENTS = (
    "SS", "SSp", "SSs", "VIS", "AUD", "OLF", "GU", "VISC",
    "PSV", "SPVC", "SPVI", "SPVO", "NTS", "CU", "GR",
    "DC", "VCO", "DCO", "PB",
    "VPM", "VPL", "LGd", "MG", "LP", "PO",
)
MOTOR_PARENTS = (
    "MO", "MOp", "MOs",
    "III", "IV", "V", "VI", "VII", "XII", "AMB",
    "RN", "SNr", "SNc", "GRN", "PGRNd", "PGRNl",
    "CBX", "CBN", "FN", "IP", "DN",
    "STR", "PAL",
)
FUNCTION_ORDER = ("Sensory", "Motor", "Other")
FUNCTION_COLORS = {"Sensory": "#4c72b0", "Motor": "#dd8452", "Other": "#999999"}
MAJOR_COLORS = {
    "Cortex": "#e41a1c",
    "Cerebral nuclei": "#ff7f00",
    "Thalamus": "#984ea3",
    "Hypothalamus": "#f781bf",
    "Midbrain": "#377eb8",
    "Brainstem": "#4daf4a",
    "Cerebellum": "#a65628",
    "Other": "#999999",
}


def _by_acronym(structures: list[dict]) -> dict[str, dict]:
    return {str(s["acronym"]): s for s in structures if s.get("acronym")}


def _by_id(structures: list[dict]) -> dict[int, dict]:
    return {int(s["id"]): s for s in structures if "id" in s}


def _descendant_ids(parent_ids: set[int], structures: list[dict]) -> set[int]:
    """Every structure whose path includes any parent id (parents included)."""
    out: set[int] = set()
    for s in structures:
        path = s.get("structure_id_path") or [s["id"]]
        if any(pid in parent_ids for pid in path):
            out.add(int(s["id"]))
    return out


@lru_cache(maxsize=8)
def group_maps(atlas_name: str) -> dict:
    """Cached id→division / id→function maps for one BrainGlobe atlas name."""
    from .atlas import atlas_structures

    structures = atlas_structures(atlas_name)
    by_acr = _by_acronym(structures)
    major_by_id: dict[int, str] = {}
    for label, acronyms in MAJOR_DIVISIONS:
        for acr in acronyms:
            s = by_acr.get(acr)
            if s is not None:
                major_by_id[int(s["id"])] = label

    def expand(acronyms: tuple[str, ...]) -> set[int]:
        parents = {int(by_acr[a]["id"]) for a in acronyms if a in by_acr}
        return _descendant_ids(parents, structures)

    return {
        "by_id": _by_id(structures),
        "by_acronym": by_acr,
        "major_by_id": major_by_id,
        "sensory_ids": expand(SENSORY_PARENTS),
        "motor_ids": expand(MOTOR_PARENTS),
    }


def major_division_for(structure: dict | None, major_by_id: dict[int, str]) -> str:
    """Deepest matching major division on the structure path, or Other."""
    if not structure:
        return MAJOR_OTHER
    path = structure.get("structure_id_path") or [structure.get("id")]
    label = MAJOR_OTHER
    for pid in path:
        if pid in major_by_id:
            label = major_by_id[pid]
    return label


def function_class_for(region_id: int, sensory_ids: set[int], motor_ids: set[int]) -> str:
    rid = int(region_id)
    if rid in sensory_ids:
        return "Sensory"
    if rid in motor_ids:
        return "Motor"
    return "Other"


def annotate_region_groups(cells: pd.DataFrame, atlas_name: str) -> pd.DataFrame:
    """Copy of ``cells`` with ``major_division`` and ``function_class`` columns."""
    maps = group_maps(atlas_name)
    by_id, by_acr = maps["by_id"], maps["by_acronym"]
    major_by_id = maps["major_by_id"]
    sensory_ids, motor_ids = maps["sensory_ids"], maps["motor_ids"]

    out = cells.copy()
    majors, functions = [], []
    has_id = "region_id" in out.columns
    for row in out.itertuples(index=False):
        acr = str(getattr(row, "region_acronym", "") or "")
        if acr in NOT_REGIONS:
            majors.append(MAJOR_OTHER)
            functions.append("Other")
            continue
        rid = int(getattr(row, "region_id")) if has_id else None
        structure = by_id.get(rid) if rid else None
        if structure is None:
            structure = by_acr.get(acr)
            rid = int(structure["id"]) if structure else 0
        majors.append(major_division_for(structure, major_by_id))
        functions.append(function_class_for(rid or 0, sensory_ids, motor_ids))
    out["major_division"] = majors
    out["function_class"] = functions
    return out


def count_by_group(cells: pd.DataFrame, column: str, order: list[str] | tuple[str, ...]
                   ) -> pd.DataFrame:
    """Counts for ``column`` in fixed ``order`` (zeros for empty bins)."""
    inside = cells[~cells["region_acronym"].isin(NOT_REGIONS)] if "region_acronym" in cells else cells
    counts = inside[column].value_counts() if len(inside) else pd.Series(dtype=int)
    rows = [{"group": g, "cells": int(counts.get(g, 0))} for g in order]
    table = pd.DataFrame(rows)
    total = max(len(cells), 1)
    table["percent_of_cells"] = (100 * table["cells"] / total).round(2)
    return table


def top_regions_in_division(cells: pd.DataFrame, division: str, limit: int = 12
                            ) -> pd.DataFrame:
    """Top leaf regions (by count) inside one major division."""
    from .export import top_regions

    subset = cells[cells["major_division"] == division]
    if not len(subset):
        return top_regions(subset)
    top = top_regions(subset)
    return top.head(limit)


def save_division_plots(cells: pd.DataFrame, atlas_name: str, path) -> None:
    """Write the multi-panel major-division / sensory-motor figure."""
    from pathlib import Path

    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    annotated = annotate_region_groups(cells, atlas_name)
    inside = annotated[~annotated["region_acronym"].isin(NOT_REGIONS)]
    major = count_by_group(annotated, "major_division", MAJOR_ORDER)
    function = count_by_group(annotated, "function_class", FUNCTION_ORDER)
    focus = ("Cortex", "Midbrain", "Brainstem")

    fig = Figure(figsize=(14.0, 9.0), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])
    ax_major = fig.add_subplot(gs[0, 0:2])
    ax_fun = fig.add_subplot(gs[0, 2])
    ax_focus = [fig.add_subplot(gs[1, i]) for i in range(3)]

    colors = [MAJOR_COLORS.get(g, "#999999") for g in major["group"]]
    ax_major.bar(major["group"], major["cells"], color=colors)
    ax_major.set_ylabel("Cells")
    ax_major.set_title("Major brain divisions")
    ax_major.tick_params(axis="x", rotation=30)
    for label in ax_major.get_xticklabels():
        label.set_horizontalalignment("right")
    ax_major.spines[["top", "right"]].set_visible(False)
    ax_major.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))

    ax_fun.bar(function["group"], function["cells"],
               color=[FUNCTION_COLORS[g] for g in function["group"]])
    for x, n in enumerate(function["cells"]):
        ax_fun.annotate(str(int(n)), (x, n), ha="center", va="bottom",
                        xytext=(0, 2), textcoords="offset points", fontsize=8)
    ax_fun.set_ylabel("Cells")
    ax_fun.set_title("Sensory vs motor")
    ax_fun.spines[["top", "right"]].set_visible(False)
    ax_fun.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))

    for ax, division in zip(ax_focus, focus):
        top = top_regions_in_division(inside, division)
        if not len(top):
            ax.text(0.5, 0.5, f"No cells in {division}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        labels = top["region_acronym"].astype(str)
        ax.bar(labels, top["cells"], color=MAJOR_COLORS.get(division, "0.35"))
        ax.set_title(f"{division}: top regions")
        ax.set_ylabel("Cells")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))
        ax.margins(x=0.01)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    fig.clear()
