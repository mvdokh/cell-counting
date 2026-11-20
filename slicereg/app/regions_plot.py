"""Bar plots of cell counts by region (and by hemisphere) for the slide view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from qtpy.QtWidgets import QDialog, QLabel, QVBoxLayout

from ..export import (HEMISPHERE_COLORS, NOT_REGIONS, save_hemisphere_bar_plot,
                      save_region_bar_plot, top_regions)
from ..region_groups import (FUNCTION_COLORS, FUNCTION_ORDER, MAJOR_COLORS, MAJOR_ORDER,
                             annotate_region_groups, count_by_group, save_division_plots,
                             top_regions_in_division)

KINDS = ("region", "hemisphere", "groups")


def _integer_cell_axis(ax) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))


class RegionsPlot(QDialog):
    """``kind``: ``region``, ``hemisphere``, or ``groups`` (major divisions + sensory/motor)."""

    def __init__(self, cells, title: str | None = None, parent=None,
                 save_dir: Path | None = None, *, kind: str = "region",
                 atlas_name: str | None = None):
        super().__init__(parent)
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        if title is None:
            title = {"region": "Cells by region",
                     "hemisphere": "Cells by region and hemisphere",
                     "groups": "Cells by brain division"}[kind]
        self.setWindowTitle(title)

        if kind == "groups":
            if not atlas_name:
                raise ValueError("atlas_name is required for kind='groups'")
            width = 14.0
            fig = Figure(figsize=(width, 9.0), layout="constrained")
            self.canvas = FigureCanvasQTAgg(fig)
            self._draw_groups(fig, cells, atlas_name, save_dir)
            height = 780
        else:
            top = top_regions(cells)
            n_reg = len(top)
            tick_size = 8 if n_reg > 20 else 9
            labels = top["region_acronym"].astype(str) if n_reg else []
            if kind == "region":
                width = max(9.0, 0.5 * n_reg + 3.0)
            else:
                width = max(10.0, 0.55 * n_reg + 4.5)
            fig = Figure(figsize=(width, 5.5), layout="constrained")
            self.canvas = FigureCanvasQTAgg(fig)
            if not len(top):
                ax = fig.subplots()
                ax.text(0.5, 0.5, "No cells in named regions yet.", ha="center", va="center")
                ax.set_axis_off()
            elif kind == "region":
                self._draw_region(fig, top, labels, tick_size, save_dir)
            else:
                self._draw_hemisphere(fig, cells, top, labels, n_reg, tick_size, save_dir)
            height = 620

        lay = QVBoxLayout(self)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)
        n = len(cells)
        lay.addWidget(QLabel(f"{n} cells" + (f" across the experiment." if n else ".")))
        self.resize(max(900, int(90 * width) if kind != "groups" else 1200), height)

    def _draw_region(self, fig, top, labels, tick_size, save_dir) -> None:
        ax = fig.subplots()
        ax.bar(labels, top["cells"], color="0.35")
        ax.set_ylabel("Cells")
        ax.set_xlabel("Region")
        ax.set_title("Cells by region")
        ax.tick_params(axis="x", rotation=45, labelsize=tick_size)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.margins(x=0.01)
        ax.spines[["top", "right"]].set_visible(False)
        _integer_cell_axis(ax)
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_region_bar_plot(top, save_dir / "top_regions.png")

    def _draw_hemisphere(self, fig, cells, top, labels, n_reg, tick_size, save_dir) -> None:
        sides = list(HEMISPHERE_COLORS)
        totals = cells["hemisphere"].value_counts().reindex(sides, fill_value=0)
        ax_total, ax_region = fig.subplots(
            1, 2, width_ratios=[1.4, max(2.2, 0.55 * n_reg)])
        fig.get_layout_engine().set(w_pad=0.4, wspace=0.12)

        ax_total.bar(sides, totals.values, color=[HEMISPHERE_COLORS[s] for s in sides])
        for x, n in enumerate(totals.values):
            ax_total.annotate(str(int(n)), (x, n), ha="center", va="bottom",
                              xytext=(0, 2), textcoords="offset points", fontsize=8)
        ax_total.set_ylabel("Cells")
        ax_total.set_xlabel("Hemisphere")
        ax_total.set_title("All cells")
        ax_total.spines[["top", "right"]].set_visible(False)
        _integer_cell_axis(ax_total)

        x = np.arange(n_reg)
        bar_w = 0.4
        for i, side in enumerate(sides):
            ax_region.bar(x + (i - 0.5) * bar_w, top[side], bar_w,
                          color=HEMISPHERE_COLORS[side], label=side)
        ax_region.set_xticks(x, labels, rotation=45, ha="right")
        ax_region.tick_params(axis="x", labelsize=tick_size)
        ax_region.set_xlabel("Region")
        ax_region.set_title("Per region")
        ax_region.legend(frameon=False, title="Hemisphere", loc="upper right")
        ax_region.margins(x=0.01)
        ax_region.spines[["top", "right"]].set_visible(False)
        _integer_cell_axis(ax_region)
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_hemisphere_bar_plot(cells, top, save_dir / "top_regions_by_hemisphere.png")

    def _draw_groups(self, fig, cells, atlas_name: str, save_dir) -> None:
        annotated = annotate_region_groups(cells, atlas_name)
        inside = annotated[~annotated["region_acronym"].isin(NOT_REGIONS)]
        if not len(inside):
            ax = fig.subplots()
            ax.text(0.5, 0.5, "No cells in named regions yet.", ha="center", va="center")
            ax.set_axis_off()
            return

        major = count_by_group(annotated, "major_division", MAJOR_ORDER)
        function = count_by_group(annotated, "function_class", FUNCTION_ORDER)
        focus = ("Cortex", "Midbrain", "Brainstem")
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])
        ax_major = fig.add_subplot(gs[0, 0:2])
        ax_fun = fig.add_subplot(gs[0, 2])
        ax_focus = [fig.add_subplot(gs[1, i]) for i in range(3)]

        ax_major.bar(major["group"], major["cells"],
                     color=[MAJOR_COLORS.get(g, "#999999") for g in major["group"]])
        ax_major.set_ylabel("Cells")
        ax_major.set_title("Major brain divisions")
        ax_major.tick_params(axis="x", rotation=30)
        for label in ax_major.get_xticklabels():
            label.set_horizontalalignment("right")
        ax_major.spines[["top", "right"]].set_visible(False)
        _integer_cell_axis(ax_major)

        ax_fun.bar(function["group"], function["cells"],
                   color=[FUNCTION_COLORS[g] for g in function["group"]])
        for x, n in enumerate(function["cells"]):
            ax_fun.annotate(str(int(n)), (x, n), ha="center", va="bottom",
                            xytext=(0, 2), textcoords="offset points", fontsize=8)
        ax_fun.set_ylabel("Cells")
        ax_fun.set_title("Sensory vs motor")
        ax_fun.spines[["top", "right"]].set_visible(False)
        _integer_cell_axis(ax_fun)

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
            _integer_cell_axis(ax)
            ax.margins(x=0.01)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_division_plots(cells, atlas_name, save_dir / "cells_by_brain_division.png")
