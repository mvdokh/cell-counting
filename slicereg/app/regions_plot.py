"""Bar plots of cell counts by region (and by hemisphere) for the slide view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from qtpy.QtWidgets import QDialog, QLabel, QVBoxLayout

from ..export import (HEMISPHERE_COLORS, save_hemisphere_bar_plot, save_region_bar_plot,
                      top_regions)


def _integer_cell_axis(ax) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))


class RegionsPlot(QDialog):
    """``kind`` is ``"region"`` (totals by region) or ``"hemisphere"`` (by side)."""

    def __init__(self, cells, title: str | None = None, parent=None,
                 save_dir: Path | None = None, *, kind: str = "region"):
        super().__init__(parent)
        if kind not in ("region", "hemisphere"):
            raise ValueError(f"kind must be 'region' or 'hemisphere', got {kind!r}")
        self.kind = kind
        if title is None:
            title = ("Cells by region" if kind == "region"
                     else "Cells by region and hemisphere")
        self.setWindowTitle(title)
        top = top_regions(cells)
        n_reg = len(top)
        tick_size = 8 if n_reg > 20 else 9
        labels = top["region_acronym"].astype(str) if n_reg else []

        if kind == "region":
            width = max(9.0, 0.5 * n_reg + 3.0)
            fig = Figure(figsize=(width, 5.5), layout="constrained")
        else:
            width = max(10.0, 0.55 * n_reg + 4.5)
            fig = Figure(figsize=(width, 5.5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(fig)

        if not len(top):
            ax = fig.subplots()
            ax.text(0.5, 0.5, "No cells in named regions yet.", ha="center", va="center")
            ax.set_axis_off()
        elif kind == "region":
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
        else:
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

        lay = QVBoxLayout(self)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)
        n = len(cells)
        lay.addWidget(QLabel(f"{n} cells" + (f" across the experiment." if n else ".")))
        self.resize(max(900, int(90 * width)), 620)
