"""Bar plots of cell counts by region (and by hemisphere) for the slide view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtWidgets import QDialog, QLabel, QVBoxLayout

from ..export import (HEMISPHERE_COLORS, save_hemisphere_bar_plot, save_region_bar_plot,
                      top_regions)


class RegionsPlot(QDialog):
    def __init__(self, cells, title: str = "Cells by region", parent=None,
                 save_dir: Path | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        top = top_regions(cells)
        fig = Figure(figsize=(10, 7.5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(fig)

        if not len(top):
            ax = fig.subplots()
            ax.text(0.5, 0.5, "No cells in named regions yet.", ha="center", va="center")
            ax.set_axis_off()
        else:
            # Region totals
            ax_reg = fig.add_subplot(2, 1, 1)
            ax_reg.bar(top["region_acronym"].astype(str), top["cells"], color="0.35")
            ax_reg.set_ylabel("Cells")
            ax_reg.set_title("Cells by region")
            ax_reg.tick_params(axis="x", rotation=45)
            for label in ax_reg.get_xticklabels():
                label.set_horizontalalignment("right")
            ax_reg.spines[["top", "right"]].set_visible(False)

            # Hemispheres: totals + per region
            sides = list(HEMISPHERE_COLORS)
            totals = cells["hemisphere"].value_counts().reindex(sides, fill_value=0)
            ax_total = fig.add_subplot(2, 2, 3)
            ax_region = fig.add_subplot(2, 2, 4)
            ax_total.bar(sides, totals.values, color=[HEMISPHERE_COLORS[s] for s in sides])
            for x, n in enumerate(totals.values):
                ax_total.annotate(str(int(n)), (x, n), ha="center", va="bottom",
                                  xytext=(0, 2), textcoords="offset points", fontsize=8)
            ax_total.set_ylabel("Cells")
            ax_total.set_title("All cells")
            ax_total.spines[["top", "right"]].set_visible(False)

            x = np.arange(len(top))
            width = 0.4
            for i, side in enumerate(sides):
                ax_region.bar(x + (i - 0.5) * width, top[side], width,
                              color=HEMISPHERE_COLORS[side], label=side)
            ax_region.set_xticks(x, top["region_acronym"].astype(str), rotation=45, ha="right")
            ax_region.set_title("Per region")
            ax_region.legend(frameon=False, title="Hemisphere")
            ax_region.spines[["top", "right"]].set_visible(False)

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_region_bar_plot(top, save_dir / "top_regions.png")
                save_hemisphere_bar_plot(cells, top, save_dir / "top_regions_by_hemisphere.png")

        lay = QVBoxLayout(self)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)
        n = len(cells)
        lay.addWidget(QLabel(f"{n} cells" + (f" across the experiment." if n else ".")))
        self.resize(1000, 750)
