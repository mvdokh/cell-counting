"""Plot each section's aligned AP position and pitch against its place in the series."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtWidgets import QDialog, QLabel, QVBoxLayout


def section_positions(project) -> tuple[list[str], list[int], np.ndarray, np.ndarray]:
    """(names, ids, AP um, pitch deg) for every section in slide order; NaN = not aligned."""
    names, sids, ap, pitch = [], [], [], []
    for s in project.slices:
        al = project.load_alignment(s["id"])
        names.append(s.get("name") or f"{s['id']:02d}")
        sids.append(s["id"])
        ap.append(np.nan if al is None else float(al["ap_um"]))
        pitch.append(np.nan if al is None else float(al.get("pitch_deg", 0.0)))
    return names, sids, np.array(ap), np.array(pitch)


class PositionsPlot(QDialog):
    def __init__(self, project, open_section, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Section AP position and pitch")
        self.open_section = open_section
        names, self.sids, ap, pitch = section_positions(project)
        x = np.arange(len(names))

        fig = Figure(figsize=(9, 6.5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(fig)
        ax_ap, ax_pitch = fig.subplots(2, 1, sharex=True)
        for ax, y, label in ((ax_ap, ap, "AP position (um)"), (ax_pitch, pitch, "Pitch (deg)")):
            ax.plot(x, y, "-", color="0.6", zorder=1)
            ax.plot(x, y, "o", color="tab:blue", picker=6, zorder=2)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        ax_pitch.set_ylim(-10, 10)
        ax_pitch.set_xticks(x, names, rotation=45, ha="right")
        ax_pitch.set_xlabel("Section (slide order)")
        self.canvas.mpl_connect("pick_event", self._on_pick)

        lay = QVBoxLayout(self)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)
        lay.addWidget(QLabel("Click a point to open that section."))
        self.resize(900, 650)

    def _on_pick(self, event) -> None:
        ind = getattr(event, "ind", None)
        if ind is None or not len(ind):
            return
        self.accept()
        self.open_section(int(self.sids[int(ind[0])]))
