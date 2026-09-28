"""Options for the brainrender 3D view, turned into ``slicereg render`` arguments."""

from __future__ import annotations

from html import escape

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                            QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QListWidget,
                            QListWidgetItem, QSpinBox, QVBoxLayout)

from ..render import HEMISPHERES, resolve_regions, split_region_text, suggest_regions

RENDER_DEFAULTS = {
    "structures": "",
    "top_regions": 5,
    "alpha": 0.3,
    "hemisphere": "both",
    "slices": True,
    "cells": True,
    "brain": True,
    "hidden_cell_types": [],
}


class RenderDialog(QDialog):
    def __init__(self, project, structures: list[dict], cell_types: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D view in brainrender")
        self.project = project
        self.structures = structures
        opts = {**RENDER_DEFAULTS, **project.data.get("render", {})}

        self.regions = QLineEdit(opts["structures"])
        self.regions.setPlaceholderText("e.g. IRt PCRt XII Mo5 Pe5 Acs5 7N")
        self.regions.textChanged.connect(self._update_preview)
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.preview.setTextFormat(Qt.RichText)

        self.top = QSpinBox()
        self.top.setRange(0, 50)
        self.top.setValue(int(opts["top_regions"]))
        self.top.setToolTip("Also show the N regions containing the most counted cells.")
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.05, 1.0)
        self.alpha.setSingleStep(0.05)
        self.alpha.setValue(float(opts["alpha"]))
        self.hemisphere = QComboBox()
        self.hemisphere.addItems(HEMISPHERES)
        self.hemisphere.setCurrentText(opts["hemisphere"])

        self.slices = QCheckBox("Show section images")
        self.slices.setChecked(opts["slices"])
        self.brain = QCheckBox("Show whole-brain outline")
        self.brain.setChecked(opts["brain"])
        self.cells = QCheckBox("Show cells")
        self.cells.setChecked(opts["cells"])
        self.cell_types = QListWidget()
        hidden = set(opts["hidden_cell_types"])
        for name in cell_types:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if name in hidden else Qt.Checked)
            self.cell_types.addItem(item)
        self.cell_types.setMaximumHeight(90)
        self.cells.toggled.connect(self.cell_types.setEnabled)
        self.cell_types.setEnabled(opts["cells"])

        form = QFormLayout()
        form.addRow("Structures", self.regions)
        form.addRow("", self.preview)
        form.addRow("Top regions by cell count", self.top)
        form.addRow("Structure opacity", self.alpha)
        form.addRow("Hemisphere", self.hemisphere)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Open 3D view")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.slices)
        lay.addWidget(self.brain)
        lay.addWidget(self.cells)
        if cell_types:
            lay.addWidget(self.cell_types)
        lay.addWidget(buttons)
        self.resize(460, self.sizeHint().height())
        self._update_preview()

    def _update_preview(self) -> None:
        queries = split_region_text(self.regions.text())
        if not queries:
            self.preview.setText("<i>No structures typed: only the top regions are shown.</i>")
            return
        found, missing = resolve_regions(queries, self.structures)
        lines = [f"{escape(q)} &rarr; <b>{escape(acr)}</b> ({escape(name)})"
                 for q, acr, name in found]
        for q in missing:
            hint = ", ".join(suggest_regions(q, self.structures))
            lines.append(f"<span style='color:#e06060'>{escape(q)}: not found"
                         + (f" (did you mean {escape(hint)}?)" if hint else "") + "</span>")
        self.preview.setText("<br>".join(lines))

    def options(self) -> dict:
        hidden = [self.cell_types.item(i).text() for i in range(self.cell_types.count())
                  if self.cell_types.item(i).checkState() != Qt.Checked]
        return {
            "structures": self.regions.text().strip(),
            "top_regions": self.top.value(),
            "alpha": round(self.alpha.value(), 2),
            "hemisphere": self.hemisphere.currentText(),
            "slices": self.slices.isChecked(),
            "cells": self.cells.isChecked(),
            "brain": self.brain.isChecked(),
            "hidden_cell_types": hidden,
        }

    def cli_args(self) -> list[str]:
        o = self.options()
        args = ["--regions", str(o["top_regions"]), "--alpha", str(o["alpha"]),
                "--hemisphere", o["hemisphere"]]
        found, _ = resolve_regions(split_region_text(o["structures"]), self.structures)
        if found:
            args += ["--structures", *[acr for _, acr, _ in found]]
        if not o["slices"]:
            args.append("--no-slices")
        if not o["brain"]:
            args.append("--no-brain")
        if not o["cells"]:
            args.append("--no-cells")
        else:
            shown = [self.cell_types.item(i).text() for i in range(self.cell_types.count())
                     if self.cell_types.item(i).checkState() == Qt.Checked]
            if o["hidden_cell_types"] and shown:
                args += ["--cell-types", *shown]
            elif o["hidden_cell_types"]:
                args.append("--no-cells")
        return args

    def save(self) -> None:
        self.project.data["render"] = self.options()
        self.project.save()
