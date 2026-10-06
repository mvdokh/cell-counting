"""Options for the brainrender 3D view, turned into ``slicereg render`` arguments."""

from __future__ import annotations

from html import escape

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                            QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                            QListWidget, QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout)

from ..render import (HEMISPHERES, STYLES, VIEW_LABELS, parse_color, resolve_regions,
                      split_color, split_region_text, suggest_regions)

RENDER_DEFAULTS = {
    "structures": "",
    "top_regions": 5,
    "alpha": 0.3,
    "hemisphere": "both",
    "slices": True,
    "cells": True,
    "brain": True,
    "brain_alpha": 0.3,
    "style": "plastic",
    "axes": False,
    "inset": True,
    "title": True,
    "cell_size": 25.0,
    "view": "three_quarter",
    "hidden_cell_types": [],
}
STYLE_LABELS = {
    "plastic": "Smooth (plastic)",
    "shiny": "Shiny",
    "glossy": "Glossy",
    "metallic": "Metallic",
    "cartoon": "Cartoon (flat, outlined)",
}


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)


def _spin(value: float, lo: float = 0.05, hi: float = 1.0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(0.05)
    s.setValue(value)
    return s


class RenderDialog(QDialog):
    def __init__(self, project, structures: list[dict], cell_types: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D view in brainrender")
        self.project = project
        self.structures = structures
        opts = {**RENDER_DEFAULTS, **project.data.get("render", {})}

        self.regions = QLineEdit(opts["structures"])
        self.regions.setPlaceholderText("e.g. IRt:red PCRt:#3080ff XII Mo5 Pe5 Acs5 7N")
        self.regions.setToolTip(
            "Allen or Paxinos acronyms. Add ':colour' to colour a structure, e.g. IRt:red, "
            "PCRt:#3080ff or XII:gold; without one the atlas colour is used.")
        self.regions.textChanged.connect(self._update_preview)
        self.pick_color = QPushButton("Colour...")
        self.pick_color.setToolTip("Pick a colour for the last structure in the list.")
        self.pick_color.clicked.connect(self._on_pick_color)
        regions_row = QHBoxLayout()
        regions_row.addWidget(self.regions, 1)
        regions_row.addWidget(self.pick_color)
        self.preview = QLabel()
        self.preview.setTextFormat(Qt.RichText)

        self.top = QSpinBox()
        self.top.setRange(0, 50)
        self.top.setValue(int(opts["top_regions"]))
        self.top.setToolTip("Also show the N regions containing the most counted cells.")
        self.alpha = _spin(float(opts["alpha"]))
        self.hemisphere = QComboBox()
        self.hemisphere.addItems(HEMISPHERES)
        self.hemisphere.setCurrentText(opts["hemisphere"])

        self.brain_alpha = _spin(float(opts["brain_alpha"]), 0.0, 1.0)
        self.brain_alpha.setToolTip("0 = invisible, 1 = solid.")
        self.style = QComboBox()
        for key in STYLES:
            self.style.addItem(STYLE_LABELS[key], key)
        self.style.setCurrentIndex(max(0, STYLES.index(opts["style"])
                                       if opts["style"] in STYLES else 0))
        self.view = QComboBox()
        for key, label in VIEW_LABELS.items():
            self.view.addItem(label, key)
        self.view.setCurrentIndex(max(0, self.view.findData(opts["view"])))
        self.view.setToolTip("Initial camera; buttons in the 3D window switch between views.")

        self.slices = QCheckBox("Show section images")
        self.slices.setChecked(opts["slices"])
        self.brain = QCheckBox("Show whole brain")
        self.brain.setChecked(opts["brain"])
        self.axes = QCheckBox("Show axes / scale bars")
        self.axes.setChecked(opts["axes"])
        self.inset = QCheckBox("Show small orientation brain in the corner")
        self.inset.setChecked(opts["inset"])
        self.title = QCheckBox("Show title")
        self.title.setChecked(opts["title"])
        self.cells = QCheckBox("Show cells")
        self.cells.setChecked(opts["cells"])
        self.cell_size = QDoubleSpinBox()
        self.cell_size.setRange(2.0, 300.0)
        self.cell_size.setSingleStep(5.0)
        self.cell_size.setDecimals(0)
        self.cell_size.setSuffix(" um")
        self.cell_size.setValue(float(opts["cell_size"]))
        self.cell_size.setToolTip("Radius of the sphere drawn for each cell.")
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
        self.cells.toggled.connect(self.cell_size.setEnabled)
        self.cell_size.setEnabled(opts["cells"])
        self.brain.toggled.connect(self.brain_alpha.setEnabled)
        self.brain_alpha.setEnabled(opts["brain"])

        form = QFormLayout()
        form.addRow("Structures", regions_row)
        form.addRow("", self.preview)
        form.addRow("Top regions by cell count", self.top)
        form.addRow("Structure opacity", self.alpha)
        form.addRow("Hemisphere", self.hemisphere)
        form.addRow("Whole-brain opacity", self.brain_alpha)
        form.addRow("Surface style", self.style)
        form.addRow("Starting view", self.view)
        form.addRow("Cell size (radius)", self.cell_size)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Open 3D view")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        for box in (self.slices, self.brain, self.axes, self.inset, self.title, self.cells):
            lay.addWidget(box)
        if cell_types:
            lay.addWidget(self.cell_types)
        hint = QLabel("In the 3D window, use the buttons on the left to snap to a view, hide "
                      "or show the title, or save an image (PNG at 2x window resolution).")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addWidget(buttons)
        self.resize(520, self.sizeHint().height())
        self._update_preview()

    def _on_pick_color(self) -> None:
        tokens = split_region_text(self.regions.text())
        if not tokens:
            return
        name, current = split_color(tokens[-1])
        rgb = parse_color(current) if current else None
        start = QColor(*(int(v * 255) for v in rgb)) if rgb else QColor("white")
        color = QColorDialog.getColor(start, self, f"Colour for {name}")
        if not color.isValid():
            return
        tokens[-1] = f"{name}:{color.name()}"
        self.regions.setText(" ".join(tokens))

    def _update_preview(self) -> None:
        queries = split_region_text(self.regions.text())
        if not queries:
            self.preview.setText("<i>No structures typed: only the top regions are shown.</i>")
            return
        found, missing = resolve_regions(queries, self.structures)
        lines = []
        for q, acr, name in found:
            name_q, color_text = split_color(q)
            line = f"{escape(name_q)} &rarr; <b>{escape(acr)}</b> ({escape(name)})"
            if color_text:
                rgb = parse_color(color_text)
                if rgb is None:
                    line += (f" <span style='color:#e06060'>unknown colour "
                             f"'{escape(color_text)}'</span>")
                else:
                    line += f" <span style='color:{_hex(rgb)}'>&#9632;&#9632;</span>"
            lines.append(line)
        for q in missing:
            hint = ", ".join(suggest_regions(q, self.structures))
            lines.append(f"<span style='color:#e06060'>{escape(split_color(q)[0])}: not found"
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
            "brain_alpha": round(self.brain_alpha.value(), 2),
            "style": self.style.currentData(),
            "axes": self.axes.isChecked(),
            "inset": self.inset.isChecked(),
            "title": self.title.isChecked(),
            "cell_size": self.cell_size.value(),
            "view": self.view.currentData(),
            "hidden_cell_types": hidden,
        }

    def cli_args(self) -> list[str]:
        o = self.options()
        args = ["--regions", str(o["top_regions"]), "--alpha", str(o["alpha"]),
                "--hemisphere", o["hemisphere"], "--brain-alpha", str(o["brain_alpha"]),
                "--style", o["style"], "--view", o["view"], "--cell-size", f"{o['cell_size']:g}"]
        found, _ = resolve_regions(split_region_text(o["structures"]), self.structures)
        if found:
            tokens = []
            for q, acr, _ in found:
                color_text = split_color(q)[1]
                rgb = parse_color(color_text) if color_text else None
                tokens.append(f"{acr}:{_hex(rgb)}" if rgb else acr)
            args += ["--structures", *tokens]
        if not o["slices"]:
            args.append("--no-slices")
        if not o["brain"]:
            args.append("--no-brain")
        if o["axes"]:
            args.append("--axes")
        if not o["inset"]:
            args.append("--no-inset")
        if not o["title"]:
            args.append("--no-title")
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
