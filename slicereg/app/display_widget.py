"""Shared display controls: atlas outline style, region colours, side-by-side template."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QCheckBox, QColorDialog, QGroupBox, QHBoxLayout, QLabel,
                            QPushButton, QSlider, QVBoxLayout)


def _rgb_to_qcolor(rgb) -> QColor:
    r, g, b = (int(round(v * 255)) for v in rgb)
    return QColor(r, g, b)


class DisplayWidget(QGroupBox):
    def __init__(self, view):
        super().__init__("Display")
        self.view = view
        d = view.display

        self.show_outlines = QCheckBox("Show atlas outlines")
        self.show_outlines.setChecked(d["outlines"])
        self.show_outlines.toggled.connect(view.set_outlines_visible)

        self.show_regions = QCheckBox("Show region colours")
        self.show_regions.setChecked(d["regions"])
        self.show_regions.toggled.connect(view.set_regions_visible)

        self.smooth = QCheckBox("Smooth outlines")
        self.smooth.setChecked(d["smooth"])
        self.smooth.toggled.connect(view.set_smooth_outlines)

        self.sigma_label = QLabel()
        self.sigma = QSlider(Qt.Horizontal)
        self.sigma.setRange(0, 30)  # sigma * 10, i.e. 0.0 - 3.0 atlas voxels of smoothing
        self.sigma.setValue(int(round(d["sigma"] * 10)))
        self.sigma.valueChanged.connect(self._on_sigma)

        self.width_label = QLabel()
        self.width = QSlider(Qt.Horizontal)
        self.width.setRange(1, 16)  # width * 2, i.e. 0.5 - 8.0 px in 0.5 px steps
        self.width.setValue(int(round(d["width"] * 2)))
        self.width.valueChanged.connect(self._on_width)

        self.color_btn = QPushButton("Outline colour...")
        self._set_color_swatch(d["color"])
        self.color_btn.clicked.connect(self._on_color)

        self.side_by_side = QCheckBox("Show average template side by side")
        self.side_by_side.setChecked(d["side_by_side"])
        self.side_by_side.toggled.connect(view.set_side_by_side)

        lay = QVBoxLayout(self)
        lay.addWidget(self.show_outlines)
        lay.addWidget(self.show_regions)
        lay.addWidget(self.smooth)

        sigma_row = QHBoxLayout()
        sigma_row.addWidget(QLabel("Smoothing"))
        sigma_row.addWidget(self.sigma, 1)
        sigma_row.addWidget(self.sigma_label)
        lay.addLayout(sigma_row)

        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Line width"))
        width_row.addWidget(self.width, 1)
        width_row.addWidget(self.width_label)
        lay.addLayout(width_row)

        lay.addWidget(self.color_btn)
        lay.addWidget(self.side_by_side)

        self._update_labels(d["sigma"], d["width"])

    def _update_labels(self, sigma: float, width: float) -> None:
        self.sigma_label.setText(f"{sigma:.1f}")
        self.width_label.setText(f"{width:.1f} px")

    def _on_sigma(self, value: int) -> None:
        sigma = value / 10.0
        self.sigma_label.setText(f"{sigma:.1f}")
        self.view.set_outline_sigma(sigma)

    def _on_width(self, value: int) -> None:
        width = value / 2.0
        self.width_label.setText(f"{width:.1f} px")
        self.view.set_outline_width(width)

    def _set_color_swatch(self, rgb) -> None:
        c = _rgb_to_qcolor(rgb)
        self.color_btn.setStyleSheet(
            f"QPushButton {{ border-left: 14px solid {c.name()}; padding-left: 6px; }}")

    def _on_color(self) -> None:
        current = _rgb_to_qcolor(self.view.display["color"])
        color = QColorDialog.getColor(current, self, "Outline colour")
        if color.isValid():
            rgb = (color.redF(), color.greenF(), color.blueF())
            self._set_color_swatch(rgb)
            self.view.set_outline_color(rgb)
