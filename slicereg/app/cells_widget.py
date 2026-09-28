"""Controls for clicking cells on a section."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QComboBox, QHBoxLayout, QInputDialog, QLabel, QPushButton,
                            QSlider, QSpinBox, QVBoxLayout, QWidget)


class CellsWidget(QWidget):
    def __init__(self, view):
        super().__init__()
        self.view = view
        self.types = QComboBox()
        self.types.currentTextChanged.connect(view.select_cell_type)
        add_type = QPushButton("New cell type...")
        add_type.clicked.connect(self._on_add_type)
        row = QHBoxLayout()
        row.addWidget(self.types, 1)
        row.addWidget(add_type)

        self.add_mode = QPushButton("Add cells (click)")
        self.select_mode = QPushButton("Select / delete")
        self.delete = QPushButton("Delete selected")
        self.add_mode.clicked.connect(lambda: view.set_cell_mode("add"))
        self.select_mode.clicked.connect(lambda: view.set_cell_mode("select"))
        self.delete.clicked.connect(view.delete_selected_cells)
        modes = QHBoxLayout()
        for b in (self.add_mode, self.select_mode, self.delete):
            modes.addWidget(b)

        size_max = max(8, int(round(view.default_cell_size * 4)))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(2, size_max)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(2, size_max)
        self.size_spin.setSuffix(" px")
        self._set_size_controls(view.cell_size)
        self.size_slider.valueChanged.connect(self._on_size_slider)
        self.size_spin.valueChanged.connect(self._on_size_spin)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Cell marker size"))
        size_row.addWidget(self.size_slider, 1)
        size_row.addWidget(self.size_spin)

        self.counts = QLabel()
        self.counts.setWordWrap(True)
        help_text = QLabel(
            "Click on each cell to mark it; Shift+drag moves the view without adding a cell, "
            "and the scroll wheel zooms. In 'Select / delete' mode, click or drag a box "
            "around marks and press Delete to remove them. Click Save when done.")
        help_text.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Cell type"))
        lay.addLayout(row)
        lay.addLayout(modes)
        lay.addLayout(size_row)
        lay.addWidget(self.counts)
        lay.addWidget(help_text)
        lay.addStretch()

    def set_types(self, names: list[str], current: str | None = None) -> None:
        self.types.blockSignals(True)
        self.types.clear()
        self.types.addItems(names)
        if current:
            self.types.setCurrentText(current)
        self.types.blockSignals(False)

    def set_counts(self, counts: dict[str, int]) -> None:
        lines = [f"{name}: {n}" for name, n in counts.items()]
        self.counts.setText("Counts - " + ", ".join(lines) + f"  (total {sum(counts.values())})")

    def _set_size_controls(self, px: float) -> None:
        px = int(round(px))
        for w in (self.size_slider, self.size_spin):
            w.blockSignals(True)
            w.setValue(px)
            w.blockSignals(False)

    def _on_size_slider(self, value: int) -> None:
        self._set_size_controls(value)
        self.view.set_cell_size(value)

    def _on_size_spin(self, value: int) -> None:
        self._set_size_controls(value)
        self.view.set_cell_size(value)

    def _on_add_type(self) -> None:
        name, ok = QInputDialog.getText(self, "New cell type", "Name (e.g. cFos+, GFP+):")
        name = name.strip()
        if ok and name:
            self.view.add_cell_type(name)
