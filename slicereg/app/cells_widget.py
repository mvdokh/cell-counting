"""Controls for clicking cells on a section."""

from __future__ import annotations

from qtpy.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QLabel,
                            QPushButton, QVBoxLayout, QWidget)


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

        self.show_outlines = QCheckBox("Show atlas outlines")
        self.show_outlines.setChecked(True)
        self.show_outlines.toggled.connect(lambda v: view.set_layer_visible("atlas outlines", v))
        self.show_regions = QCheckBox("Show region colours")
        self.show_regions.toggled.connect(lambda v: view.set_layer_visible("atlas regions", v))

        self.counts = QLabel()
        self.counts.setWordWrap(True)
        help_text = QLabel(
            "Click on each cell to mark it. In 'Select / delete' mode, click or drag a box "
            "around marks and press Delete to remove them. Click Save when done.")
        help_text.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Cell type"))
        lay.addLayout(row)
        lay.addLayout(modes)
        lay.addWidget(self.show_outlines)
        lay.addWidget(self.show_regions)
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

    def _on_add_type(self) -> None:
        name, ok = QInputDialog.getText(self, "New cell type", "Name (e.g. cFos+, GFP+):")
        name = name.strip()
        if ok and name:
            self.view.add_cell_type(name)
