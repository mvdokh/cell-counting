"""Slide overview: review detected sections, crop them, and pick one to work on."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (QApplication, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                            QProgressDialog, QPushButton, QVBoxLayout, QWidget)

from ..atlas import atlas_structures
from ..detect import detect_sections
from ..export import collect_cells, export_project
from .layers import add_channels
from .render_dialog import RenderDialog


def _rect(b):
    x0, y0, x1, y1 = b
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


class SlideView:
    def __init__(self, app):
        self.app = app
        self.project = app.project
        self.viewer = app.viewer
        self.shapes = None

    def can_leave(self) -> bool:
        return True

    def activate(self) -> None:
        self.viewer.status = "Building slide thumbnail (first time only)..."
        QApplication.processEvents()
        thumb = self.project.thumbnail()
        f = self.project.data["thumb_factor"]
        add_channels(self.viewer, thumb, "slide", scale=(f, f), translate=((f - 1) / 2,) * 2)

        if self.project.slices:
            entries = [(s["id"], s["bbox"]) for s in self.project.slices]
        else:
            entries = [(-1, b) for b in self._detect()]
        self.shapes = self.viewer.add_shapes(
            [_rect(b) for _, b in entries] or None, shape_type="rectangle", name="sections",
            edge_color="yellow", face_color="transparent", edge_width=max(8, 2.5 * f),
            features={"sid": [sid for sid, _ in entries], "label": [""] * len(entries)},
            text={"string": "{label}", "color": "yellow", "size": 11, "anchor": "upper_left"},
        )
        self.shapes.feature_defaults = {"sid": -1, "label": "new"}
        self._build_dock()
        self._refresh_labels()
        self.viewer.mouse_double_click_callbacks.append(self._on_double_click)
        self.viewer.layers.selection.active = self.shapes
        self.shapes.mode = "select"
        self.viewer.status = "Review the section boxes, then click 'Crop & save sections'."

    def deactivate(self) -> None:
        if self._on_double_click in self.viewer.mouse_double_click_callbacks:
            self.viewer.mouse_double_click_callbacks.remove(self._on_double_click)

    def _detect(self) -> list[list[int]]:
        thumb = self.project.thumbnail()
        boxes, threshold, dark = detect_sections(thumb)
        self.project.data["threshold"] = threshold
        self.project.data["dark_background"] = dark
        self.project.save()
        f = self.project.data["thumb_factor"]
        h, w = self.project.slide.shape[:2]
        return [[min(v * f, lim) for v, lim in zip(b, (w, h, w, h))] for b in boxes]

    def _build_dock(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        help_text = QLabel(
            "1. Check the yellow boxes: drag to move/resize (select tool), draw new ones with "
            "the rectangle tool, delete with the Delete key.\n"
            "2. Click 'Crop & save sections'.\n"
            "3. Double-click a section (or a list entry) to align it and count cells.\n"
            "Box labels: id, 'A' = aligned, (n) = cells counted.")
        help_text.setWordWrap(True)
        lay.addWidget(help_text)
        for text, slot in [("Re-detect sections", self._on_redetect),
                           ("Crop && save sections", self._on_crop)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            lay.addWidget(b)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(
            lambda item: self._open(item.data(Qt.UserRole)))
        lay.addWidget(self.list)
        open_btn = QPushButton("Open selected section")
        open_btn.clicked.connect(self._on_open_selected)
        lay.addWidget(open_btn)
        for text, slot in [("Export all cells (CSV)", self._on_export),
                           ("3D view in brainrender", self._on_render)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            lay.addWidget(b)
        self.app.add_dock(w, "Slide")

    def _describe(self, sid: int) -> str:
        st = self.project.status(sid)
        name = self.project.get_slice(sid).get("name")
        parts = [f"slice {sid:02d}" + (f" ({name})" if name else "")]
        if not st["cropped"]:
            parts.append("not imported" if self.project.is_folder else "not cropped")
        if st["aligned"]:
            parts.append("aligned")
        if st["counted"]:
            cells = self.project.load_cells(sid)
            parts.append(f"{0 if cells is None else len(cells)} cells")
        return " | ".join(parts)

    def _short_label(self, sid: int) -> str:
        st = self.project.status(sid)
        label = self.project.get_slice(sid).get("name") or f"{sid:02d}"
        if st["aligned"]:
            label += " A"
        if st["counted"]:
            cells = self.project.load_cells(sid)
            label += f" ({0 if cells is None else len(cells)})"
        return label

    def _refresh_labels(self) -> None:
        feats = self.shapes.features.copy()
        known = {s["id"] for s in self.project.slices}
        feats["label"] = [self._short_label(int(s)) if int(s) in known else "new"
                          for s in feats["sid"]]
        self.shapes.features = feats
        self.shapes.refresh_text()
        self.list.clear()
        if not self.project.slices:
            item = QListWidgetItem("No sections saved yet - click 'Crop & save sections', "
                                   "or double-click a box to crop and open it.")
            item.setFlags(Qt.NoItemFlags)
            self.list.addItem(item)
        for s in self.project.slices:
            item = QListWidgetItem(self._describe(s["id"]))
            item.setData(Qt.UserRole, s["id"])
            self.list.addItem(item)

    def _current_entries(self) -> list[tuple[int, list[int]]]:
        h, w = self.project.slide.shape[:2]
        entries = []
        for verts, sid in zip(self.shapes.data, self.shapes.features["sid"]):
            y0, x0 = np.floor(verts.min(axis=0)).astype(int)
            y1, x1 = np.ceil(verts.max(axis=0)).astype(int)
            box = [max(0, x0), max(0, y0), min(w, x1), min(h, y1)]
            if box[2] - box[0] > 10 and box[3] - box[1] > 10:
                entries.append((int(sid), box))
        return entries

    def _on_redetect(self) -> None:
        old = self._current_entries()
        new = []
        for b in self._detect():
            match = max(old, key=lambda e: _iou(e[1], b), default=None)
            sid = match[0] if match is not None and _iou(match[1], b) > 0.5 else -1
            new.append((sid, b))
        self.shapes.data = []
        self.shapes.add_rectangles([_rect(b) for _, b in new])
        feats = self.shapes.features.copy()
        feats["sid"] = [sid for sid, _ in new]
        self.shapes.features = feats
        self._refresh_labels()

    def _on_crop(self) -> None:
        entries = self._current_entries()
        if not entries:
            QMessageBox.warning(None, "No sections", "Draw at least one section box first.")
            return
        to_crop = self.project.set_boxes(entries)
        dlg = QProgressDialog("Cropping sections...", None, 0, len(to_crop))
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        for i, sid in enumerate(to_crop):
            dlg.setValue(i)
            dlg.setLabelText(f"Cropping slice {sid:02d} ({i + 1}/{len(to_crop)})")
            QApplication.processEvents()
            self.project.crop_slice(sid)
        self.project.save()
        dlg.setValue(len(to_crop))
        self.shapes.data = []
        self.shapes.add_rectangles([_rect(s["bbox"]) for s in self.project.slices])
        feats = self.shapes.features.copy()
        feats["sid"] = [s["id"] for s in self.project.slices]
        self.shapes.features = feats
        self._refresh_labels()
        self.viewer.status = f"Saved {len(self.project.slices)} sections to {self.project.root}"

    def _has_unsaved_boxes(self) -> bool:
        saved = [(s["id"], s["bbox"]) for s in self.project.slices]
        if self._current_entries() != saved:
            return True
        return not all(self.project.status(sid)["cropped"] for sid, _ in saved)

    def _open(self, sid: int) -> None:
        QTimer.singleShot(0, lambda: self.app.open_slice(sid))

    def _open_at(self, y: float, x: float) -> None:
        """Open the section whose box contains (y, x), cropping unsaved boxes first."""
        if self._has_unsaved_boxes():
            self._on_crop()
        for s in self.project.slices:
            x0, y0, x1, y1 = s["bbox"]
            if x0 <= x < x1 and y0 <= y < y1:
                self._open(s["id"])
                return

    def _on_open_selected(self) -> None:
        selected = sorted(self.shapes.selected_data)
        if selected:
            y, x = self.shapes.data[selected[0]].mean(axis=0)
            self._open_at(y, x)
            return
        item = self.list.currentItem()
        if item is not None and item.data(Qt.UserRole) is not None:
            if self._has_unsaved_boxes():
                self._on_crop()
            self._open(item.data(Qt.UserRole))
            return
        QMessageBox.information(None, "Nothing selected",
                                "Click a section box on the slide (select tool) or an entry "
                                "in the list first, or just double-click a section.")

    def _on_double_click(self, viewer, event) -> None:
        if self.shapes is None or self.shapes.mode.startswith("add"):
            return
        y, x = event.position[-2:]
        QTimer.singleShot(0, lambda: self._open_at(y, x))

    def _on_export(self) -> None:
        summary = export_project(self.project)
        QMessageBox.information(None, "Export", summary)

    def _on_render(self) -> None:
        cells = collect_cells(self.project)
        dlg = RenderDialog(self.project, atlas_structures(self.project.data["atlas"]),
                           list(dict.fromkeys(cells["cell_type"])),
                           parent=getattr(self.viewer.window, "_qt_window", None))
        if not dlg.exec():
            return
        dlg.save()
        subprocess.Popen([sys.executable, "-m", "slicereg", "render", str(self.project.root),
                          *dlg.cli_args()])
        self.viewer.status = "Opening brainrender in a separate window..."
