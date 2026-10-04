"""Overview of a folder project: every section image at its position on the slide."""

from __future__ import annotations

import cv2
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QApplication, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                            QProgressDialog, QPushButton, QVBoxLayout, QWidget)

from ..folder import CONFIG_NAME, grid_layout
from .layers import add_channels
from .slide_view import SlideView, _rect

GAP_UM = 600.0
OVERVIEW_PX = 500  # overview width of the largest section, in screen-independent pixels


def _overview_tile(preview: np.ndarray, mask: np.ndarray, pad: float = 0.03) -> np.ndarray:
    """The section's tissue only, cropped to it, each channel stretched to 0-255.

    Stretching per section evens out sections imaged with different laser/gain
    settings, so they look alike side by side."""
    img = preview.astype(np.float32)
    if mask.shape != preview.shape[:2] or not mask.any():
        mask = np.ones(preview.shape[:2], bool)
    out = np.zeros(img.shape, np.uint8)
    for c in range(img.shape[2]):
        lo, hi = np.percentile(img[..., c][mask], [0.5, 99.7])
        out[..., c] = np.clip((img[..., c] - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    out[~mask] = 0
    ys, xs = np.nonzero(mask)
    p = int(pad * max(np.ptp(xs), np.ptp(ys)))
    y0, y1 = max(0, ys.min() - p), min(mask.shape[0], ys.max() + 1 + p)
    x0, x1 = max(0, xs.min() - p), min(mask.shape[1], xs.max() + 1 + p)
    return out[y0:y1, x0:x1]


class FolderView(SlideView):
    def activate(self) -> None:
        self._import_missing()
        self._boxes = self._build_overview()
        self._build_dock()
        self._refresh_labels()
        self.viewer.mouse_double_click_callbacks.append(self._on_double_click)
        self.viewer.status = "Double-click a section to align it and count cells."

    def _import_missing(self) -> None:
        to_import = self.project.sync_folder()
        if not to_import:
            return
        dlg = QProgressDialog("Importing sections...", None, 0, len(to_import))
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        failed = []
        for i, sid in enumerate(to_import):
            s = self.project.get_slice(sid)
            dlg.setValue(i)
            dlg.setLabelText(f"Importing {s['file']} ({i + 1}/{len(to_import)})")
            QApplication.processEvents()
            try:
                self.project.import_section(sid)
            except Exception as e:  # one unreadable file shouldn't block the others
                failed.append(f"{s['file']}: {e}")
        self.project.save()
        dlg.setValue(len(to_import))
        if failed:
            QMessageBox.warning(None, "Some sections could not be read", "\n".join(failed))

    def _build_overview(self) -> dict[int, list[float]]:
        """Show every imported section at true size in its slide cell; return the
        sections' boxes as {sid: [x0, y0, x1, y1]} in overview microns."""
        sizes, previews = {}, {}
        for s in self.project.slices:
            if not self.project.status(s["id"])["cropped"]:
                continue
            preview = self.project.load_preview(s["id"])
            tile = _overview_tile(preview, self.project.load_mask(s["id"]))
            um_per_preview_px = (s.get("pixel_um") or 1.0) * s["size"][0] / preview.shape[1]
            sizes[s["id"]] = (s["col"], s["row"], tile.shape[1] * um_per_preview_px,
                              tile.shape[0] * um_per_preview_px)
            previews[s["id"]] = tile
        if not sizes:
            return {}
        corners = grid_layout(sizes, GAP_UM)
        um_px = max(max(w, h) for _, _, w, h in sizes.values()) / OVERVIEW_PX
        boxes = {sid: [x, y, x + sizes[sid][2], y + sizes[sid][3]]
                 for sid, (x, y) in corners.items()}
        width = int(np.ceil(max(b[2] for b in boxes.values()) / um_px)) + 1
        height = int(np.ceil(max(b[3] for b in boxes.values()) / um_px)) + 1
        n_ch = max(p.shape[2] for p in previews.values())
        dtype = next(iter(previews.values())).dtype
        canvas = np.zeros((height, width, n_ch), dtype)
        for sid, (x0, y0, x1, y1) in boxes.items():
            c0, r0 = int(round(x0 / um_px)), int(round(y0 / um_px))
            cw, rh = max(1, int(round((x1 - x0) / um_px))), max(1, int(round((y1 - y0) / um_px)))
            small = cv2.resize(previews[sid], (cw, rh), interpolation=cv2.INTER_AREA)
            small = small.reshape(rh, cw, -1)
            canvas[r0:r0 + rh, c0:c0 + cw, :small.shape[2]] = small
        add_channels(self.viewer, canvas, "slide", channels=self.project.data.get("channels"),
                     limits=(0, 255), scale=(um_px, um_px), translate=(um_px / 2, um_px / 2))
        sids = list(boxes)
        self.shapes = self.viewer.add_shapes(
            [_rect(boxes[sid]) for sid in sids], shape_type="rectangle", name="sections",
            edge_color="yellow", face_color="transparent", edge_width=2.5 * um_px,
            features={"sid": sids, "label": [""] * len(sids)},
            text={"string": "{label}", "color": "yellow", "size": 11, "anchor": "upper_left"})
        self.shapes.mode = "pan_zoom"
        return boxes

    def _build_dock(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        folder = self.project.data["source"]
        help_text = QLabel(
            f"Sections from {folder}, laid out by their column/row on the slide.\n"
            "Double-click a section (or a list entry) to align it and count cells.\n"
            "Labels: name, 'A' = aligned, (n) = cells counted.\n"
            f"File names, channels and section order are set in {CONFIG_NAME} in that "
            "folder; click 'Rescan folder' after editing it or adding images.")
        help_text.setWordWrap(True)
        lay.addWidget(help_text)
        rescan = QPushButton("Rescan folder")
        rescan.clicked.connect(self.app.show_slide)
        lay.addWidget(rescan)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda item: self._open(item.data(Qt.UserRole)))
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

    def _refresh_labels(self) -> None:
        if self.shapes is not None:
            super()._refresh_labels()
            return
        self.list.clear()
        for s in self.project.slices:
            item = QListWidgetItem(self._describe(s["id"]))
            item.setData(Qt.UserRole, s["id"])
            self.list.addItem(item)

    def _open(self, sid) -> None:
        if sid is None:
            return
        if not self.project.status(sid)["cropped"]:
            QMessageBox.warning(None, "Not imported",
                                f"{self.project.get_slice(sid)['file']} could not be read.")
            return
        super()._open(sid)

    def _section_at(self, y: float, x: float) -> int | None:
        for sid, (x0, y0, x1, y1) in self._boxes.items():
            if x0 <= x < x1 and y0 <= y < y1:
                return sid
        return None

    def _on_double_click(self, viewer, event) -> None:
        y, x = event.position[-2:]
        sid = self._section_at(y, x)
        if sid is not None:
            self._open(sid)

    def _on_open_selected(self) -> None:
        item = self.list.currentItem()
        if item is not None and item.data(Qt.UserRole) is not None:
            self._open(item.data(Qt.UserRole))
            return
        QMessageBox.information(None, "Nothing selected",
                                "Pick a section in the list first, or double-click a section.")
