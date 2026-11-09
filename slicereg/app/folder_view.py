"""Overview of a folder project: every section image at its position on the slide."""

from __future__ import annotations

import cv2
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel,
                            QListWidget, QListWidgetItem, QMessageBox, QProgressDialog,
                            QPushButton, QVBoxLayout, QWidget)

from ..experiment import (collect_experiment_cells, experiment_cell_types, experiment_root,
                          open_slide_project, sibling_slides, slide_number,
                          sync_experiment_cell_types, sync_experiment_flips)
from ..folder import CONFIG_NAME, grid_layout
from .brightness_widget import BrightnessWidget, proxy_layers_for, save_levels_on_change
from .layers import add_channels, equalize_channels, get_levels, remap_levels
from .slide_view import SlideView, _rect

GAP_UM = 600.0
OVERVIEW_PX = 500  # overview width of the largest section, in screen-independent pixels


def _overview_tile(preview: np.ndarray, mask: np.ndarray, pad: float = 0.03) -> np.ndarray:
    """The section's tissue only, cropped to it and equalized per channel.

    Equalization maps each channel's tissue percentiles to the full dtype range so
    sections imaged with different laser settings look about the same; the overview
    brightness panel then adjusts the shared display."""
    if mask.shape != preview.shape[:2] or not mask.any():
        mask = np.ones(preview.shape[:2], bool)
    out = preview.copy()
    out[~mask] = 0
    ys, xs = np.nonzero(mask)
    p = int(pad * max(np.ptp(xs), np.ptp(ys)))
    y0, y1 = max(0, ys.min() - p), min(mask.shape[0], ys.max() + 1 + p)
    x0, x1 = max(0, xs.min() - p), min(mask.shape[1], xs.max() + 1 + p)
    tile = out[y0:y1, x0:x1]
    return equalize_channels(tile, mask[y0:y1, x0:x1])


class FolderView(SlideView):
    def activate(self) -> None:
        self._import_missing()
        self._boxes = self._build_overview()
        self._build_dock()
        self._refresh_labels()
        self.list.currentItemChanged.connect(self._on_list_selection)
        self.viewer.mouse_double_click_callbacks.append(self._on_double_click)
        self.viewer.mouse_drag_callbacks.append(self._on_click)
        self.viewer.status = ("Click a section to select it, double-click to align it and "
                              "count cells.")

    def deactivate(self) -> None:
        if getattr(self, "brightness_widget", None) is not None:
            self.brightness_widget.disconnect_layers()
            self.brightness_widget = None
        if self._on_click in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.remove(self._on_click)
        super().deactivate()

    def _import_missing(self) -> None:
        to_import = self.project.sync_folder()
        if to_import:
            self._import(to_import)
        skipped = getattr(self.project, "skipped_files", None)
        if skipped:
            QMessageBox.information(
                None, "Files not used",
                "These files are at a slide position that already has an image, so they "
                f"were not used:\n{chr(10).join(skipped)}\n\nTo use one of them instead, "
                f"set 'exclude' in {CONFIG_NAME} to a pattern matching the other file and "
                "click 'Rescan folder' (that section's alignment and cells would need redoing).")
        error = getattr(self.project, "hemisphere_error", None)
        if error:
            QMessageBox.warning(None, "Hemispheres file not used",
                                f"{error}\n\nSection orientation "
                                "was left as it was.")
        if experiment_root(self.project) is not None:
            sync_experiment_cell_types(self.project)
        sync_experiment_flips(self.project)
        self.viewer.status = "Checking section orientation..."
        QApplication.processEvents()
        changed = self.project.apply_hemispheres()
        if changed:
            names = ", ".join(self.project.get_slice(sid).get("name", str(sid))
                              for sid in changed)
            QMessageBox.information(
                None, "Sections mirrored",
                f"Updated {len(changed)} section(s) from the hemispheres file / flip "
                f"manifest so orientation matches: {names}.\n\n"
                "Alignments, landmarks and cells were mirrored along with the images; "
                "export again to update the CSVs.")

    def _import(self, to_import: list[int]) -> None:
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
            self._tiles = {}
            self._placements = {}
            return {}
        corners = grid_layout(sizes, GAP_UM)
        um_px = max(max(w, h) for _, _, w, h in sizes.values()) / OVERVIEW_PX
        boxes = {sid: [x, y, x + sizes[sid][2], y + sizes[sid][3]]
                 for sid, (x, y) in corners.items()}
        width = int(np.ceil(max(b[2] for b in boxes.values()) / um_px)) + 1
        height = int(np.ceil(max(b[3] for b in boxes.values()) / um_px)) + 1
        n_ch = max(p.shape[2] for p in previews.values())
        dtype = next(iter(previews.values())).dtype
        self._tiles = {}
        self._placements = {}
        canvas = np.zeros((height, width, n_ch), dtype)
        for sid, (x0, y0, x1, y1) in boxes.items():
            c0, r0 = int(round(x0 / um_px)), int(round(y0 / um_px))
            cw, rh = max(1, int(round((x1 - x0) / um_px))), max(1, int(round((y1 - y0) / um_px)))
            small = cv2.resize(previews[sid], (cw, rh), interpolation=cv2.INTER_AREA)
            small = small.reshape(rh, cw, -1)
            self._tiles[sid] = small
            self._placements[sid] = (r0, c0, rh, cw)
            levels = get_levels(self.project, small.shape[2], dtype, sid,
                                key="overview_levels")
            painted = remap_levels(small, levels)
            canvas[r0:r0 + rh, c0:c0 + cw, :painted.shape[2]] = painted
        self._overview_image = canvas
        self.image_layers = add_channels(
            self.viewer, canvas, "slide",
            channels=self.project.data.get("channels"),
            levels=get_levels(self.project, n_ch, dtype, key="overview_levels"),
            scale=(um_px, um_px), translate=(um_px / 2, um_px / 2))
        self._levels_timer = save_levels_on_change(
            self.project, self.image_layers, key="overview_levels")
        sids = list(boxes)
        self._edge_um = 2.5 * um_px
        self.shapes = self.viewer.add_shapes(
            [_rect(boxes[sid]) for sid in sids], shape_type="rectangle", name="sections",
            edge_color="yellow", face_color="transparent", edge_width=self._edge_um,
            features={"sid": sids, "label": [""] * len(sids)},
            text={"string": "{label}", "color": "yellow", "size": 11, "anchor": "upper_left"})
        self.shapes.mode = "pan_zoom"
        return boxes

    def _paint_section_tile(self, sid: int, levels: list[dict] | None = None) -> None:
        """Rewrite one section's overview pixels from its equalized tile + levels."""
        tile = getattr(self, "_tiles", {}).get(sid)
        place = getattr(self, "_placements", {}).get(sid)
        if tile is None or place is None or not getattr(self, "image_layers", None):
            return
        if levels is None:
            levels = get_levels(self.project, tile.shape[2], tile.dtype, sid,
                                key="overview_levels")
        painted = remap_levels(tile, levels)
        r0, c0, rh, cw = place
        canvas = self._overview_image
        canvas[r0:r0 + rh, c0:c0 + cw, :painted.shape[2]] = painted
        # Refresh napari layers from the shared canvas (one layer per channel).
        if len(self.image_layers) == 1:
            self.image_layers[0].data = canvas[..., 0]
        else:
            for i, layer in enumerate(self.image_layers):
                layer.data = canvas[..., i]

    def select(self, sid: int) -> None:
        """Highlight a section on the overview and in the list."""
        for row in range(self.list.count()):
            if self.list.item(row).data(Qt.UserRole) == sid:
                self.list.setCurrentRow(row)
                break
        self._highlight(sid)

    def _highlight(self, sid: int | None) -> None:
        if self.shapes is None or not len(self.shapes.data):
            return
        sids = [int(s) for s in self.shapes.features["sid"]]
        self.shapes.edge_color = [[0, 1, 1, 1] if s == sid else [1, 1, 0, 1] for s in sids]
        self.shapes.edge_width = [self._edge_um * (2.5 if s == sid else 1) for s in sids]
        name = self.project.get_slice(sid).get("name") if sid is not None else None
        if name:
            flipped = " (flipped)" if self.project.get_slice(sid).get("user_flipped") else ""
            self.viewer.status = (f"Selected {name}{flipped}: brightness sliders below are for "
                                  "this section; double-click / Open to align and count.")
        self._bind_brightness(sid)

    def _bind_brightness(self, sid: int | None) -> None:
        """Point the existing brightness sliders at the selected section (or the whole slide)."""
        host = getattr(self, "_brightness_host", None)
        if host is None or not getattr(self, "image_layers", None):
            return
        if getattr(self, "_brightness_sid", object()) == sid and self.brightness_widget is not None:
            return
        if self.brightness_widget is not None:
            self.brightness_widget.disconnect_layers()
            self.brightness_widget.setParent(None)
            self.brightness_widget.deleteLater()
            self.brightness_widget = None
        while host.count():
            item = host.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._brightness_sid = sid
        if sid is not None and sid in getattr(self, "_tiles", {}):
            tile = self._tiles[sid]
            name = self.project.get_slice(sid).get("name") or f"slice {sid}"
            levels = get_levels(self.project, tile.shape[2], tile.dtype, sid,
                                key="overview_levels")
            proxies = proxy_layers_for(self.image_layers, levels, tile.dtype)
            self.brightness_widget = BrightnessWidget(
                project=self.project, layers=proxies, image=tile, sid=sid,
                slice_entry=self.project.get_slice(sid),
                levels_key="overview_levels",
                title=f"Brightness ({name})",
                allow_separate=False,
                custom_apply=lambda lv, s=sid: self._paint_section_tile(s, lv),
                hint=("These sliders adjust only the selected section's equalized preview. "
                      "Click empty space or clear the list selection for whole-slide brightness."))
        else:
            self.brightness_widget = BrightnessWidget(
                project=self.project, layers=self.image_layers, image=self._overview_image,
                levels_key="overview_levels", title="Brightness (all sections)",
                allow_separate=False,
                hint=("Select a section to make these sliders adjust that section's preview. "
                      "With nothing selected they adjust the whole-slide display."))
        host.addWidget(self.brightness_widget)

    def _on_list_selection(self, item, _previous=None) -> None:
        if item is not None and item.data(Qt.UserRole) is not None:
            self._highlight(item.data(Qt.UserRole))
        else:
            self._highlight(None)

    def _on_click(self, viewer, event):
        """Select the section under a click; dragging still pans the view."""
        if event.button != 1:
            return
        start = np.asarray(event.pos, dtype=float)
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = dragged or np.hypot(*(np.asarray(event.pos, float) - start)) > 4
            yield
        if not dragged:
            sid = self._section_at(*event.position[-2:])
            if sid is not None:
                self.select(sid)

    def _build_dock(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        folder = self.project.data["source"]
        exp = experiment_root(self.project)
        slides = sibling_slides(self.project)
        if exp is not None and len(slides) > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Slide"))
            self.slide_combo = QComboBox()
            current = slide_number(self.project)
            for entry in slides:
                n = entry["slide"]
                mark = "" if entry["project"] is not None else " (new)"
                self.slide_combo.addItem(f"{n}{mark}", entry)
                if n == current:
                    self.slide_combo.setCurrentIndex(self.slide_combo.count() - 1)
            self.slide_combo.currentIndexChanged.connect(self._on_slide_changed)
            row.addWidget(self.slide_combo, 1)
            lay.addLayout(row)

        rescan = QPushButton("Rescan folder")
        rescan.clicked.connect(self.app.show_slide)
        lay.addWidget(rescan)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda item: self._open(item.data(Qt.UserRole)))
        lay.addWidget(self.list)
        open_btn = QPushButton("Open selected section")
        open_btn.clicked.connect(self._on_open_selected)
        lay.addWidget(open_btn)
        flip_btn = QPushButton("Flip selected section horizontally")
        flip_btn.setToolTip("Mirror the selected section's image left-right. Use the "
                            "checkbox below to choose what happens to an existing alignment.")
        flip_btn.clicked.connect(self._on_flip_selected)
        lay.addWidget(flip_btn)
        self.mirror_alignment = QCheckBox("Also mirror the alignment (and cell atlas coords)")
        self.mirror_alignment.setChecked(True)
        self.mirror_alignment.setToolTip(
            "On: the atlas outline stays on the same tissue after the flip.\n"
            "Off: keep the current alignment as it is in the image; only the section "
            "image (and cell marks) flip.")
        lay.addWidget(self.mirror_alignment)
        for text, slot in [("Plot AP && pitch (all slides)", self._on_plot),
                           ("Plot cells by region (all slides)", self._on_regions_plot),
                           ("Export all cells (CSV)", self._on_export),
                           ("3D view in brainrender (all slides)", self._on_render)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            lay.addWidget(b)
        self._brightness_host = QVBoxLayout()
        lay.addLayout(self._brightness_host)
        self.brightness_widget = None
        self._brightness_sid = object()  # force first bind
        if getattr(self, "image_layers", None):
            self._bind_brightness(None)
        help = (
            f"Sections from {folder}, laid out by their column/row on the slide.\n"
            "Click a section to select it; double-click it (or a list entry) to align it "
            "and count cells.\n"
            "Labels: name, 'F' = flipped, 'A' = aligned, (n) = cells counted.\n"
            "Sections are equalized first; select one and the brightness sliders above "
            "adjust that section's preview (with nothing selected they adjust the whole "
            "slide). Opening a section still has its own brightness.\n"
            "Flip horizontally mirrors the selected section in slide and slice views; "
            "flips are listed in slicereg_experiment.json (or project.json). The checkbox "
            "chooses whether an existing atlas alignment is mirrored too.\n"
        )
        if exp is not None and len(slides) > 1:
            help += (
                f"This slide is part of {exp.name}: use the slide menu to switch. "
                "Cell types, AP/pitch, region plots and brainrender are shared across "
                "every slide (after the last section of slide N comes the first of N+1).\n"
            )
        help += (
            f"File names, channels and section order are set in {CONFIG_NAME} in that "
            "folder; click 'Rescan folder' after editing it or adding images."
        )
        help_text = QLabel(help)
        help_text.setWordWrap(True)
        lay.addWidget(help_text)
        self.app.add_dock(w, "Slide")

    def _on_slide_changed(self, index: int) -> None:
        if index < 0 or not hasattr(self, "slide_combo"):
            return
        entry = self.slide_combo.itemData(index)
        if entry is None:
            return
        if entry["slide"] == slide_number(self.project):
            return
        self.viewer.status = f"Opening slide {entry['slide']}..."
        QApplication.processEvents()
        try:
            proj = open_slide_project(entry)
        except Exception as e:
            QMessageBox.warning(None, "Could not open slide", str(e))
            # put the combo back
            current = slide_number(self.project)
            for i in range(self.slide_combo.count()):
                if self.slide_combo.itemData(i)["slide"] == current:
                    self.slide_combo.blockSignals(True)
                    self.slide_combo.setCurrentIndex(i)
                    self.slide_combo.blockSignals(False)
                    break
            return
        self.app.switch_project(proj)

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

    def _selected_sid(self) -> int | None:
        item = self.list.currentItem()
        if item is not None and item.data(Qt.UserRole) is not None:
            return int(item.data(Qt.UserRole))
        return None

    def _on_flip_selected(self) -> None:
        sid = self._selected_sid()
        if sid is None:
            QMessageBox.information(None, "Nothing selected",
                                    "Click a section on the overview or in the list first.")
            return
        if not self.project.status(sid)["cropped"]:
            QMessageBox.warning(None, "Not imported",
                                "Import this section before flipping it.")
            return
        name = self.project.get_slice(sid).get("name") or f"slice {sid}"
        mirror_al = self.mirror_alignment.isChecked()
        has_al = self.project.status(sid)["aligned"]
        if has_al and not mirror_al:
            ok = QMessageBox.question(
                None, "Keep alignment?",
                f"Flip {name} but leave its atlas alignment where it is?\n\n"
                "The outline will no longer match the tissue until you re-align.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ok != QMessageBox.Yes:
                return
        self.project.flip_section_horizontal(sid, mirror_alignment=mirror_al)
        self.viewer.status = (f"Flipped {name}"
                              + (" (alignment mirrored)." if mirror_al and has_al
                                 else " (alignment kept)." if has_al else "."))
        # Rebuild the overview so the flipped preview shows.
        self.app.show_slide()
        # Reselect after rebuild.
        QApplication.processEvents()
        if isinstance(self.app.view, FolderView):
            self.app.view.select(sid)

    def _on_plot(self) -> None:
        from .positions_plot import PositionsPlot

        PositionsPlot(
            self.project, self._open,
            parent=getattr(self.viewer.window, "_qt_window", None),
            open_experiment_section=self.app.open_experiment_section).exec()

    def _on_regions_plot(self) -> None:
        from .regions_plot import RegionsPlot

        cells = collect_experiment_cells(self.project)
        if not len(cells):
            QMessageBox.information(None, "No cells",
                                    "Count cells on at least one section first.")
            return
        # Drop the slide column for ranking helpers that expect CELL_COLUMNS.
        plot_cells = cells.drop(columns=["slide"], errors="ignore")
        title = "Cells by region"
        if experiment_root(self.project) is not None:
            title += " (all slides)"
        RegionsPlot(plot_cells, title=title,
                    parent=getattr(self.viewer.window, "_qt_window", None),
                    save_dir=self.project.root).exec()

    def _on_render(self) -> None:
        import subprocess
        import sys

        from ..atlas import atlas_structures
        from .render_dialog import RenderDialog

        cells = collect_experiment_cells(self.project)
        counted = ([] if cells.empty or "cell_type" not in cells.columns
                   else list(dict.fromkeys(cells["cell_type"].astype(str))))
        types = list(dict.fromkeys([*experiment_cell_types(self.project), *counted]))
        dlg = RenderDialog(self.project, atlas_structures(self.project.data["atlas"]),
                           types,
                           parent=getattr(self.viewer.window, "_qt_window", None))
        if not dlg.exec():
            return
        dlg.save()
        args = [sys.executable, "-m", "slicereg", "render", str(self.project.root),
                *dlg.cli_args()]
        if experiment_root(self.project) is not None:
            args.append("--all-slides")
        subprocess.Popen(args)
        self.viewer.status = "Opening brainrender in a separate window..."
