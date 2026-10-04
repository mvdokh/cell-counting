"""One section: atlas alignment and cell counting."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
from napari.layers.points._points_constants import Mode as PointsMode
from napari.layers.points._points_mouse_bindings import add as add_point_on_click
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import QProcess, QTimer
from qtpy.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QPushButton, QTabWidget,
                            QVBoxLayout, QWidget)

from ..atlas import boundaries, label_bbox
from ..export import map_cells
from ..overlay import draw_outlines, save_overlay_png, warp_atlas
from ..transform import Alignment, SliceTransform, initial_alignment
from .align_widget import AlignWidget
from .cells_widget import CellsWidget
from .display_widget import DisplayWidget
from .layers import (CELL_COLORS, DISPLAY_DEFAULTS, add_channels, channel_index,
                     contrast_limits, outline_colormap)

CELL_PREFIX = "cells: "
PICK_RADIUS_SCREEN_PX = 12
SMOOTH_OUTLINE_DEBOUNCE_MS = 150


class SliceView:
    def __init__(self, app, sid: int):
        self.app = app
        self.viewer = app.viewer
        self.project = app.project
        self.sid = sid
        self.slice = self.project.get_slice(sid)
        self.ds = int(self.slice["ds"])
        self.atlas = app.atlas
        self.image = self.project.load_image(sid)
        self.preview = self.project.load_preview(sid)
        saved = self.project.load_alignment(sid)
        self.alignment = Alignment.from_dict(saved) if saved else self._initial_alignment()
        self.tf = SliceTransform(self.alignment)
        self.labels = np.zeros(self.preview.shape[:2], np.uint32)
        self.dirty = False
        self.landmark_mode = False
        self.cell_layers: dict[str, object] = {}
        self._display = {**DISPLAY_DEFAULTS, **self.project.data.get("display", {})}
        self._default_cell_size = max(15.0, self.image.shape[1] / 150)
        self._cell_size = float(self._display.get("cell_size") or self._default_cell_size)
        self._side_dx = self.image.shape[1] * 1.05
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._refresh_overlay)
        # Smooth outlines re-fit a contour through every atlas region, which is too
        # slow to recompute on every tick while a slider is being dragged. Keep the
        # cheap raster outline live during dragging and only compute the smooth one
        # once the alignment/display value has settled.
        self._smooth_timer = QTimer()
        self._smooth_timer.setSingleShot(True)
        self._smooth_timer.timeout.connect(self._refresh_smooth_outline)
        self._active = False
        self._deepslice = None
        self._deepslice_running = False
        self._deepslice_crop = (0.0, 0.0, *self._image_size())
        self._deepslice_log: list[str] = []
        self._pre_deepslice: Alignment | None = None

    @property
    def display(self) -> dict:
        return self._display

    @property
    def default_cell_size(self) -> float:
        return self._default_cell_size

    @property
    def cell_size(self) -> float:
        return self._cell_size

    # ---------------------------------------------------------------- setup
    def _initial_alignment(self) -> Alignment:
        ids = [s["id"] for s in self.project.slices]
        aligned = [(i, self.project.load_alignment(i)) for i in ids
                   if i != self.sid and self.project.status(i)["aligned"]]
        spacing = float(self.project.data.get("section_spacing_um", 0.0))
        pos = ids.index(self.sid)
        ap, pitch, yaw, rot, flip = self.atlas.extent_um[0] / 2, 0.0, 0.0, 0.0, False
        if aligned:
            nb_id, nb = min(aligned, key=lambda e: abs(ids.index(e[0]) - pos))
            ap = nb["ap_um"] + spacing * (pos - ids.index(nb_id))
            pitch, yaw, rot, flip = nb["pitch_deg"], nb["yaw_deg"], nb["rotation_deg"], nb["flip"]
        ap = float(np.clip(ap, 0, self.atlas.extent_um[0]))
        pixel_um = self.slice.get("pixel_um")
        return initial_alignment(self.atlas, self._image_size(), self._mask_bbox(), ap,
                                 pitch, yaw, rot, flip,
                                 scale=1.0 / pixel_um if pixel_um else None)

    def _image_size(self) -> tuple[int, int]:
        return int(self.image.shape[1]), int(self.image.shape[0])

    def _mask_bbox(self):
        bb = label_bbox(self.project.load_mask(self.sid))
        return None if bb is None else [v * self.ds for v in bb]

    def activate(self) -> None:
        v = self.viewer
        d = self._display
        off = (self.ds - 1) / 2
        overlay_kw = dict(scale=(self.ds, self.ds), translate=(off, off))
        add_channels(v, self.image, "section", channels=self.project.data.get("channels"))
        tmpl = self.atlas.template
        tmpl_clim = contrast_limits(tmpl[tmpl.shape[0] // 2])
        self.template_layer = v.add_image(
            np.zeros(self.preview.shape[:2], np.float32), name="atlas template",
            colormap="gray", blending="additive", opacity=0.6, visible=False,
            contrast_limits=tmpl_clim, **overlay_kw)
        self.regions_layer = v.add_labels(
            self.labels, name="atlas regions", opacity=0.3, visible=d["regions"],
            colormap=DirectLabelColormap(color_dict=self.atlas.color_dict()), **overlay_kw)
        self.outline_layer = v.add_image(
            np.zeros(self.preview.shape[:2], np.float32), name="atlas outlines",
            colormap=outline_colormap(d["color"]), contrast_limits=(0, 1),
            blending="translucent", visible=d["outlines"], **overlay_kw)

        side_kw = dict(scale=(self.ds, self.ds), translate=(off, off + self._side_dx))
        self.side_template_layer = v.add_image(
            np.zeros(self.preview.shape[:2], np.float32), name="average template (side)",
            colormap="gray", opacity=1.0, visible=d["side_by_side"],
            contrast_limits=tmpl_clim, **side_kw)
        self.side_outline_layer = v.add_image(
            np.zeros(self.preview.shape[:2], np.float32), name="atlas outlines (side)",
            colormap=outline_colormap(d["color"]), contrast_limits=(0, 1),
            blending="translucent", visible=d["side_by_side"] and d["outlines"], **side_kw)

        self.cursor = v.add_points(
            np.empty((0, 2)), name="cursor", size=max(20, self.image.shape[1] / 80),
            face_color="transparent", border_color="cyan", border_width=0.15,
            symbol="cross", visible=False)
        self.cursor.editable = False

        self.landmarks = v.add_points(
            np.empty((0, 2)), name="landmarks", size=max(20, self.image.shape[1] / 120),
            face_color="yellow", border_color="black", symbol="cross")
        self.landmarks.mouse_drag_callbacks.append(self._on_landmark_drag)
        self._sync_landmark_points()
        self._load_cells()

        self._build_dock()
        self.align_widget.load(self.alignment)
        v.mouse_move_callbacks.append(self._on_mouse_move)
        self._refresh_overlay()
        self._refresh_smooth_outline()
        self._on_tab(self.tabs.currentIndex())
        self.dirty = False
        self._active = True

    def deactivate(self) -> None:
        self._active = False
        self._timer.stop()
        self._smooth_timer.stop()
        if self._deepslice_running:
            self._deepslice.kill()
        if self._on_mouse_move in self.viewer.mouse_move_callbacks:
            self.viewer.mouse_move_callbacks.remove(self._on_mouse_move)

    def _build_dock(self) -> None:
        ids = [s["id"] for s in self.project.slices]
        pos = ids.index(self.sid)
        w = QWidget()
        lay = QVBoxLayout(w)
        name = self.slice.get("name")
        title = QLabel(f"<b>Slice {self.sid:02d}{f' - {name}' if name else ''}</b> "
                       f"({pos + 1} of {len(ids)})")
        lay.addWidget(title)
        nav = QHBoxLayout()
        for text, target in (("< Prev", pos - 1), ("Slide", None), ("Next >", pos + 1)):
            b = QPushButton(text)
            if target is None:
                b.clicked.connect(self.app.show_slide)
            elif 0 <= target < len(ids):
                b.clicked.connect(lambda _=False, t=ids[target]: self.app.open_slice(t))
            else:
                b.setEnabled(False)
            nav.addWidget(b)
        lay.addLayout(nav)
        save = QPushButton("Save (alignment + cells)")
        save.clicked.connect(self.save)
        lay.addWidget(save)
        self.hover = QLabel(" ")
        self.hover.setWordWrap(True)
        lay.addWidget(self.hover)

        self.display_widget = DisplayWidget(self)
        lay.addWidget(self.display_widget)

        self.tabs = QTabWidget()
        self.align_widget = AlignWidget(self)
        self.cells_widget = CellsWidget(self)
        self.tabs.addTab(self.align_widget, "1. Align to atlas")
        self.tabs.addTab(self.cells_widget, "2. Count cells")
        self.tabs.currentChanged.connect(self._on_tab)
        if self.project.status(self.sid)["aligned"]:
            self.tabs.setCurrentIndex(1)
        lay.addWidget(self.tabs)
        self.cells_widget.set_types(list(self.cell_layers), self._current_type)
        self._update_counts()
        self.app.add_dock(w, f"Slice {self.sid:02d}")

    # ------------------------------------------------------------ alignment
    def alignment_changed(self) -> None:
        self.dirty = True
        if not self._timer.isActive():
            self._timer.start(20)
        self._schedule_smooth_outline()

    def _refresh_overlay(self) -> None:
        """Cheap, always-live refresh: atlas sampling, region colours, raster outline.

        Kept fast enough to run on every tick while a slider is being dragged (unlike
        the smooth outline, see ``_refresh_smooth_outline``).
        """
        self.tf = SliceTransform(self.alignment)
        labels, tmpl = warp_atlas(self.tf, self.atlas, self.preview.shape[:2], self.ds)
        self.labels = labels
        self.regions_layer.data = labels
        self.template_layer.data = tmpl
        self.side_template_layer.data = tmpl
        if not self._display.get("smooth", True):
            outline = self._raster_outline(labels)
            self.outline_layer.data = outline
            self.side_outline_layer.data = outline

    def _raster_outline(self, labels: np.ndarray) -> np.ndarray:
        outlines = cv2.dilate(boundaries(labels).astype(np.uint8), np.ones((2, 2), np.uint8))
        return outlines.astype(np.float32)

    def _schedule_smooth_outline(self) -> None:
        if self._display.get("smooth", True):
            self._smooth_timer.start(SMOOTH_OUTLINE_DEBOUNCE_MS)

    def _refresh_smooth_outline(self) -> None:
        """Sub-pixel contour outline: fits every region boundary, which is too slow
        to run on every tick, so this only runs once the alignment/display settings
        have stopped changing for ``SMOOTH_OUTLINE_DEBOUNCE_MS``."""
        if not self._display.get("smooth", True):
            return
        d = self._display
        al = self.alignment
        contours_um = self.atlas.plane_contours(al.ap_um, al.pitch_deg, al.yaw_deg,
                                                 d.get("sigma", 1.0))
        off = (self.ds - 1) / 2
        contours_prev = [(self.tf.plane_to_image(c) - off) / self.ds for c in contours_um]
        outline = draw_outlines(contours_prev, self.preview.shape[:2], width=d.get("width", 2.0))
        self.outline_layer.data = outline
        self.side_outline_layer.data = outline

    # ------------------------------------------------------------- display
    def set_outlines_visible(self, visible: bool) -> None:
        self._set_display(outlines=visible)

    def set_regions_visible(self, visible: bool) -> None:
        self._set_display(regions=visible)

    def set_smooth_outlines(self, smooth: bool) -> None:
        self._set_display(smooth=smooth)

    def set_outline_sigma(self, sigma: float) -> None:
        self._set_display(sigma=float(sigma))

    def set_outline_width(self, width: float) -> None:
        self._set_display(width=float(width))

    def set_outline_color(self, rgb) -> None:
        self._set_display(color=[float(v) for v in rgb])
        cmap = outline_colormap(self._display["color"])
        self.outline_layer.colormap = cmap
        self.side_outline_layer.colormap = cmap

    def set_side_by_side(self, on: bool) -> None:
        self._set_display(side_by_side=on)
        QTimer.singleShot(50, self.viewer.reset_view)

    def _set_display(self, **kwargs) -> None:
        self._display.update(kwargs)
        self.project.data["display"] = dict(self._display)
        self.project.save()
        self._apply_visibility()
        self._display_changed()

    def _apply_visibility(self) -> None:
        d = self._display
        self.outline_layer.visible = d["outlines"]
        self.regions_layer.visible = d["regions"]
        self.side_template_layer.visible = d["side_by_side"]
        self.side_outline_layer.visible = d["side_by_side"] and d["outlines"]

    def _display_changed(self) -> None:
        if not self._timer.isActive():
            self._timer.start(20)
        self._schedule_smooth_outline()

    def auto_fit(self) -> None:
        al = self.alignment
        new = initial_alignment(self.atlas, self._image_size(), self._mask_bbox(), al.ap_um,
                                al.pitch_deg, al.yaw_deg, al.rotation_deg, al.flip)
        new.landmarks = al.landmarks
        self.alignment = new
        self.align_widget.load(new)
        self.alignment_changed()

    def run_deepslice(self, invert: bool = False) -> None:
        from .. import deepslice

        if not deepslice.supports_atlas(self.atlas.name):
            QMessageBox.information(
                None, "DeepSlice", "DeepSlice only predicts positions in the Allen mouse atlas "
                f"(allen_mouse_*); this project uses {self.atlas.name}.")
            return
        if not deepslice.is_installed():
            QMessageBox.information(
                None, "DeepSlice is not installed",
                "DeepSlice runs in its own Python environment. Set it up once from a "
                "terminal (with the slicereg environment active):\n\n"
                "    slicereg deepslice-setup\n\n"
                "This downloads TensorFlow, DeepSlice and its model weights (~2 GB).")
            return
        if self._deepslice_running:
            return
        d = self.project.slice_dir(self.sid)
        png, out = d / "deepslice_input.png", d / "deepslice.json"
        out.unlink(missing_ok=True)
        ch = channel_index(self.project, self.project.data.get("align_channel"))
        mask = self.project.load_mask(self.sid)
        x0, y0, x1, y1 = deepslice.write_input_image(
            self.preview if ch is None else self.preview[..., [ch]], png, invert,
            mask=mask if mask.shape == self.preview.shape[:2] else None)
        w, h = self._image_size()
        sx, sy = w / self.preview.shape[1], h / self.preview.shape[0]
        self._deepslice_crop = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        cmd = deepslice.worker_command(png, out)
        proc = QProcess()
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._on_deepslice_output(proc))
        proc.finished.connect(lambda code, _status: self._on_deepslice_finished(code, out))
        proc.errorOccurred.connect(
            lambda err: err == QProcess.ProcessError.FailedToStart
            and self._on_deepslice_finished(-1, out))
        # Kept referenced after it finishes: dropping it inside its own signal can crash Qt.
        self._deepslice = proc
        self._deepslice_running = True
        self._deepslice_log = []
        self.align_widget.set_deepslice_running(True)
        self.align_widget.deepslice_status.setText("Starting DeepSlice...")
        proc.start(cmd[0], cmd[1:])

    def _on_deepslice_output(self, proc) -> None:
        text = bytes(proc.readAllStandardOutput()).decode(errors="replace")
        self._deepslice_log.append(text)
        lines = [ln.strip() for ln in text.replace("\r", "\n").splitlines()
                 if ln.strip() and "\x1b" not in ln and "WARNING" not in ln]
        if lines and self._active:
            self.align_widget.deepslice_status.setText(lines[-1][:200])

    def _on_deepslice_finished(self, code: int, out) -> None:
        from .. import deepslice

        self._deepslice_running = False
        if not self._active:
            return
        self.align_widget.set_deepslice_running(False)
        status = self.align_widget.deepslice_status
        if code != 0 or not out.exists():
            status.setText("DeepSlice failed, see the terminal for details.")
            print("".join(self._deepslice_log)[-5000:])
            return
        anchoring = deepslice.uncrop_anchoring(json.loads(out.read_text()),
                                               self._deepslice_crop, self._image_size())
        new = deepslice.anchoring_to_alignment(anchoring, self.atlas.name,
                                               self.atlas.plane_size_um, self._image_size())
        old = self.alignment
        if new.flip != old.flip:
            new.flip, new.yaw_deg = old.flip, -new.yaw_deg
        self._pre_deepslice = old
        self.alignment = new
        self.align_widget.load(new)
        self._sync_landmark_points()
        self.alignment_changed()
        self.align_widget.deepslice_undo.setEnabled(True)
        status.setText(f"DeepSlice: AP {new.ap_um:.0f} um, pitch {new.pitch_deg:+.1f}, "
                       f"yaw {new.yaw_deg:+.1f} deg. Refine by hand, then Save.")

    def undo_deepslice(self) -> None:
        if self._pre_deepslice is None:
            return
        self.alignment, self._pre_deepslice = self._pre_deepslice, None
        self.align_widget.load(self.alignment)
        self._sync_landmark_points()
        self.alignment_changed()
        self.align_widget.deepslice_undo.setEnabled(False)
        self.align_widget.deepslice_status.setText("Restored the alignment from before DeepSlice.")

    def set_landmark_mode(self, on: bool) -> None:
        self.landmark_mode = on
        self.landmarks.cursor = "crosshair" if on else "standard"
        self.viewer.status = ("Landmark mode: drag atlas features onto the tissue"
                              if on else "Landmark mode off")

    def clear_landmarks(self) -> None:
        self.alignment.landmarks = []
        self._sync_landmark_points()
        self.alignment_changed()

    def _sync_landmark_points(self) -> None:
        lm = np.asarray(self.alignment.landmarks, dtype=float).reshape(-1, 4)
        self.landmarks.data = lm[:, [3, 2]]
        if hasattr(self, "align_widget"):
            self.align_widget.n_landmarks.setText(f"{len(lm)} landmarks")

    def _nearest_landmark(self, yx) -> int | None:
        lm = np.asarray(self.alignment.landmarks, dtype=float).reshape(-1, 4)
        if not len(lm):
            return None
        d = np.hypot(lm[:, 3] - yx[0], lm[:, 2] - yx[1])
        i = int(np.argmin(d))
        return i if d[i] * self.viewer.camera.zoom <= PICK_RADIUS_SCREEN_PX else None

    def _on_landmark_drag(self, layer, event):
        if event.type != "mouse_press":
            return
        start = np.asarray(event.position[-2:], dtype=float)
        al = self.alignment
        if "Shift" in event.modifiers and event.button == 1:
            layer.mouse_pan = False
            t0 = (al.tx, al.ty)
            lm0 = [list(p) for p in al.landmarks]
            yield
            while event.type == "mouse_move":
                dy, dx = np.asarray(event.position[-2:]) - start
                al.tx, al.ty = float(t0[0] + dx), float(t0[1] + dy)
                al.landmarks = [[s, t, float(x + dx), float(y + dy)] for s, t, x, y in lm0]
                self._sync_landmark_points()
                self.align_widget.load(al)
                self.alignment_changed()
                yield
            layer.mouse_pan = True
            return
        if not self.landmark_mode:
            return
        idx = self._nearest_landmark(start)
        if event.button == 2:
            if idx is not None:
                del al.landmarks[idx]
                self._sync_landmark_points()
                self.alignment_changed()
            return
        if event.button != 1:
            return
        layer.mouse_pan = False
        if idx is None:
            s, t = self.tf.image_to_plane([[start[1], start[0]]])[0]
            al.landmarks.append([float(s), float(t), float(start[1]), float(start[0])])
            idx = len(al.landmarks) - 1
            self._sync_landmark_points()
        yield
        while event.type == "mouse_move":
            y, x = event.position[-2:]
            al.landmarks[idx][2:] = [float(x), float(y)]
            self._sync_landmark_points()
            self.alignment_changed()
            yield
        layer.mouse_pan = True

    # ---------------------------------------------------------------- cells
    def _load_cells(self) -> None:
        df = self.project.load_cells(self.sid)
        types = ["cell"] if df is None or not len(df) else list(dict.fromkeys(df["cell_type"]))
        for name in types:
            sub = None if df is None else df[df["cell_type"] == name]
            yx = np.empty((0, 2)) if sub is None else sub[["y_px", "x_px"]].to_numpy(float)
            self._add_cell_layer(name, yx)
        self._current_type = types[0]

    def _add_cell_layer(self, name: str, yx: np.ndarray):
        color = CELL_COLORS[len(self.cell_layers) % len(CELL_COLORS)]
        layer = self.viewer.add_points(
            yx, name=CELL_PREFIX + name, size=self._cell_size,
            face_color=color, border_color="white", border_width=0.1)
        layer.current_face_color = color
        layer.current_size = self._cell_size
        layer._drag_modes = {**type(layer)._drag_modes, PointsMode.ADD: self._add_cell_or_pan}
        layer.events.data.connect(self._on_cells_changed)
        self.cell_layers[name] = layer
        return layer

    def _add_cell_or_pan(self, layer, event):
        """Add-mode click handler: Shift+drag pans the view instead of adding a cell."""
        if "Shift" not in event.modifiers:
            yield from add_point_on_click(layer, event)
            return
        cam = self.viewer.camera
        c0 = np.asarray(cam.center, dtype=float)
        p0 = np.asarray(event.pos, dtype=float)
        yield
        while event.type == "mouse_move":
            dx, dy = (np.asarray(event.pos, dtype=float) - p0) / cam.zoom
            c = c0.copy()
            c[-1] -= dx
            c[-2] -= dy
            cam.center = tuple(c)
            yield

    def set_cell_size(self, px: float) -> None:
        self._cell_size = float(px)
        for layer in self.cell_layers.values():
            layer.size = self._cell_size
            layer.current_size = self._cell_size
        self._display["cell_size"] = self._cell_size
        self.project.data["display"] = dict(self._display)
        self.project.save()

    def add_cell_type(self, name: str) -> None:
        if name not in self.cell_layers:
            self._add_cell_layer(name, np.empty((0, 2)))
        self.cells_widget.set_types(list(self.cell_layers), name)
        self.select_cell_type(name)

    def select_cell_type(self, name: str) -> None:
        if name in self.cell_layers:
            self._current_type = name
            if self.tabs.currentIndex() == 1:
                self.set_cell_mode("add")

    def set_cell_mode(self, mode: str) -> None:
        layer = self.cell_layers[self._current_type]
        self.viewer.layers.selection.active = layer
        layer.mode = mode

    def delete_selected_cells(self) -> None:
        layer = self.cell_layers[self._current_type]
        if layer.selected_data:
            layer.remove_selected()

    def _on_cells_changed(self, event=None) -> None:
        action = getattr(event, "action", None)
        if action is None or str(action) in ("added", "removed", "changed"):
            if action is not None and str(action) == "added":
                self._remove_offsection_cells()
            self.dirty = True
            self._update_counts()

    def _remove_offsection_cells(self) -> None:
        """Drop any cell marked outside the section image (e.g. on the side panel)."""
        h, w = self.image.shape[:2]
        for layer in self.cell_layers.values():
            data = np.asarray(layer.data, dtype=float)
            if not len(data):
                continue
            y, x = data[:, 0], data[:, 1]
            keep = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            if not keep.all():
                layer.data = data[keep]

    def _update_counts(self) -> None:
        if hasattr(self, "cells_widget"):
            self.cells_widget.set_counts({n: len(l.data) for n, l in self.cell_layers.items()})

    # ------------------------------------------------------------- general
    def _on_tab(self, index: int) -> None:
        if index == 0:
            self.landmarks.visible = True
            self.viewer.layers.selection.active = self.landmarks
            self.landmarks.mode = "pan_zoom"
            for layer in self.cell_layers.values():
                layer.mode = "pan_zoom"
        else:
            self.align_widget.landmark_mode.setChecked(False)
            self.landmarks.visible = False
            self.set_cell_mode("add")

    def _on_mouse_move(self, viewer, event) -> None:
        y, x = event.position[-2:]
        h, w = self.image.shape[:2]
        side_on = self._display.get("side_by_side", False)
        dx = self._side_dx
        mirror = None
        if 0 <= x < w and 0 <= y < h:
            mx = x
            if side_on:
                mirror = (y, mx + dx)
        elif side_on and 0 <= x - dx < w and 0 <= y < h:
            mx = x - dx
            mirror = (y, mx)
        else:
            if side_on:
                self.cursor.visible = False
            return
        i = min(int(mx // self.ds), self.labels.shape[1] - 1)
        j = min(int(y // self.ds), self.labels.shape[0] - 1)
        rid = int(self.labels[j, i])
        al = self.alignment
        p = self.atlas.plane_to_3d(self.tf.image_to_plane([[mx, y]]), al.ap_um, al.pitch_deg,
                                   al.yaw_deg)[0]
        self.hover.setText(f"{self.atlas.acronym(rid)} - {self.atlas.region_name(rid)}\n"
                           f"AP {p[0]:.0f}  DV {p[1]:.0f}  ML {p[2]:.0f} um")
        if mirror is not None:
            self.cursor.data = np.array([mirror])
            self.cursor.visible = True

    def save(self) -> None:
        al = self.alignment
        self.project.save_alignment(self.sid, al.to_dict())
        self._refresh_overlay()
        save_overlay_png(self.project.slice_dir(self.sid) / "overlay.png", self.preview,
                         self.labels)
        frames = []
        for name, layer in self.cell_layers.items():
            yx = np.asarray(layer.data, dtype=float).reshape(-1, 2)
            frames.append(map_cells(self.atlas, al, yx[:, ::-1], [name] * len(yx), self.sid))
        had_cells = self.project.status(self.sid)["counted"]
        n = sum(len(f) for f in frames)
        if n or had_cells or self.tabs.currentIndex() == 1:
            df = pd.concat([f for f in frames if len(f)] or frames[:1], ignore_index=True)
            self.project.save_cells(self.sid, df)
        self.dirty = False
        self.viewer.status = f"Saved slice {self.sid:02d}: alignment and {n} cells"

    def can_leave(self) -> bool:
        if not self.dirty:
            return True
        ans = QMessageBox.question(
            None, "Unsaved changes", f"Save changes to slice {self.sid:02d}?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if ans == QMessageBox.Save:
            self.save()
            return True
        return ans == QMessageBox.Discard
