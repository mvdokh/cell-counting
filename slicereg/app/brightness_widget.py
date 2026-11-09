"""Brightness levels for section and slide-overview images.

Section levels are shared across sections unless one has its own. Overview levels
(``overview_levels``) are separate: the overview equalizes each section first, then
these controls fine-tune the display.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel,
                            QPushButton, QVBoxLayout)
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider

from .layers import apply_levels, full_range, get_levels, layer_levels, store_levels

HIST_SAMPLE_PX = 1_000_000
GAMMA_RANGE = (0.2, 2.0)


def save_levels_on_change(project, layers, sid: int | None = None,
                          delay_ms: int = 400, key: str = "levels") -> QTimer:
    """Save the layers' levels (debounced) whenever they change in napari's controls.
    Keep the returned timer alive for as long as the layers are shown."""
    timer = QTimer()
    timer.setSingleShot(True)
    timer.setInterval(delay_ms)
    timer.timeout.connect(lambda: store_levels(project, layer_levels(layers), sid, key=key))

    def restart(_event=None) -> None:
        timer.start()

    for layer in layers:
        layer.events.contrast_limits.connect(restart)
        layer.events.gamma.connect(restart)
    return timer


def _layer_rgb(layer) -> tuple[float, float, float]:
    try:
        r, g, b = (float(v) for v in layer.colormap.colors[-1][:3])
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0.8, 0.8, 0.8
    return r, g, b


class _ChannelRow:
    """Visibility, histogram, range and gamma controls for one channel layer."""

    def __init__(self, widget: "BrightnessWidget", layer, values: np.ndarray):
        self.widget = widget
        self.layer = layer
        lo, hi = widget.range
        rgb = _layer_rgb(layer)
        hex_color = "#" + "".join(f"{int((0.45 + 0.55 * v) * 255):02x}" for v in rgb)

        self.show = QCheckBox(layer.name.split("(")[-1].rstrip(")"))
        self.show.setStyleSheet(f"QCheckBox {{ color: {hex_color}; font-weight: bold; }}")
        self.show.setChecked(layer.visible)
        self.show.toggled.connect(lambda on: setattr(layer, "visible", on))

        fig = Figure(figsize=(3.2, 0.9), layout="constrained")
        fig.patch.set_alpha(0)
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setStyleSheet("background: transparent")
        self.canvas.setFixedHeight(80)
        ax = fig.subplots()
        bins = np.linspace(lo, hi, 257)
        counts, edges = np.histogram(values, bins=bins)
        ax.stairs(np.maximum(counts, 0.8), edges, fill=True, color=rgb, alpha=0.6)
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.tick_params(labelsize=7, colors="0.6", length=2)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("0.5")
        ax.set_facecolor("none")
        self.curve_ax = ax.twinx()
        self.curve_ax.set_ylim(0, 1.02)
        self.curve_ax.set_axis_off()
        self.curve, = self.curve_ax.plot([], [], color="white", lw=1.2)
        self.lo_line = ax.axvline(lo, color="white", lw=0.8, ls="--")
        self.hi_line = ax.axvline(hi, color="white", lw=0.8, ls="--")

        decimals = 0 if widget.is_int else 3
        self.range = QLabeledDoubleRangeSlider(Qt.Horizontal)
        self.range.setRange(lo, hi)
        self.range.setDecimals(decimals)
        self.range.setSingleStep(1.0 if widget.is_int else (hi - lo) / 1000)
        self.range.setToolTip("Values at or below the left handle are black; at or above "
                              "the right handle, full brightness.")
        self.range.valueChanged.connect(self._on_range)

        self.gamma = QLabeledDoubleSlider(Qt.Horizontal)
        self.gamma.setRange(*GAMMA_RANGE)
        self.gamma.setDecimals(2)
        self.gamma.setSingleStep(0.01)
        self.gamma.setToolTip("Below 1 brightens dim signal, above 1 darkens it; the ends "
                              "of the range stay where they are.")
        self.gamma.valueChanged.connect(self._on_gamma)

        layer.events.contrast_limits.connect(self.update_from_layer)
        layer.events.gamma.connect(self.update_from_layer)
        layer.events.visible.connect(self._on_layer_visible)
        self.update_from_layer()

    def add_to(self, lay: QVBoxLayout) -> None:
        lay.addWidget(self.show)
        lay.addWidget(self.canvas)
        for label, w in (("Range", self.range), ("Gamma", self.gamma)):
            row = QHBoxLayout()
            text = QLabel(label)
            text.setFixedWidth(44)
            row.addWidget(text)
            row.addWidget(w, 1)
            lay.addLayout(row)

    def disconnect(self) -> None:
        for event, fn in ((self.layer.events.contrast_limits, self.update_from_layer),
                          (self.layer.events.gamma, self.update_from_layer),
                          (self.layer.events.visible, self._on_layer_visible)):
            try:
                event.disconnect(fn)
            except (TypeError, ValueError, RuntimeError):
                pass

    def _on_range(self, value) -> None:
        if self.widget.syncing:
            return
        lo, hi = (float(v) for v in value)
        apply_levels([self.layer], [{"min": lo, "max": hi, "gamma": self.layer.gamma}])
        if isinstance(self.layer, _LevelsProxy):
            self.update_from_layer()

    def _on_gamma(self, value: float) -> None:
        if self.widget.syncing:
            return
        self.layer.gamma = float(value)
        if isinstance(self.layer, _LevelsProxy):
            self.update_from_layer()

    def _on_layer_visible(self, _event=None) -> None:
        self.show.blockSignals(True)
        self.show.setChecked(self.layer.visible)
        self.show.blockSignals(False)

    def update_from_layer(self, _event=None) -> None:
        lo, hi = (float(v) for v in self.layer.contrast_limits)
        gamma = float(self.layer.gamma)
        self.widget.syncing = True
        try:
            r0, r1 = self.widget.range
            self.range.setRange(min(r0, lo), max(r1, hi))
            self.range.setValue((lo, hi))
            self.gamma.setValue(min(max(gamma, GAMMA_RANGE[0]), GAMMA_RANGE[1]))
        finally:
            self.widget.syncing = False
        self.lo_line.set_xdata([lo, lo])
        self.hi_line.set_xdata([hi, hi])
        r0, r1 = self.widget.range
        x = np.linspace(r0, r1, 200)
        self.curve.set_data(x, np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1) ** gamma)
        self.canvas.draw_idle()
        self.widget.changed()


class _LevelsProxy:
    """Minimal napari Image stand-in so BrightnessWidget can drive a custom apply."""

    def __init__(self, name, colormap, dtype, level: dict):
        from napari.utils.events import EmitterGroup

        self.name = name
        self.colormap = colormap
        lo, hi = float(level["min"]), float(level["max"])
        self.contrast_limits = (lo, hi)
        self.contrast_limits_range = full_range(dtype)
        self.gamma = float(level.get("gamma", 1.0))
        self.visible = True
        self.data = np.zeros((2, 2), dtype)
        self.events = EmitterGroup(source=self, contrast_limits=None, gamma=None, visible=None)


def proxy_layers_for(layers, levels, dtype) -> list:
    """Build proxy layers matching ``layers`` names/colormaps with the given levels."""
    return [_LevelsProxy(layer.name, layer.colormap, dtype, lv)
            for layer, lv in zip(layers, levels)]


class BrightnessWidget(QGroupBox):
    def __init__(self, view=None, *, project=None, layers=None, image=None,
                 sid: int | None = None, slice_entry: dict | None = None,
                 levels_key: str = "levels", title: str = "Brightness",
                 allow_separate: bool | None = None, custom_apply=None,
                 hint: str | None = None):
        super().__init__(title)
        if view is not None:
            project = view.project
            layers = view.image_layers
            image = view.image
            sid = view.sid
            slice_entry = view.slice
            if allow_separate is None:
                allow_separate = True
        if allow_separate is None:
            allow_separate = sid is not None and levels_key == "levels" and custom_apply is None
        self.view = view
        self.project = project
        self.sid = sid
        self.slice_entry = slice_entry if slice_entry is not None else {}
        self.layers = list(layers)
        self.levels_key = levels_key
        self.allow_separate = allow_separate
        self.custom_apply = custom_apply
        dtype = image.dtype
        self.is_int = bool(np.issubdtype(dtype, np.integer))
        self.range = full_range(dtype)
        self.syncing = False
        self._ready = False
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._save)

        self.separate = None
        if allow_separate:
            self.separate = QCheckBox("Separate levels for this section")
            self.separate.setChecked("levels" in self.slice_entry)
            self.separate.setToolTip(
                "Off: these levels are shared by every section.\n"
                "On: this section keeps its own levels (e.g. if it was imaged differently).\n"
                "The slide overview has its own brightness controls.")
            self.separate.toggled.connect(self._on_separate)

        step = max(1, int(np.ceil(np.sqrt(image.shape[0] * image.shape[1] / HIST_SAMPLE_PX))))
        sample = image[::step, ::step]
        tissue = sample.any(axis=2) if sample.ndim == 3 else sample > 0
        if not tissue.any():
            tissue = np.ones(sample.shape[:2], bool)
        n_ch = sample.shape[2] if sample.ndim == 3 else 1
        self._tissue_values = [
            (sample[..., i] if sample.ndim == 3 else sample)[tissue] for i in range(n_ch)]

        lay = QVBoxLayout(self)
        if hint is not None:
            hint_text = hint
        elif levels_key == "overview_levels" and custom_apply is not None:
            hint_text = ("Adjust this section's equalized preview on the overview. "
                         "Other sections are unchanged; Overview brightness below still "
                         "applies to the whole slide.")
        elif levels_key == "overview_levels":
            hint_text = ("Each section is equalized first so different laser settings "
                         "look similar; these controls adjust the overview display.")
        else:
            hint_text = "Levels are only changed by you; they are saved with the project."
        hint_label = QLabel(hint_text)
        hint_label.setWordWrap(True)
        lay.addWidget(hint_label)
        self.rows = []
        for i, layer in enumerate(self.layers):
            row = _ChannelRow(self, layer, self._tissue_values[min(i, len(self._tissue_values) - 1)])
            row.add_to(lay)
            self.rows.append(row)

        auto_row = QHBoxLayout()
        self.auto_btn = QPushButton("Auto")
        self.auto_btn.setToolTip("Set each channel's range from the tissue, clipping the "
                                 "given percentage of the darkest and brightest pixels.")
        self.auto_btn.clicked.connect(self.auto)
        self.clip = QDoubleSpinBox()
        self.clip.setRange(0.0, 10.0)
        self.clip.setDecimals(2)
        self.clip.setSingleStep(0.05)
        self.clip.setValue(0.1)
        self.clip.setSuffix(" % clip")
        auto_row.addWidget(self.auto_btn, 1)
        auto_row.addWidget(self.clip)
        lay.addLayout(auto_row)
        full = QPushButton("Full range (no adjustment)")
        full.setToolTip("Show the raw values: black = 0, full brightness = the image's maximum "
                        "possible value, gamma 1.")
        full.clicked.connect(self.full_range)
        lay.addWidget(full)
        if self.separate is not None:
            lay.addWidget(self.separate)
        self._ready = True

    def changed(self) -> None:
        if self._ready and not self.syncing:
            if self.custom_apply is not None:
                self.custom_apply(layer_levels(self.layers))
            self._save_timer.start()

    def _apply(self, levels: list[dict]) -> None:
        apply_levels(self.layers, levels)
        if self.custom_apply is not None:
            self.custom_apply(layer_levels(self.layers))

    def _save(self) -> None:
        levels = layer_levels(self.layers)
        if self.custom_apply is not None:
            store_levels(self.project, levels, self.sid, key=self.levels_key)
            return
        sid = None
        if self.allow_separate and self.separate is not None and self.separate.isChecked():
            sid = self.sid
        store_levels(self.project, levels, sid, key=self.levels_key)

    def auto(self) -> None:
        clip = float(self.clip.value())
        levels = []
        for layer, values in zip(self.layers, self._tissue_values):
            lo, hi = np.percentile(values, [clip, 100 - clip]) if values.size else self.range
            levels.append({"min": float(lo), "max": float(hi), "gamma": float(layer.gamma)})
        self._apply(levels)

    def full_range(self) -> None:
        lo, hi = self.range
        self._apply([{"min": lo, "max": hi, "gamma": 1.0}] * len(self.layers))

    def _on_separate(self, on: bool) -> None:
        if on:
            store_levels(self.project, layer_levels(self.layers), self.sid, key=self.levels_key)
            return
        self.slice_entry.pop("levels", None)
        self.project.save()
        apply_levels(self.layers, get_levels(self.project, len(self.layers),
                                             np.dtype(self.layers[0].data.dtype),
                                             key=self.levels_key))

    def disconnect_layers(self) -> None:
        self._save_timer.stop()
        for row in self.rows:
            row.disconnect()
