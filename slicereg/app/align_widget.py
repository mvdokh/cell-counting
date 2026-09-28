"""Controls for fitting the atlas plane to a section."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                            QPushButton, QSlider, QVBoxLayout, QWidget)

# Approximate CCF AP coordinate of bregma, only used for the hint label.
BREGMA_AP_UM = 5400.0


def _spin(lo, hi, step, decimals, suffix):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


class AlignWidget(QWidget):
    FIELDS = ("ap_um", "pitch_deg", "yaw_deg", "rotation_deg", "scale_x", "scale_y", "tx", "ty")

    def __init__(self, view):
        super().__init__()
        self.view = view
        ap_max = float(view.atlas.extent_um[0])
        step = float(view.atlas.res[0])
        self.spins = {
            "ap_um": _spin(0, ap_max, step, 0, " um"),
            "pitch_deg": _spin(-30, 30, 0.5, 1, " deg"),
            "yaw_deg": _spin(-30, 30, 0.5, 1, " deg"),
            "rotation_deg": _spin(-180, 180, 0.5, 1, " deg"),
            "scale_x": _spin(0.001, 50, 0.004, 4, " px/um"),
            "scale_y": _spin(0.001, 50, 0.004, 4, " px/um"),
            "tx": _spin(-1e6, 1e6, 10, 1, " px"),
            "ty": _spin(-1e6, 1e6, 10, 1, " px"),
        }
        self.ap_slider = QSlider(Qt.Horizontal)
        self.ap_slider.setRange(0, int(ap_max / step))
        self.ap_slider.valueChanged.connect(lambda v: self.spins["ap_um"].setValue(v * step))
        self._step = step
        self.bregma = QLabel()
        self.lock = QCheckBox("Lock aspect ratio (scale X and Y together)")
        self.flip = QCheckBox("Mirror atlas left/right")

        form = QFormLayout()
        form.addRow("AP position", self.spins["ap_um"])
        form.addRow("", self.ap_slider)
        form.addRow("", self.bregma)
        form.addRow("Tilt up/down (pitch)", self.spins["pitch_deg"])
        form.addRow("Tilt left/right (yaw)", self.spins["yaw_deg"])
        form.addRow("Rotation", self.spins["rotation_deg"])
        form.addRow("Scale X", self.spins["scale_x"])
        form.addRow("Scale Y", self.spins["scale_y"])
        form.addRow("", self.lock)
        form.addRow("Centre X", self.spins["tx"])
        form.addRow("Centre Y", self.spins["ty"])
        form.addRow("", self.flip)
        self.spacing = _spin(-5000, 5000, 10, 0, " um")
        self.spacing.setValue(float(view.project.data.get("section_spacing_um", 0.0)))
        self.spacing.setToolTip("AP distance between consecutive slices on this slide; used to "
                                "guess the AP position of slices you have not aligned yet "
                                "(negative = next slice is more anterior).")
        self.spacing.valueChanged.connect(self._on_spacing)
        form.addRow("Section spacing", self.spacing)

        self.deepslice = QPushButton("Predict alignment with DeepSlice")
        self.deepslice.setToolTip(
            "Run DeepSlice on this section to guess its AP position, tilt, rotation and "
            "scale in the Allen atlas. Takes ~10-30 s; refine the result by hand afterwards.")
        self.deepslice_invert = QCheckBox("Invert image (dark tissue on light background)")
        self.deepslice_invert.setToolTip(
            "DeepSlice was trained mostly on brightfield sections. If the prediction is off "
            "for fluorescence images, try again with this ticked.")
        self.deepslice_undo = QPushButton("Undo DeepSlice")
        self.deepslice_undo.setEnabled(False)
        self.deepslice_status = QLabel("")
        self.deepslice_status.setWordWrap(True)
        ds_row = QHBoxLayout()
        ds_row.addWidget(self.deepslice, 1)
        ds_row.addWidget(self.deepslice_undo)

        self.autofit = QPushButton("Auto-fit to tissue outline")
        self.landmark_mode = QPushButton("Landmark mode (warp)")
        self.landmark_mode.setCheckable(True)
        self.clear_landmarks = QPushButton("Clear landmarks")
        self.n_landmarks = QLabel("0 landmarks")
        row = QHBoxLayout()
        row.addWidget(self.landmark_mode)
        row.addWidget(self.clear_landmarks)

        help_text = QLabel(
            "Shift+drag on the image moves the atlas.\n"
            "Landmark mode: press on an atlas feature and drag it onto the matching tissue "
            "feature; drag an existing landmark to adjust it, right-click to delete it.")
        help_text.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.addLayout(ds_row)
        lay.addWidget(self.deepslice_invert)
        lay.addWidget(self.deepslice_status)
        lay.addLayout(form)
        lay.addWidget(self.autofit)
        lay.addLayout(row)
        lay.addWidget(self.n_landmarks)
        lay.addWidget(help_text)
        lay.addStretch()

        self._last_scale = (1.0, 1.0)
        for spin in self.spins.values():
            spin.valueChanged.connect(self._on_edit)
        self.flip.toggled.connect(self._on_edit)
        self.deepslice.clicked.connect(
            lambda: view.run_deepslice(self.deepslice_invert.isChecked()))
        self.deepslice_undo.clicked.connect(view.undo_deepslice)
        self.autofit.clicked.connect(view.auto_fit)
        self.landmark_mode.toggled.connect(view.set_landmark_mode)
        self.clear_landmarks.clicked.connect(view.clear_landmarks)

    def load(self, al) -> None:
        for name, spin in self.spins.items():
            spin.blockSignals(True)
            spin.setValue(float(getattr(al, name)))
            spin.blockSignals(False)
        self.flip.blockSignals(True)
        self.flip.setChecked(bool(al.flip))
        self.flip.blockSignals(False)
        self._last_scale = (al.scale_x, al.scale_y)
        self._sync_ap_labels(al.ap_um)
        self.n_landmarks.setText(f"{len(al.landmarks)} landmarks")

    def set_deepslice_running(self, running: bool) -> None:
        self.deepslice.setEnabled(not running)
        self.deepslice.setText("DeepSlice running..." if running
                               else "Predict alignment with DeepSlice")

    def _on_spacing(self, value: float) -> None:
        self.view.project.data["section_spacing_um"] = float(value)
        self.view.project.save()

    def _sync_ap_labels(self, ap: float) -> None:
        self.ap_slider.blockSignals(True)
        self.ap_slider.setValue(int(round(ap / self._step)))
        self.ap_slider.blockSignals(False)
        self.bregma.setText(f"approx. {(BREGMA_AP_UM - ap) / 1000:+.2f} mm from bregma")

    def _on_edit(self, *_):
        al = self.view.alignment
        sx, sy = self.spins["scale_x"].value(), self.spins["scale_y"].value()
        if self.lock.isChecked():
            old_x, old_y = self._last_scale
            if sx != old_x and old_x > 0:
                sy = old_y * sx / old_x
            elif sy != old_y and old_y > 0:
                sx = old_x * sy / old_y
            for name, v in (("scale_x", sx), ("scale_y", sy)):
                self.spins[name].blockSignals(True)
                self.spins[name].setValue(v)
                self.spins[name].blockSignals(False)
        self._last_scale = (sx, sy)
        for name in self.FIELDS:
            setattr(al, name, float(self.spins[name].value()))
        al.scale_x, al.scale_y = sx, sy
        al.flip = self.flip.isChecked()
        self._sync_ap_labels(al.ap_um)
        self.view.alignment_changed()
