"""Top-level napari window: switches between the slide overview and one slice."""

from __future__ import annotations

from pathlib import Path

import napari
from qtpy.QtCore import QEvent, QObject, QTimer
from qtpy.QtWidgets import QApplication

from ..atlas import Atlas
from ..io import Project


class _CloseGuard(QObject):
    def __init__(self, app: "App"):
        super().__init__()
        self.app = app

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Close and self.app.view is not None:
            if not self.app.view.can_leave():
                event.ignore()
                return True
        return False


class App:
    def __init__(self, project: Project):
        self.project = project
        self.viewer = napari.Viewer(title=f"slicereg - {Path(project.data['source']).name}")
        self._atlas: Atlas | None = None
        self._docks = []
        self.view = None
        self._close_guard = _CloseGuard(self)
        qt_window = getattr(self.viewer.window, "_qt_window", None)
        if qt_window is not None:
            qt_window.installEventFilter(self._close_guard)
            qt_window.showMaximized()
        self.show_slide()

    @property
    def atlas(self) -> Atlas:
        if self._atlas is None:
            self.viewer.status = f"Loading atlas {self.project.data['atlas']}..."
            QApplication.processEvents()
            self._atlas = Atlas(self.project.data["atlas"])
        return self._atlas

    def add_dock(self, widget, name: str) -> None:
        dock = self.viewer.window.add_dock_widget(widget, name=name, area="right")
        self._docks.append(dock)

    def _set_view(self, view) -> None:
        if self.view is not None:
            self.view.deactivate()
        for dock in self._docks:
            self.viewer.window.remove_dock_widget(dock)
        self._docks = []
        self.viewer.layers.clear()
        self.view = view
        view.activate()
        self.viewer.reset_view()
        QTimer.singleShot(50, self.viewer.reset_view)

    def show_slide(self) -> None:
        from .folder_view import FolderView
        from .slide_view import SlideView

        if self.view is not None and not self.view.can_leave():
            return
        self._set_view(FolderView(self) if self.project.is_folder else SlideView(self))

    def open_slice(self, sid: int) -> None:
        from .slice_view import SliceView

        if self.view is not None and not self.view.can_leave():
            return
        self._set_view(SliceView(self, sid))


def run(path: str, project_dir: str | None = None) -> None:
    project = Project.open_or_create(path, project_dir)
    App(project)
    napari.run()
