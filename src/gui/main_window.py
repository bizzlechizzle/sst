"""Main application window - Social Screenshot Tool."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox,
    QSplitter, QGroupBox, QCheckBox, QSpinBox, QSlider,
    QTabWidget, QScrollArea, QGridLayout, QMessageBox, QStatusBar,
    QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QSettings
from PySide6.QtGui import QPixmap, QImage, QFont

from ..core.models import Project, Frame, Category
from ..core.config import VIDEO_EXTENSIONS, DEFAULT_QUOTAS, JPEG_QUALITY
from ..analysis.pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)


# =============================================================================
# WORKER THREADS
# =============================================================================

class AnalysisWorker(QThread):
    """Background thread for video analysis."""

    progress = Signal(int, str)  # percent, message
    finished = Signal(object)    # Project result
    error = Signal(str)          # error message

    def __init__(self, folder_path: Path):
        super().__init__()
        self.folder_path = folder_path
        self.pipeline = None

    def run(self):
        """Run analysis in background."""
        try:
            self.pipeline = AnalysisPipeline()
            self.pipeline.set_progress_callback(
                lambda pct, msg: self.progress.emit(pct, msg)
            )

            project = self.pipeline.analyze_project(self.folder_path)
            self.finished.emit(project)

        except Exception as e:
            logger.exception("Analysis failed")
            self.error.emit(str(e))


class ExportWorker(QThread):
    """Background thread for exporting frames."""

    progress = Signal(int, int, str)  # current, total, path
    finished = Signal(list)            # list of paths
    error = Signal(str)

    def __init__(self, frames: list, output_folder: Path, presets: list,
                 quality: int = 90, max_dimension: int = None, lut_path: Path = None):
        super().__init__()
        self.frames = frames
        self.output_folder = output_folder
        self.presets = presets
        self.quality = quality
        self.max_dimension = max_dimension
        self.lut_path = lut_path

    def run(self):
        """Run export in background."""
        from ..export.screenshot import extract_frame, export_screenshot
        from ..crops import crop_to_preset

        paths = []
        total = len(self.frames) * len(self.presets)
        current = 0

        for frame in self.frames:
            try:
                # Extract frame once
                image = extract_frame(frame.video.path, frame.timestamp_sec)

                # Export each preset
                for preset in self.presets:
                    try:
                        # Apply crop
                        cropped = crop_to_preset(image, preset, frame.faces)

                        # Resize if max dimension set
                        if self.max_dimension:
                            width, height = cropped.size
                            if width > self.max_dimension or height > self.max_dimension:
                                ratio = self.max_dimension / max(width, height)
                                new_size = (int(width * ratio), int(height * ratio))
                                from PIL import Image
                                cropped = cropped.resize(new_size, Image.LANCZOS)

                        # Generate output path
                        filename = (
                            f"{frame.video.stem}_"
                            f"{frame.timestamp_sec:.1f}s_"
                            f"{frame.category.name.lower()}_"
                            f"{preset}.jpg"
                        )
                        category_folder = self.output_folder / frame.category.name.lower()
                        category_folder.mkdir(parents=True, exist_ok=True)
                        output_path = category_folder / filename

                        # Save
                        export_screenshot(cropped, output_path, self.quality)
                        paths.append(output_path)

                        current += 1
                        self.progress.emit(current, total, str(output_path))

                    except Exception as e:
                        logger.error(f"Failed to export {preset} for {frame.timecode}: {e}")
                        current += 1
                        continue

            except Exception as e:
                logger.error(f"Failed to extract frame {frame.timecode}: {e}")
                current += len(self.presets)
                continue

        self.finished.emit(paths)


class ThumbnailWorker(QThread):
    """Background thread for loading thumbnails."""

    thumbnail_ready = Signal(str, object)  # frame_id, QPixmap
    finished = Signal()

    def __init__(self, frames: list):
        super().__init__()
        self.frames = frames
        self._running = True

    def stop(self):
        """Stop the worker."""
        self._running = False

    def run(self):
        """Load thumbnails in background."""
        from ..export.screenshot import extract_frame

        for frame in self.frames:
            if not self._running:
                break

            try:
                # Extract frame
                image = extract_frame(frame.video.path, frame.timestamp_sec)

                # Scale down for thumbnail
                image.thumbnail((160, 90))

                # Convert to QPixmap
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                data = image.tobytes("raw", "RGB")
                qimage = QImage(data, image.width, image.height,
                               image.width * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)

                frame_id = f"{frame.video.stem}_{frame.timestamp_sec}"
                self.thumbnail_ready.emit(frame_id, pixmap)

            except Exception as e:
                logger.debug(f"Failed to load thumbnail for {frame.timecode}: {e}")
                continue

        self.finished.emit()


class ClipExportWorker(QThread):
    """Background thread for exporting video clips."""

    progress = Signal(int, int, str)  # current, total, path
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, clips: list, output_folder: Path, preset: str,
                 crf: int = 22, lut_path: Path = None):
        super().__init__()
        self.clips = clips
        self.output_folder = output_folder
        self.preset = preset
        self.crf = crf
        self.lut_path = lut_path

    def run(self):
        """Export clips in background."""
        from ..export.clip import export_clip_with_preset

        paths = []
        total = len(self.clips)

        for i, clip in enumerate(self.clips):
            try:
                path = export_clip_with_preset(
                    clip.video,
                    clip.start_sec,
                    clip.end_sec,
                    self.output_folder / clip.category.name.lower(),
                    self.preset,
                    crf=self.crf,
                    lut_path=self.lut_path,
                )
                paths.append(path)
                self.progress.emit(i + 1, total, str(path))

            except Exception as e:
                logger.error(f"Failed to export clip: {e}")
                continue

        self.finished.emit(paths)


# =============================================================================
# THUMBNAIL WIDGET
# =============================================================================

class ThumbnailWidget(QWidget):
    """Widget displaying a frame thumbnail with selection checkbox."""

    clicked = Signal(object)      # Frame
    selection_changed = Signal()  # Selection state changed

    def __init__(self, frame: Frame, parent=None):
        super().__init__(parent)
        self.frame = frame
        self.selected = True  # Default to selected
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Selection checkbox
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self._on_selection_changed)

        # Thumbnail container
        thumb_container = QWidget()
        thumb_layout = QHBoxLayout(thumb_container)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.addWidget(self.checkbox)

        # Thumbnail image
        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(160, 90)
        self.thumbnail.setStyleSheet(
            "background: #2a2a2a; border: 2px solid #444; border-radius: 4px;"
        )
        self.thumbnail.setAlignment(Qt.AlignCenter)
        self.thumbnail.setText("...")
        thumb_layout.addWidget(self.thumbnail)

        layout.addWidget(thumb_container)

        # Info label
        score = self.frame.scores.get('final', 0)
        info = f"{self.frame.timecode[:8]} | {score:.0%}"
        self.info_label = QLabel(info)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 10px; color: #aaa;")
        layout.addWidget(self.info_label)

        self.setFixedWidth(190)
        self.setCursor(Qt.PointingHandCursor)

    def _on_selection_changed(self, state):
        """Handle checkbox state change."""
        self.selected = state == Qt.Checked
        self._update_style()
        self.selection_changed.emit()

    def _update_style(self):
        """Update visual style based on selection."""
        if self.selected:
            self.thumbnail.setStyleSheet(
                "background: #2a2a2a; border: 2px solid #4CAF50; border-radius: 4px;"
            )
        else:
            self.thumbnail.setStyleSheet(
                "background: #1a1a1a; border: 2px solid #333; border-radius: 4px;"
            )
            self.thumbnail.setGraphicsEffect(None)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.frame)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Toggle selection on double click."""
        self.checkbox.setChecked(not self.checkbox.isChecked())
        super().mouseDoubleClickEvent(event)

    def set_thumbnail(self, pixmap: QPixmap):
        """Set the thumbnail image."""
        scaled = pixmap.scaled(
            156, 86,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.thumbnail.setPixmap(scaled)
        self.thumbnail.setText("")


# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Social Screenshot Tool")
        self.setMinimumSize(1280, 800)

        # State
        self.project: Optional[Project] = None
        self.worker: Optional[AnalysisWorker] = None
        self.export_worker: Optional[ExportWorker] = None
        self.thumb_worker: Optional[ThumbnailWorker] = None
        self.clip_worker: Optional[ClipExportWorker] = None
        self.lut_folder_path: Optional[Path] = None
        self.frame_widgets: dict[str, ThumbnailWidget] = {}

        # Settings
        self.settings = QSettings("SocialScreenshotTool", "SST")

        self._setup_ui()
        self._setup_status_bar()
        self._load_settings()
        self._apply_dark_theme()

    def _apply_dark_theme(self):
        """Apply dark theme to application."""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #1e1e1e; color: #e0e0e0; }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4CAF50;
            }
            QPushButton {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px 16px;
                color: #e0e0e0;
            }
            QPushButton:hover { background-color: #444; }
            QPushButton:pressed { background-color: #555; }
            QPushButton:disabled { background-color: #2a2a2a; color: #666; }
            QPushButton#primaryBtn {
                background-color: #4CAF50;
                border: none;
                color: white;
                font-weight: bold;
            }
            QPushButton#primaryBtn:hover { background-color: #45a049; }
            QPushButton#primaryBtn:disabled { background-color: #2a5a2e; }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 4px;
                text-align: center;
                background-color: #2a2a2a;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab {
                background-color: #2a2a2a;
                border: 1px solid #444;
                padding: 8px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #333;
                border-bottom-color: #333;
            }
            QSpinBox, QComboBox {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QScrollArea { border: none; }
            QSlider::groove:horizontal {
                height: 6px;
                background: #444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
        """)

    def _setup_ui(self):
        """Create the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # Left panel: Controls
        left_panel = self._create_control_panel()
        left_panel.setMaximumWidth(320)

        # Right panel: Results
        right_panel = self._create_results_panel()

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([320, 960])

        layout.addWidget(splitter)

    def _create_control_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        # === PROJECT SECTION ===
        project_group = QGroupBox("Project")
        project_layout = QVBoxLayout(project_group)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: #888; font-style: italic;")
        project_layout.addWidget(self.folder_label)

        select_btn = QPushButton("Select Video Folder...")
        select_btn.clicked.connect(self._select_folder)
        project_layout.addWidget(select_btn)

        self.analyze_btn = QPushButton("Analyze Videos")
        self.analyze_btn.setObjectName("primaryBtn")
        self.analyze_btn.clicked.connect(self._start_analysis)
        self.analyze_btn.setEnabled(False)
        project_layout.addWidget(self.analyze_btn)

        layout.addWidget(project_group)

        # === PROGRESS SECTION ===
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("color: #888;")
        progress_layout.addWidget(self.progress_label)

        layout.addWidget(progress_group)

        # === EXPORT SETTINGS ===
        export_group = QGroupBox("Screenshot Export")
        export_layout = QVBoxLayout(export_group)

        # Crop presets
        crop_label = QLabel("Crop Formats:")
        export_layout.addWidget(crop_label)

        self.crop_checkboxes = {}
        crop_grid = QGridLayout()
        crops = [
            ('square', 'Square (1:1)'),
            ('portrait', 'Portrait (4:5)'),
            ('story', 'Story (9:16)'),
            ('landscape', 'Landscape (16:9)'),
        ]
        for i, (key, label) in enumerate(crops):
            cb = QCheckBox(label)
            cb.setChecked(key in ['square', 'portrait', 'story'])
            self.crop_checkboxes[key] = cb
            crop_grid.addWidget(cb, i // 2, i % 2)
        export_layout.addLayout(crop_grid)

        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Max Size:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "1080px (Instagram)",
            "2048px (Facebook HD)",
            "1600px (Twitter)",
            "Original",
        ])
        res_row.addWidget(self.resolution_combo)
        export_layout.addLayout(res_row)

        # Quality slider
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Quality:"))
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(60, 100)
        self.quality_slider.setValue(90)
        self.quality_label = QLabel("90%")
        self.quality_slider.valueChanged.connect(
            lambda v: self.quality_label.setText(f"{v}%")
        )
        quality_row.addWidget(self.quality_slider)
        quality_row.addWidget(self.quality_label)
        export_layout.addLayout(quality_row)

        # Quotas
        export_layout.addWidget(QLabel("Quotas per category:"))
        self.quota_spinboxes = {}
        for category in ['People', 'Details', 'Venue']:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{category}:"))
            spinbox = QSpinBox()
            spinbox.setRange(1, 200)
            spinbox.setValue(DEFAULT_QUOTAS.get(f'{category.lower()}_screenshots', 50))
            self.quota_spinboxes[category.lower()] = spinbox
            row.addWidget(spinbox)
            export_layout.addLayout(row)

        layout.addWidget(export_group)

        # === CLIP SETTINGS ===
        clip_group = QGroupBox("Clip Export")
        clip_layout = QVBoxLayout(clip_group)

        # Clip type
        clip_type_row = QHBoxLayout()
        clip_type_row.addWidget(QLabel("Clip Type:"))
        self.clip_type_combo = QComboBox()
        self.clip_type_combo.addItems([
            "Single (2-18s)",
            "Multi (8-58s)",
        ])
        clip_type_row.addWidget(self.clip_type_combo)
        clip_layout.addLayout(clip_type_row)

        # Clip format
        clip_format_row = QHBoxLayout()
        clip_format_row.addWidget(QLabel("Format:"))
        self.clip_format_combo = QComboBox()
        self.clip_format_combo.addItems([
            "Square (1:1)",
            "Portrait (9:16)",
            "Landscape (16:9)",
        ])
        clip_format_row.addWidget(self.clip_format_combo)
        clip_layout.addLayout(clip_format_row)

        layout.addWidget(clip_group)

        # === LUT SETTINGS ===
        lut_group = QGroupBox("Color (LUT)")
        lut_layout = QVBoxLayout(lut_group)

        self.lut_label = QLabel("No LUT folder")
        self.lut_label.setStyleSheet("color: #888; font-style: italic;")
        lut_layout.addWidget(self.lut_label)

        lut_btn = QPushButton("Select LUT Folder...")
        lut_btn.clicked.connect(self._select_lut_folder)
        lut_layout.addWidget(lut_btn)

        self.auto_lut_checkbox = QCheckBox("Auto-apply LUT to LOG footage")
        self.auto_lut_checkbox.setChecked(True)
        lut_layout.addWidget(self.auto_lut_checkbox)

        layout.addWidget(lut_group)

        # === EXPORT BUTTONS ===
        layout.addStretch()

        # Selection info
        self.selection_label = QLabel("Select frames to export")
        self.selection_label.setStyleSheet("color: #888;")
        self.selection_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.selection_label)

        # Export screenshots button
        self.export_btn = QPushButton("Export Screenshots")
        self.export_btn.setObjectName("primaryBtn")
        self.export_btn.clicked.connect(self._export_screenshots)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)

        # Export clips button
        self.export_clips_btn = QPushButton("Export Clips")
        self.export_clips_btn.clicked.connect(self._export_clips)
        self.export_clips_btn.setEnabled(False)
        layout.addWidget(self.export_clips_btn)

        return panel

    def _create_results_panel(self) -> QWidget:
        """Create the results tab panel."""
        self.tabs = QTabWidget()

        # Create tab for each category
        self.category_tabs = {}
        for category in [Category.PEOPLE, Category.DETAILS, Category.VENUE]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            content = QWidget()
            grid = QGridLayout(content)
            grid.setSpacing(8)
            grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)

            scroll.setWidget(content)

            self.category_tabs[category] = {
                'scroll': scroll,
                'content': content,
                'grid': grid,
                'widgets': [],
            }

            self.tabs.addTab(scroll, category.name.capitalize())

        return self.tabs

    def _setup_status_bar(self):
        """Create status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Select a video folder to begin")

    def _load_settings(self):
        """Load saved settings."""
        # LUT folder
        lut_folder = self.settings.value("lut_folder", "")
        if lut_folder and Path(lut_folder).exists():
            self.lut_folder_path = Path(lut_folder)
            self.lut_label.setText(str(self.lut_folder_path.name))

        # Last folder
        last_folder = self.settings.value("last_folder", "")
        if last_folder and Path(last_folder).exists():
            self.folder_path = Path(last_folder)
            self.folder_label.setText(str(self.folder_path))
            self.folder_label.setStyleSheet("color: #e0e0e0;")
            self.analyze_btn.setEnabled(True)

        # Quality
        quality = self.settings.value("quality", 90, type=int)
        self.quality_slider.setValue(quality)

        # Resolution
        res_index = self.settings.value("resolution_index", 0, type=int)
        self.resolution_combo.setCurrentIndex(res_index)

    def _save_settings(self):
        """Save current settings."""
        if self.lut_folder_path:
            self.settings.setValue("lut_folder", str(self.lut_folder_path))
        if hasattr(self, 'folder_path'):
            self.settings.setValue("last_folder", str(self.folder_path))
        self.settings.setValue("quality", self.quality_slider.value())
        self.settings.setValue("resolution_index", self.resolution_combo.currentIndex())

    def closeEvent(self, event):
        """Save settings on close."""
        self._save_settings()

        # Stop any running workers
        if self.thumb_worker and self.thumb_worker.isRunning():
            self.thumb_worker.stop()
            self.thumb_worker.wait()

        super().closeEvent(event)

    # =========================================================================
    # FOLDER SELECTION
    # =========================================================================

    def _select_folder(self):
        """Open folder selection dialog."""
        start_folder = str(self.folder_path) if hasattr(self, 'folder_path') else str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Video Folder",
            start_folder,
        )

        if folder:
            self.folder_path = Path(folder)
            self.folder_label.setText(str(self.folder_path))
            self.folder_label.setStyleSheet("color: #e0e0e0;")
            self.analyze_btn.setEnabled(True)

            # Count videos
            video_count = 0
            for ext in VIDEO_EXTENSIONS:
                video_count += len(list(self.folder_path.rglob(f"*{ext}")))
                video_count += len(list(self.folder_path.rglob(f"*{ext.upper()}")))

            self.status_bar.showMessage(f"Found {video_count} video files in folder")
            self._save_settings()

    def _select_lut_folder(self):
        """Open LUT folder selection."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select LUT Folder",
            str(Path.home()),
        )
        if folder:
            self.lut_folder_path = Path(folder)
            self.lut_label.setText(str(self.lut_folder_path.name))
            self.lut_label.setStyleSheet("color: #e0e0e0;")

            # Count LUTs
            from ..export.lut import list_available_luts
            luts = list_available_luts(self.lut_folder_path)
            self.status_bar.showMessage(f"Found {len(luts)} LUT files")
            self._save_settings()

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    def _start_analysis(self):
        """Start video analysis."""
        if not hasattr(self, 'folder_path'):
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.analyze_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.export_clips_btn.setEnabled(False)

        # Clear previous results
        self._clear_results()

        # Start worker
        self.worker = AnalysisWorker(self.folder_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.start()

    def _clear_results(self):
        """Clear all result widgets."""
        # Stop thumbnail worker if running
        if self.thumb_worker and self.thumb_worker.isRunning():
            self.thumb_worker.stop()
            self.thumb_worker.wait()

        # Clear widgets
        for tab_data in self.category_tabs.values():
            for widget in tab_data['widgets']:
                widget.deleteLater()
            tab_data['widgets'].clear()

        self.frame_widgets.clear()

    def _on_progress(self, percent: int, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)
        self.status_bar.showMessage(message)

    def _on_analysis_finished(self, project: Project):
        """Handle analysis completion."""
        self.project = project
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.export_clips_btn.setEnabled(True)

        # Populate results
        self._populate_results()

        # Start thumbnail loading
        self._start_thumbnail_loading()

        total_frames = (
            len(project.people_frames) +
            len(project.details_frames) +
            len(project.venue_frames)
        )
        total_clips = (
            len(project.people_clips) +
            len(project.details_clips) +
            len(project.venue_clips)
        )

        self.progress_label.setText("Analysis complete!")
        self.status_bar.showMessage(
            f"Found {total_frames} frames and {total_clips} clip suggestions"
        )
        self._update_selection_count()

    def _on_analysis_error(self, error: str):
        """Handle analysis error."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.progress_label.setText(f"Error: {error}")
        self.status_bar.showMessage(f"Error: {error}")

        QMessageBox.critical(self, "Analysis Error", error)

    def _populate_results(self):
        """Populate result tabs with frames."""
        if not self.project:
            return

        frame_lists = {
            Category.PEOPLE: self.project.people_frames,
            Category.DETAILS: self.project.details_frames,
            Category.VENUE: self.project.venue_frames,
        }

        for category, frames in frame_lists.items():
            tab_data = self.category_tabs[category]
            grid = tab_data['grid']

            # Add thumbnails
            col_count = 5
            for i, frame in enumerate(frames[:100]):  # Limit to 100 per category
                widget = ThumbnailWidget(frame)
                widget.clicked.connect(self._on_frame_clicked)
                widget.selection_changed.connect(self._update_selection_count)

                row = i // col_count
                col = i % col_count
                grid.addWidget(widget, row, col)
                tab_data['widgets'].append(widget)

                # Store reference for thumbnail loading
                frame_id = f"{frame.video.stem}_{frame.timestamp_sec}"
                self.frame_widgets[frame_id] = widget

            # Update tab title with count
            tab_index = list(self.category_tabs.keys()).index(category)
            self.tabs.setTabText(tab_index, f"{category.name.capitalize()} ({len(frames)})")

    def _start_thumbnail_loading(self):
        """Start loading thumbnails in background."""
        if not self.project:
            return

        # Collect all frames (limited)
        all_frames = []
        for category in [Category.PEOPLE, Category.DETAILS, Category.VENUE]:
            frames = self.project.get_frames_by_category(category)[:100]
            all_frames.extend(frames)

        self.thumb_worker = ThumbnailWorker(all_frames)
        self.thumb_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self.thumb_worker.finished.connect(self._on_thumbnails_finished)
        self.thumb_worker.start()

    def _on_thumbnail_ready(self, frame_id: str, pixmap: QPixmap):
        """Handle loaded thumbnail."""
        if frame_id in self.frame_widgets:
            self.frame_widgets[frame_id].set_thumbnail(pixmap)

    def _on_thumbnails_finished(self):
        """Handle thumbnail loading complete."""
        self.status_bar.showMessage("Thumbnails loaded")

    def _on_frame_clicked(self, frame: Frame):
        """Handle frame thumbnail click."""
        self.status_bar.showMessage(
            f"Selected: {frame.video.filename} @ {frame.timecode} "
            f"({frame.category.name}, score: {frame.scores.get('final', 0):.0%})"
        )

    def _update_selection_count(self):
        """Update the selection count label."""
        count = sum(
            1 for widget in self.frame_widgets.values()
            if widget.selected
        )
        self.selection_label.setText(f"{count} frames selected for export")

    # =========================================================================
    # EXPORT
    # =========================================================================

    def _get_max_dimension(self) -> Optional[int]:
        """Get max dimension from resolution combo."""
        index = self.resolution_combo.currentIndex()
        dimensions = [1080, 2048, 1600, None]
        return dimensions[index] if index < len(dimensions) else None

    def _export_screenshots(self):
        """Export selected frames as screenshots."""
        if not self.project:
            return

        # Get output folder
        output_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(Path.home() / "Pictures" / "SST"),
        )

        if not output_folder:
            return

        output_path = Path(output_folder)

        # Get selected crops
        crops = [name for name, cb in self.crop_checkboxes.items() if cb.isChecked()]
        if not crops:
            QMessageBox.warning(self, "No Crops", "Please select at least one crop format.")
            return

        # Get quotas
        quotas = {cat: spinbox.value() for cat, spinbox in self.quota_spinboxes.items()}

        # Collect selected frames (respecting quotas)
        frames_to_export = []

        for category in [Category.PEOPLE, Category.DETAILS, Category.VENUE]:
            tab_data = self.category_tabs[category]
            quota = quotas[category.name.lower()]
            count = 0

            for widget in tab_data['widgets']:
                if widget.selected and count < quota:
                    frames_to_export.append(widget.frame)
                    count += 1

        if not frames_to_export:
            QMessageBox.warning(self, "No Frames", "No frames selected for export.")
            return

        # Disable UI
        self.export_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Get LUT path if applicable
        lut_path = None
        if self.auto_lut_checkbox.isChecked() and self.lut_folder_path:
            from ..export.lut import list_available_luts
            luts = list_available_luts(self.lut_folder_path)
            if luts:
                lut_path = luts[0]  # Use first available LUT

        # Start export worker
        self.export_worker = ExportWorker(
            frames_to_export,
            output_path,
            crops,
            quality=self.quality_slider.value(),
            max_dimension=self._get_max_dimension(),
            lut_path=lut_path,
        )
        self.export_worker.progress.connect(self._on_export_progress)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_worker.error.connect(self._on_export_error)
        self.export_worker.start()

    def _on_export_progress(self, current: int, total: int, path: str):
        """Handle export progress."""
        percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        filename = Path(path).name if path else ""
        self.progress_label.setText(f"Exporting {current}/{total}: {filename}")
        self.status_bar.showMessage(f"Exported: {filename}")

    def _on_export_finished(self, paths: list):
        """Handle export completion."""
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)

        self.progress_label.setText(f"Export complete! {len(paths)} files")
        self.status_bar.showMessage(f"Exported {len(paths)} screenshots")

        # Show completion dialog
        QMessageBox.information(
            self,
            "Export Complete",
            f"Successfully exported {len(paths)} screenshots.\n\n"
            f"Files saved to: {paths[0].parent.parent if paths else 'output folder'}"
        )

    def _on_export_error(self, error: str):
        """Handle export error."""
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)

        self.progress_label.setText(f"Export failed: {error}")
        QMessageBox.critical(self, "Export Error", error)

    def _export_clips(self):
        """Export video clips."""
        if not self.project:
            return

        # Get output folder
        output_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder for Clips",
            str(Path.home() / "Videos" / "SST"),
        )

        if not output_folder:
            return

        output_path = Path(output_folder)

        # Get clip format
        format_map = {0: 'square', 1: 'story', 2: 'landscape'}
        preset = format_map.get(self.clip_format_combo.currentIndex(), 'landscape')

        # Collect clips
        all_clips = (
            self.project.people_clips[:10] +
            self.project.details_clips[:5] +
            self.project.venue_clips[:5]
        )

        if not all_clips:
            QMessageBox.warning(self, "No Clips", "No clip suggestions available.")
            return

        # Disable UI
        self.export_clips_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start clip export
        self.clip_worker = ClipExportWorker(
            all_clips,
            output_path,
            preset,
        )
        self.clip_worker.progress.connect(self._on_clip_export_progress)
        self.clip_worker.finished.connect(self._on_clip_export_finished)
        self.clip_worker.start()

    def _on_clip_export_progress(self, current: int, total: int, path: str):
        """Handle clip export progress."""
        percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.progress_label.setText(f"Exporting clip {current}/{total}")

    def _on_clip_export_finished(self, paths: list):
        """Handle clip export completion."""
        self.progress_bar.setVisible(False)
        self.export_clips_btn.setEnabled(True)

        self.progress_label.setText(f"Exported {len(paths)} clips")
        QMessageBox.information(
            self,
            "Clip Export Complete",
            f"Successfully exported {len(paths)} video clips."
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Run the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Social Screenshot Tool")
    app.setOrganizationName("SocialScreenshotTool")

    # Set app style
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
