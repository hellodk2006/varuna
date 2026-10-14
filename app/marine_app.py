
import csv
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QLinearGradient, QPen, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFrame, QStackedWidget,
    QFileDialog, QMessageBox, QSizePolicy, QScrollArea, QProgressBar,
    QComboBox, QLineEdit, QCheckBox, QSlider, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView
)

from sonar_integration import (
    run_detection, refilter_overlay, DEFAULT_MODEL_PATH, DEFAULT_CLASSES_PATH,
    DEFAULT_RESULTS_DIR,
)
from severity import mission_severity, colors_for
from audio import play_ping, play_alert
from report_generator import generate_mission_report
from xtf_inference import run_xtf_mission
import sync_client


class DetectionWorker(QThread):
    """Runs the NLM -> YOLO11s sonar pipeline off the GUI thread so the app
    doesn't freeze while a detection is in progress.

    Dispatches on file extension: a .xtf mission goes through the full
    XTF -> frames -> YOLO -> geotag pipeline (xtf_inference.run_xtf_mission);
    anything else (PNG/JPG/TIFF) goes through the original single-image
    pipeline (sonar_integration.run_detection).
    """

    finished_ok = Signal(dict)     # emits the pipeline's result dict
    finished_err = Signal(str)     # emits an error message

    def __init__(self, model_path, classes_path, image_path, confidence):
        super().__init__()
        self.model_path = model_path
        self.classes_path = classes_path
        self.image_path = image_path
        self.confidence = confidence

    def run(self):
        try:
            if Path(self.image_path).suffix.lower() == ".xtf":
                results = run_xtf_mission(
                    self.image_path,
                    model_path=self.model_path,
                    classes_path=self.classes_path,
                    confidence=self.confidence,
                )
                # Normalize a few keys so the single-image result-handling
                # code in on_analysis_done() below can display an XTF
                # mission result without needing its own branch for every
                # field name.
                mission_detections = results.get("detections", [])
                results["image"] = results.get("mission", Path(self.image_path).name)
                results["confidence_threshold"] = results.get("base_confidence", self.confidence)
                results["severity"] = mission_severity(mission_detections)
                # Representative preview: the first annotated frame that
                # actually has a detection on it, if any.
                detection_frames = results.get("detection_frames", [])
                results["overlay_path"] = (
                    detection_frames[0]["overlay_path"] if detection_frames else None
                )
            else:
                results = run_detection(
                    self.image_path,
                    model_path=self.model_path,
                    classes_path=self.classes_path,
                    confidence=self.confidence,
                )
            self.finished_ok.emit(results)
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")


# ============================================================
# MarineVision UI
# Redesigned to prevent clipping/overlap and closely match
# the supplied MarineVision prototype.
# ============================================================

NAVY = "#100F3F"
NAVY_2 = "#15144A"
NAVY_3 = "#24205F"
BG = "#F7F4EE"
WHITE = "#FFFFFF"
TEXT = "#141A3D"
MUTED = "#747795"
BORDER = "#D9D8E2"
PURPLE = "#3430A8"
PURPLE_LIGHT = "#E9E8FA"
ORANGE = "#F58A27"
ORANGE_DARK = "#DB7518"
GREEN = "#2E9B72"
BLUE = "#315FAF"

# Change this one value to switch the font used across the whole app.
# Falls back silently to a system default if the font isn't installed.
FONT_FAMILY = "Segoe UI"


def label(text, size=14, color=TEXT, bold=False, object_name=None):
    w = QLabel(text)
    if object_name:
        w.setObjectName(object_name)
    f = QFont(FONT_FAMILY)
    f.setPointSize(size)
    f.setWeight(QFont.Bold if bold else QFont.Normal)
    w.setFont(f)
    w.setStyleSheet(f"color:{color}; background:transparent; border:none; padding:0;")
    return w


def make_button(text, kind="secondary", height=46):
    b = QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    b.setFixedHeight(height)
    b.setFont(QFont(FONT_FAMILY, 11))
    b.setStyleSheet({
        "primary": f"""
            QPushButton {{
                background:{ORANGE}; color:white; border:none;
                border-radius:7px; padding:0 22px; font-weight:700;
            }}
            QPushButton:hover {{ background:{ORANGE_DARK}; }}
        """,
        "secondary": f"""
            QPushButton {{
                background:white; color:{TEXT}; border:1px solid {BORDER};
                border-radius:7px; padding:0 20px; font-weight:600;
            }}
            QPushButton:hover {{ background:{PURPLE_LIGHT}; border-color:#B8B5DE; }}
        """,
        "ghost": """
            QPushButton {
                background:transparent; color:white;
                border:1px solid rgba(255,255,255,80);
                border-radius:7px; padding:0 20px; font-weight:600;
            }
            QPushButton:hover { background:rgba(255,255,255,18); }
        """,
        "nav": """
            QPushButton {
                background:transparent; color:#BDBBD9;
                border:none; border-radius:6px;
                padding:0 12px; font-size:15px;
            }
            QPushButton:hover { color:white; background:rgba(255,255,255,14); }
            QPushButton[active="true"] {
                color:white; background:#302C75;
            }
        """
    }[kind])
    return b


class HeroBackground(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(405)
        self.setMaximumHeight(430)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        r = self.rect()
        g = QLinearGradient(0, 0, r.width(), r.height())
        g.setColorAt(0, QColor(NAVY))
        g.setColorAt(0.65, QColor(NAVY_2))
        g.setColorAt(1, QColor("#211F5D"))
        p.fillRect(r, g)

        # Subtle sonar rings on the right.
        cx = r.width() - 160
        cy = 145
        p.setPen(QPen(QColor(255, 255, 255, 18), 1))
        for radius in (70, 110, 150, 190):
            p.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        p.setPen(QPen(QColor(255, 255, 255, 10), 1))
        p.drawLine(cx - 205, cy, cx + 205, cy)
        p.drawLine(cx, cy - 205, cx, cy + 205)
        p.end()


class StatBlock(QFrame):
    def __init__(self, value, title):
        super().__init__()
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 0, 18, 0)
        lay.setSpacing(3)

        v = label(value, 25, WHITE, True)
        t = label(title, 11, "#B8B7D2")
        lay.addWidget(v)
        lay.addWidget(t)


class CapabilityCard(QFrame):
    def __init__(self, icon, title, description, accent):
        super().__init__()
        self.setMinimumHeight(178)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setStyleSheet("QFrame { background:transparent; border:none; }")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(10)

        icon_box = QLabel(icon)
        icon_box.setAlignment(Qt.AlignCenter)
        icon_box.setFixedSize(38, 38)
        icon_box.setStyleSheet(
            f"background:{accent}18; color:{accent}; border-radius:9px; "
            "font-size:18px; font-weight:700;"
        )

        title_w = label(title, 16, TEXT, True)
        title_w.setMinimumHeight(22)

        body = label(description, 12, MUTED)
        body.setWordWrap(True)
        body.setMinimumHeight(54)

        lay.addWidget(icon_box, 0, Qt.AlignLeft)
        lay.addWidget(title_w)
        lay.addWidget(body)
        lay.addStretch(1)


class MissionRow(QFrame):
    def __init__(self, name, date, status, detections, show_divider=True):
        super().__init__()
        self.setObjectName("missionRow")
        self.setMinimumHeight(62)
        divider_css = f"border-bottom:1px solid {BORDER};" if show_divider else "border:none;"
        self.setStyleSheet(f"""
            QFrame#missionRow {{
                background:transparent;
                {divider_css}
                border-radius:0px;
            }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 8, 18, 8)
        lay.setSpacing(18)

        name_w = label(name, 13, TEXT, True)
        name_w.setMinimumWidth(220)

        date_w = label(date, 12, MUTED)
        date_w.setMinimumWidth(145)

        det = label(f"{detections} detections", 12, MUTED)
        det.setMinimumWidth(125)

        status_w = label(status, 11, GREEN, True)
        status_w.setAlignment(Qt.AlignCenter)
        status_w.setFixedWidth(82)
        status_w.setFixedHeight(26)
        status_w.setStyleSheet(
            f"color:{GREEN}; background:#EAF7F1; border-radius:13px;"
        )

        lay.addWidget(name_w)
        lay.addWidget(date_w)
        lay.addWidget(det)
        lay.addStretch(1)
        lay.addWidget(status_w)


class HomePage(QWidget):
    def __init__(self, navigate):
        super().__init__()
        self.navigate = navigate

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setStyleSheet(f"background:{BG};")
        root = QVBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---------------- HERO ----------------
        hero = HeroBackground()
        h = QVBoxLayout(hero)
        h.setContentsMargins(72, 42, 72, 30)
        h.setSpacing(0)

        title = label(
            "AI-Powered Sonar Detection & Mapping",
            39, WHITE, True
        )
        title.setMinimumHeight(52)
        title.setMaximumHeight(58)

        subtitle = label(
            "Intelligent systems for real-time underwater debris detection",
            16, "#C9C8DC"
        )
        subtitle.setMinimumHeight(24)

        h.addWidget(title)
        h.addSpacing(7)
        h.addWidget(subtitle)
        h.addSpacing(12)

        offline_badge = QFrame()
        offline_badge.setStyleSheet(
            "QFrame { background:rgba(46,155,114,38); border:1px solid rgba(46,155,114,140); "
            "border-radius:13px; }"
        )
        offline_badge.setFixedHeight(28)
        offline_badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        badge_lay = QHBoxLayout(offline_badge)
        badge_lay.setContentsMargins(12, 0, 14, 0)
        badge_lay.setSpacing(7)
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{GREEN}; font-size:9px; background:transparent;")
        badge_lay.addWidget(dot)
        badge_lay.addWidget(label(
            "100% Offline Inference — no cloud, no internet required",
            11, "#CFF3E4", True,
        ))
        h.addWidget(offline_badge, 0, Qt.AlignLeft)

        h.addSpacing(18)
        h.addStretch(1)

        stats = QHBoxLayout()
        stats.setContentsMargins(0, 0, 0, 0)
        stats.setSpacing(0)

        data = [
            ("595", "Sonar Frames"),
            ("4", "Detect Classes"),
            ("87.7%", "Avg. Confidence"),
            ("124 ms", "Inference Time"),
        ]

        for i, (value, text) in enumerate(data):
            stats.addWidget(StatBlock(value, text), 1)
            if i < len(data) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.VLine)
                line.setFixedHeight(48)
                line.setStyleSheet("color:rgba(255,255,255,35);")
                stats.addWidget(line)

        h.addLayout(stats)
        root.addWidget(hero)

        # ---------------- CAPABILITIES ----------------
        section = QWidget()
        sl = QVBoxLayout(section)
        sl.setContentsMargins(72, 38, 72, 25)
        sl.setSpacing(14)

        top = QHBoxLayout()
        top.setSpacing(10)

        heading_box = QVBoxLayout()
        heading_box.setSpacing(3)

        heading = label("What MarineVision can do", 29, TEXT, True)
        heading.setMinimumHeight(38)

        heading_box.addWidget(heading)

        top.addLayout(heading_box)
        top.addStretch(1)
        sl.addLayout(top)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)

        cards = [
            ("◎", "Real-time Sonar Analysis",
             "Process sonar imagery and inspect signal quality in one workspace.",
             PURPLE),
            ("☑", "Multi-class Detection",
             "Detect marine debris and anomalies with confidence scores.",
             ORANGE),
            ("⌂", "Geospatial Mapping",
             "Connect detections to mission coordinates and sonar tracks.",
             "#D7961D"),
            ("▧", "Mission Reporting",
             "Create clear mission summaries with findings and model metrics.",
             BLUE),
        ]

        for i, item in enumerate(cards):
            grid.addWidget(CapabilityCard(*item), 0, i)

        sl.addLayout(grid)

        # ---------------- RECENT MISSIONS ----------------
        recent_top = QHBoxLayout()
        recent_top.setContentsMargins(0, 12, 0, 0)

        recent_box = QVBoxLayout()
        recent_box.setSpacing(3)
        recent_box.addWidget(label("Recent Missions", 27, TEXT, True))
        recent_top.addLayout(recent_box)
        recent_top.addStretch(1)
        sl.addLayout(recent_top)

        recent_container = QFrame()
        recent_container.setObjectName("recentContainer")
        recent_container.setStyleSheet(f"""
            QFrame#recentContainer {{
                background:white;
                border:1px solid {BORDER};
                border-radius:10px;
            }}
        """)
        recent = QVBoxLayout(recent_container)
        recent.setContentsMargins(0, 0, 0, 0)
        recent.setSpacing(0)
        recent.addWidget(MissionRow("Mission 024 · Arabian Sea", "11 Sep 2026",
                                    "Completed", 24))
        recent.addWidget(MissionRow("Mission 023 · Konkan Coast", "09 Sep 2026",
                                    "Completed", 18))
        recent.addWidget(MissionRow("Mission 022 · Training Run", "06 Sep 2026",
                                    "Completed", 31, show_divider=False))
        sl.addWidget(recent_container)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 15, 0, 5)
        footer.addWidget(label("© 2026 MarineVision", 10, "#8B887D"))
        footer.addStretch(1)
        footer.addWidget(label("AI-assisted marine intelligence", 10, "#8B887D"))
        sl.addLayout(footer)

        root.addWidget(section)
        root.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)


class ScanningIndicator(QWidget):
    """Small animated radar sweep shown in the result panel while a
    detection is running, so 'analysis in progress' reads as an active
    sonar scan rather than a frozen screen."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(72, 72)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._angle = 0
        self._timer.start(35)
        self.show()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        self._angle = (self._angle + 8) % 360
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(4, 4, -4, -4)
        cx, cy = r.center().x(), r.center().y()
        radius = min(r.width(), r.height()) / 2

        p.setPen(QPen(QColor(255, 255, 255, 55), 1.4))
        for frac in (0.4, 0.7, 1.0):
            rr = radius * frac
            p.drawEllipse(int(cx - rr), int(cy - rr), int(rr * 2), int(rr * 2))

        p.save()
        p.translate(cx, cy)
        p.rotate(self._angle)
        g = QLinearGradient(0, 0, radius, 0)
        g.setColorAt(0, QColor(245, 138, 39, 210))
        g.setColorAt(1, QColor(245, 138, 39, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(g)
        p.drawPie(int(-radius), int(-radius), int(radius * 2), int(radius * 2), 0, 35 * 16)
        p.restore()

        p.setPen(QPen(QColor(245, 138, 39), 2))
        p.setBrush(QColor(245, 138, 39))
        p.drawEllipse(int(cx) - 3, int(cy) - 3, 6, 6)
        p.end()


class SeverityBanner(QFrame):
    """Colored alert strip: red for high-priority finds (e.g. a mine-like
    object), orange for medium, green/quiet when the mission is clear."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(0)
        self.setVisible(False)
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(14, 0, 14, 0)
        self._label = label("", 12, WHITE, True)
        self._lay.addWidget(self._label)

    def set_severity(self, severity: dict):
        colors = colors_for(severity["level"])
        self.setStyleSheet(
            f"background:{colors['solid']}; border-radius:7px;"
        )
        icon = "⚠" if severity["level"] == "high" else ("●" if severity["level"] != "none" else "✓")
        self._label.setText(f"{icon}  {severity['label']}  ·  {severity['reason']}")
        self.setFixedHeight(38)
        self.setVisible(True)

    def clear(self):
        self.setVisible(False)
        self.setFixedHeight(0)


class PageShell(QWidget):
    def __init__(self, eyebrow, title, subtitle):
        super().__init__()

        # Wrapped in a QScrollArea (same pattern as HomePage) so pages whose
        # content is taller than the visible workspace scroll instead of
        # being compressed - an uncompressed page was forcing fixed-height
        # widgets (combo boxes, progress bars) to hold their size while
        # plain labels collapsed toward 0px, which is what produced the
        # overlapping / vanishing text on the Analyze and Model pages.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setStyleSheet(f"background:{BG};")
        root = QVBoxLayout(content)
        root.setContentsMargins(56, 44, 56, 40)
        root.setSpacing(8)

        root.addWidget(label(title, 30, TEXT, True))
        root.addWidget(label(subtitle, 14, MUTED))
        root.addSpacing(26)

        self.body = QVBoxLayout()
        self.body.setSpacing(16)
        root.addLayout(self.body)
        root.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)


class AnalyzePage(PageShell):
    def __init__(self):
        super().__init__(
            "Detection Workspace",
            "Analyze Sonar Data",
            "Load side-scan sonar imagery, configure detection and review the output."
        )
        self.selected_file = None

        steps = QHBoxLayout()
        steps.setSpacing(0)
        for i, text in enumerate(["1  Load", "2  Preprocess", "3  Detect", "4  Results"]):
            s = label(text, 12, WHITE if i == 0 else MUTED, True)
            s.setAlignment(Qt.AlignCenter)
            s.setFixedHeight(38)
            s.setStyleSheet(
                f"background:{PURPLE if i == 0 else '#ECEBF2'};"
                f"color:{WHITE if i == 0 else MUTED}; border-radius:5px;"
            )
            steps.addWidget(s, 1)
            if i < 3:
                steps.addSpacing(5)
        self.body.addLayout(steps)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)

        # Upload card
        upload = QFrame()
        upload.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        upload.setStyleSheet("QFrame { background:transparent; border:none; }")
        ul = QVBoxLayout(upload)
        ul.setContentsMargins(25, 24, 25, 24)
        ul.setSpacing(12)

        ul.addWidget(label("SONAR INPUT", 11, "#7779A1", True))
        self.file_title = label("No sonar file selected", 17, TEXT, True)
        self.file_title.setWordWrap(True)
        self.file_hint = label(
            "PNG, JPG, TIFF, XTF or CSV files",
            12, MUTED
        )
        self.file_hint.setWordWrap(True)

        browse = make_button("Browse Sonar Files", "secondary", 44)
        browse.clicked.connect(self.browse)

        ul.addWidget(self.file_title)
        ul.addWidget(self.file_hint)
        ul.addSpacing(5)
        ul.addWidget(browse, 0, Qt.AlignLeft)
        ul.addStretch(1)

        # Config card
        config = QFrame()
        config.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        config.setStyleSheet(f"""
            QFrame {{ background:transparent; border:none; }}
            QComboBox, QLineEdit {{
                background:#FAFAFC; border:1px solid {BORDER};
                border-radius:6px; padding:8px 10px; color:{TEXT};
            }}
        """)
        cl = QVBoxLayout(config)
        cl.setContentsMargins(25, 24, 25, 24)
        cl.setSpacing(12)
        cl.addWidget(label("DETECTION SETTINGS", 11, "#7779A1", True))

        cl.addWidget(label("Model", 11, MUTED))
        self.model = QComboBox()
        self.model.addItems([
            "YOLO11s Sonar Detector (bundled)",
            "Custom Model"
        ])
        self.model.setMinimumHeight(38)
        cl.addWidget(self.model)

        cl.addSpacing(4)
        cl.addWidget(label("Model weights (.pt)", 11, MUTED))
        weights_row = QHBoxLayout()
        weights_row.setSpacing(8)
        # Defaults to the trained weights bundled in sonar_model/models/,
        # so most users never need to touch this - it's here for
        # "Custom Model" or testing a different checkpoint.
        self.weights_path = (
            str(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else None
        )
        default_label = (
            DEFAULT_MODEL_PATH.name if self.weights_path
            else "No weights file selected"
        )
        self.weights_label = label(default_label, 12, MUTED)
        self.weights_label.setWordWrap(True)
        weights_browse = make_button("Browse", "secondary", 38)
        weights_browse.clicked.connect(self.browse_weights)
        weights_row.addWidget(self.weights_label, 1)
        weights_row.addWidget(weights_browse, 0)
        cl.addLayout(weights_row)

        cl.addSpacing(4)
        cl.addWidget(label("Confidence threshold", 11, MUTED))
        self.threshold = QLineEdit("0.25")
        self.threshold.setMinimumHeight(38)
        cl.addWidget(self.threshold)

        # Manual device/image location - only meaningful for a plain image
        # (PNG/JPG/etc), since an XTF mission already carries its own real
        # navigation data (see xtf_mission.py) and doesn't need this.
        cl.addSpacing(4)
        self.location_label = label("Device location (optional)", 11, MUTED)
        cl.addWidget(self.location_label)
        location_row = QHBoxLayout()
        location_row.setSpacing(8)
        self.lat_input = QLineEdit()
        self.lat_input.setPlaceholderText("Latitude")
        self.lat_input.setMinimumHeight(38)
        self.lon_input = QLineEdit()
        self.lon_input.setPlaceholderText("Longitude")
        self.lon_input.setMinimumHeight(38)
        location_row.addWidget(self.lat_input)
        location_row.addWidget(self.lon_input)
        cl.addLayout(location_row)
        self.location_hint = label(
            "Pins this image on the Mission Map. Leave blank to skip.",
            10, MUTED
        )
        self.location_hint.setWordWrap(True)
        cl.addWidget(self.location_hint)

        cl.addSpacing(10)
        self.start_btn = make_button("Start Analysis", "primary", 44)
        self.start_btn.clicked.connect(self.start_analysis)
        cl.addWidget(self.start_btn)
        cl.addStretch(1)

        grid.addWidget(upload, 0, 0)
        grid.addWidget(config, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self.body.addLayout(grid)

        result = QFrame()
        result.setMinimumHeight(245)
        result.setStyleSheet(
            f"background:#11113E; border-radius:10px;"
        )
        rl = QVBoxLayout(result)
        rl.setContentsMargins(24, 20, 24, 20)
        rl.setSpacing(9)

        header_row = QHBoxLayout()
        header_row.addWidget(label("RESULT PREVIEW", 11, "#A9A7CF", True))
        header_row.addStretch(1)
        rl.addLayout(header_row)

        self.severity_banner = SeverityBanner()
        rl.addWidget(self.severity_banner)

        content_row = QHBoxLayout()
        content_row.setSpacing(16)

        text_col = QVBoxLayout()
        self.result_label = label(
            "Load a sonar image to preview AI detections.",
            16, WHITE, True
        )
        self.result_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(90)
        text_col.addWidget(self.result_label)
        text_col.addStretch(1)
        content_row.addLayout(text_col, 1)

        self.scanning_indicator = ScanningIndicator()
        self.scanning_indicator.hide()
        content_row.addWidget(self.scanning_indicator, 0, Qt.AlignTop)

        rl.addLayout(content_row)

        self.result_image = QLabel()
        self.result_image.setAlignment(Qt.AlignCenter)
        self.result_image.hide()
        rl.addWidget(self.result_image)

        # Live sensitivity slider: re-filters and redraws the ALREADY
        # DETECTED boxes by confidence without re-running the model, so it
        # only ranges from the run's own threshold up to 100%.
        self.sensitivity_row = QWidget()
        sens_lay = QHBoxLayout(self.sensitivity_row)
        sens_lay.setContentsMargins(0, 4, 0, 0)
        sens_lay.setSpacing(10)
        sens_lay.addWidget(label("Sensitivity", 11, "#A9A7CF"))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimumWidth(160)
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_changed)
        sens_lay.addWidget(self.sensitivity_slider, 1)
        self.sensitivity_value_label = label("", 11, "#A9A7CF", True)
        self.sensitivity_value_label.setMinimumWidth(46)
        sens_lay.addWidget(self.sensitivity_value_label)
        self.sensitivity_row.hide()
        rl.addWidget(self.sensitivity_row)

        actions_row = QHBoxLayout()
        actions_row.addStretch(1)
        self.report_btn = make_button("Generate PDF Report", "ghost", 38)
        self.report_btn.clicked.connect(self.generate_report)
        self.report_btn.hide()
        actions_row.addWidget(self.report_btn, 0)
        rl.addLayout(actions_row)

        self.body.addWidget(result)

        self.worker = None
        self.sync_worker = None
        self.last_results = None
        self._pending_lat = None
        self._pending_lon = None
        self._preview_path = str(Path(tempfile.gettempdir()) / "marinevision_sensitivity_preview.png")

    def browse_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO11 Weights", "",
            "PyTorch Weights (*.pt);;All Files (*)"
        )
        if path:
            self.weights_path = path
            self.weights_label.setText(Path(path).name)

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Sonar Data", "",
            "Sonar/Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.xtf);;"
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.selected_file = path
            self.file_title.setText(Path(path).name)
            self.file_hint.setText(path)
            self.result_label.setText(
                "SONAR IMAGE LOADED\n\n"
                f"{Path(path).name}\n\n"
                "Ready for preprocessing and detection."
            )
            self.severity_banner.clear()
            self.sensitivity_row.hide()
            self.report_btn.hide()
            self.last_results = None

            # An XTF mission already carries real navigation data (see
            # xtf_mission.py) - manual location entry only applies to a
            # plain image, which has no GPS of its own.
            is_xtf = Path(path).suffix.lower() == ".xtf"
            self.location_label.setVisible(not is_xtf)
            self.lat_input.setVisible(not is_xtf)
            self.lon_input.setVisible(not is_xtf)
            self.location_hint.setVisible(not is_xtf)
            if is_xtf:
                self.location_hint.setText(
                    "This mission's real ship track will be read from the "
                    "XTF file's own navigation data automatically."
                )
            else:
                self.location_hint.setText(
                    "Pins this image on the Mission Map. Leave blank to skip."
                )

    def start_analysis(self):
        if not self.selected_file:
            QMessageBox.warning(
                self, "No File Selected",
                "Please select a sonar image or mission file first."
            )
            return

        if not self.weights_path:
            QMessageBox.warning(
                self, "No Model Selected",
                "Please browse to your trained YOLO11 weights (.pt) file first."
            )
            return

        try:
            confidence = float(self.threshold.text())
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Confidence",
                "Confidence threshold must be a number between 0 and 1."
            )
            return

        # Manual device location only applies to a plain image (an XTF
        # mission gets its real position from the file itself). Blank is
        # fine - it just means this analysis won't be pinned on the map.
        self._pending_lat = None
        self._pending_lon = None
        is_xtf = Path(self.selected_file).suffix.lower() == ".xtf"
        if not is_xtf:
            lat_text = self.lat_input.text().strip()
            lon_text = self.lon_input.text().strip()
            if lat_text or lon_text:
                try:
                    self._pending_lat = float(lat_text)
                    self._pending_lon = float(lon_text)
                except ValueError:
                    QMessageBox.warning(
                        self, "Invalid Location",
                        "Latitude and longitude must both be numbers "
                        "(e.g. 15.3961, 73.7213) - or leave both blank."
                    )
                    return

        self.result_image.hide()
        self.severity_banner.clear()
        self.sensitivity_row.hide()
        self.report_btn.hide()
        self.result_label.setText(
            "ANALYZING...\n\n"
            f"File: {Path(self.selected_file).name}\n"
            f"Model: {self.model.currentText()}\n"
            f"Confidence: {confidence}\n\n"
            "Running YOLO11 inference - this may take a moment."
        )
        self.start_btn.setEnabled(False)
        self.scanning_indicator.start()
        play_ping()

        self.worker = DetectionWorker(
            self.weights_path, str(DEFAULT_CLASSES_PATH),
            self.selected_file, confidence,
        )
        self.worker.finished_ok.connect(self.on_analysis_done)
        self.worker.finished_err.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_done(self, results: dict):
        self.start_btn.setEnabled(True)
        self.scanning_indicator.stop()

        # Stamp the manually-entered device location (plain images only -
        # an XTF mission already has its own real journey_start/end/track)
        # into both the in-memory result and the JSON already written to
        # disk, so geo_map.py's Mission Map picks it up on next refresh.
        is_xtf_mission = "mission" in results and "navigation_source" in results
        if not is_xtf_mission and self._pending_lat is not None:
            results["latitude"] = self._pending_lat
            results["longitude"] = self._pending_lon
            json_path = results.get("results_json_path")
            if json_path and Path(json_path).exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        saved = json.load(f)
                    saved["latitude"] = self._pending_lat
                    saved["longitude"] = self._pending_lon
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(saved, f, indent=4)
                except Exception:
                    pass  # location just won't show on the map; analysis itself still succeeded

        self.last_results = results

        detections = results.get("detections", [])
        count = results.get("detection_count", len(detections))

        if detections:
            counts_by_class = {}
            for d in detections:
                counts_by_class[d["class_name"]] = counts_by_class.get(d["class_name"], 0) + 1
            summary = ",  ".join(f"{n}x {name}" for name, n in counts_by_class.items())
        else:
            summary = "No objects detected above the confidence threshold."

        json_path = results.get("results_json_path")
        saved_note = f"\nSaved: {Path(json_path).name}" if json_path else ""

        self.result_label.setText(
            "ANALYSIS COMPLETE\n\n"
            f"File: {results.get('image', Path(self.selected_file).name)}\n"
            f"{count} detection(s):  {summary}"
            f"{saved_note}"
        )

        severity = results.get("severity") or mission_severity(detections)
        self.severity_banner.set_severity(severity)
        if severity["level"] == "high":
            play_alert()

        overlay_path = results.get("overlay_path")
        if overlay_path and Path(overlay_path).exists():
            pixmap = QPixmap(overlay_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaledToHeight(220, Qt.SmoothTransformation)
                self.result_image.setPixmap(pixmap)
                self.result_image.show()

        # The live sensitivity re-filter only makes sense for a single
        # image (refilter_overlay redraws one image_path in place) - an
        # XTF mission has many frames, so skip it there and point people
        # at the PDF/CSV instead, which cover every frame.
        is_xtf_mission = "mission" in results and "navigation_source" in results

        if detections and not is_xtf_mission:
            base_pct = round(results.get("confidence_threshold", 0.25) * 100)
            self.sensitivity_slider.blockSignals(True)
            self.sensitivity_slider.setRange(base_pct, 100)
            self.sensitivity_slider.setValue(base_pct)
            self.sensitivity_slider.blockSignals(False)
            self.sensitivity_value_label.setText(f"{base_pct}%")
            self.sensitivity_row.show()
        else:
            self.sensitivity_row.hide()

        self.report_btn.show()

        # Best-effort copy to the Docker backend's shared database, and a
        # confirmation popup once that succeeds. XTF missions aren't sent
        # here - the backend's /upload endpoint only understands a single
        # image, and every mission is already saved locally (JSON, CSV,
        # overlays, PDF) regardless of whether the backend is reachable.
        if not is_xtf_mission:
            self.sync_worker = sync_client.SyncWorker(
                self.selected_file,
                results.get("confidence_threshold", 0.25),
            )
            self.sync_worker.done.connect(self.on_sync_done)
            self.sync_worker.start()

    def on_sync_done(self, outcome: dict):
        if outcome.get("saved"):
            QMessageBox.information(
                self, "Report Saved",
                f"Report #{outcome['id']} has been saved to the backend database."
            )
        # No popup on failure: if the Docker backend is off, popping up an
        # error on every single analysis would get in the way of a normal
        # (offline) demo. The mission is still saved locally either way.
        self.sync_worker = None

    def on_analysis_error(self, message: str):
        self.start_btn.setEnabled(True)
        self.scanning_indicator.stop()
        self.result_label.setText(
            "ANALYSIS FAILED\n\n" + message
        )
        QMessageBox.critical(self, "Detection Error", message)

    def on_sensitivity_changed(self, value: int):
        self.sensitivity_value_label.setText(f"{value}%")
        if not self.last_results:
            return
        refilter_overlay(self.last_results, value, self._preview_path)
        pixmap = QPixmap(self._preview_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaledToHeight(220, Qt.SmoothTransformation)
            self.result_image.setPixmap(pixmap)
            self.result_image.show()

    def generate_report(self):
        if not self.last_results:
            return
        default_name = f"{Path(self.last_results.get('image', 'mission')).stem}_report.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mission Report", default_name, "PDF Files (*.pdf)"
        )
        if not path:
            return
        try:
            generate_mission_report(self.last_results, path)
            QMessageBox.information(
                self, "Report Generated", f"Mission report saved to:\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Report Failed", f"Could not generate report:\n{e}"
            )


# The real Leaflet-based map (device pins + XTF ship tracks) lives in its
# own module; imported here (mid-file, not at the top) so it can safely
# import PageShell/label/make_button/MUTED/BORDER back from this module
# without a circular-import failure - those names all exist by this point
# in marine_app.py's top-to-bottom execution.
from geo_map import MapPage


class ReportsPage(PageShell):
    """Real detection history: every run of run_detection() saves a JSON
    file into results/, and this page reads them back so past missions
    are a browsable record, not a one-off demo screen."""

    COLUMNS = ["File", "Analyzed", "Detections", "Top Confidence", "Status"]

    def __init__(self):
        super().__init__(
            "Mission History",
            "Mission Reports",
            "Review detection results and export mission evidence."
        )

        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by file name or detected class…")
        self.search_box.setMinimumHeight(38)
        self.search_box.setStyleSheet(f"""
            QLineEdit {{
                background:#FAFAFC; border:1px solid {BORDER}; border-radius:6px;
                padding:0 12px; color:{TEXT};
            }}
        """)
        self.search_box.textChanged.connect(self.apply_filter)
        toolbar.addWidget(self.search_box, 1)

        refresh_btn = make_button("↻  Refresh", "secondary", 38)
        refresh_btn.clicked.connect(self.reload_history)
        toolbar.addWidget(refresh_btn)
        self.body.addLayout(toolbar)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setMinimumHeight(280)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background:white; border:1px solid {BORDER};
                border-radius:8px; gridline-color:#ECEBF0;
            }}
            QHeaderView::section {{
                background:#F2F1F6; color:{TEXT};
                padding:10px; border:none; font-weight:700;
            }}
            QTableWidget::item {{
                padding:9px; color:{TEXT};
            }}
            QTableWidget::item:selected {{
                background:{PURPLE_LIGHT}; color:{TEXT};
            }}
        """)
        self.body.addWidget(self.table)

        self.empty_label = label(
            "No missions yet — run a detection from the Analyze page and it will show up here.",
            12, MUTED
        )
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.hide()
        self.body.addWidget(self.empty_label)

        buttons = QHBoxLayout()
        create = make_button("Generate PDF Report", "primary", 44)
        csv_btn = make_button("Export CSV", "secondary", 44)
        js_btn = make_button("Export JSON", "secondary", 44)

        create.clicked.connect(self.create_report_for_selection)
        csv_btn.clicked.connect(self.export_csv)
        js_btn.clicked.connect(self.export_json)

        buttons.addWidget(create)
        buttons.addWidget(csv_btn)
        buttons.addWidget(js_btn)
        buttons.addStretch(1)
        self.body.addLayout(buttons)

        self._all_records = []   # every mission read from disk
        self._records = []       # currently-filtered/displayed records
        self.reload_history()

    def _load_records(self):
        records = []
        if DEFAULT_RESULTS_DIR.exists():
            for json_path in sorted(
                DEFAULT_RESULTS_DIR.glob("*_detection.json"), reverse=True
            ):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                data["_json_path"] = str(json_path)
                records.append(data)
        return records

    def reload_history(self):
        self._all_records = self._load_records()
        self.apply_filter()

    def apply_filter(self):
        query = self.search_box.text().strip().lower()
        if not query:
            self._records = list(self._all_records)
        else:
            self._records = [
                r for r in self._all_records
                if query in r.get("image", "").lower()
                or any(query in d.get("class_name", "").lower()
                       for d in r.get("detections", []))
            ]
        self._populate_table()

    def _populate_table(self):
        self.table.setRowCount(len(self._records))
        self.empty_label.setVisible(len(self._records) == 0)
        self.table.setVisible(len(self._records) > 0)

        for row, rec in enumerate(self._records):
            detections = rec.get("detections", [])
            count = rec.get("detection_count", len(detections))
            top_conf = max((d.get("confidence_percent", 0) for d in detections), default=0)
            severity = rec.get("severity") or mission_severity(detections)

            timestamp = rec.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp).strftime("%d %b %Y, %H:%M")
            except Exception:
                timestamp = timestamp or "—"

            values = [
                rec.get("image", "unknown"),
                timestamp,
                str(count),
                f"{top_conf:.1f}%" if detections else "—",
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

            status_item = QTableWidgetItem(severity["label"])
            colors = colors_for(severity["level"])
            status_item.setForeground(QColor(colors["text"]))
            self.table.setItem(row, 4, status_item)

    def _selected_record(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        idx = rows[0].row()
        if 0 <= idx < len(self._records):
            return self._records[idx]
        return None

    def create_report_for_selection(self):
        record = self._selected_record()
        if not record:
            QMessageBox.information(
                self, "Select a Mission",
                "Select a row in the table first, then click Generate PDF Report."
            )
            return
        default_name = f"{Path(record.get('image', 'mission')).stem}_report.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mission Report", default_name, "PDF Files (*.pdf)"
        )
        if not path:
            return
        try:
            generate_mission_report(record, path)
            QMessageBox.information(self, "Report Generated", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Report Failed", f"Could not generate report:\n{e}")

    def export_csv(self):
        if not self._records:
            QMessageBox.information(self, "Nothing to Export", "No mission history to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV Report", "mission_history.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)
                for row in range(self.table.rowCount()):
                    writer.writerow([
                        self.table.item(row, c).text() if self.table.item(row, c) else ""
                        for c in range(self.table.columnCount())
                    ])
            QMessageBox.information(self, "Export CSV", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def export_json(self):
        if not self._records:
            QMessageBox.information(self, "Nothing to Export", "No mission history to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON Report", "mission_history.json", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            exportable = [
                {k: v for k, v in r.items() if k != "_json_path"}
                for r in self._records
            ]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(exportable, f, indent=2)
            QMessageBox.information(self, "Export JSON", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))


# Real figures from training/train_yolo.md in the source model repo (not
# placeholders) - baseline YOLO11s run on the cleaned DRISHTI-SSS dataset.
MODEL_TRAINING_INFO = {
    "architecture": "YOLO11s",
    "image_size": "640 x 640",
    "epochs": "50",
    "batch_size": "8",
    "gpu": "NVIDIA RTX 5060 Laptop GPU (~8 GB VRAM)",
    "preprocessing": "Non-Local Means (NLM) denoising - no resize/crop/rotate",
    "dataset": "DRISHTI-SSS (cleaned)",
}
MODEL_METRICS = {
    "Validation": {"Precision": 0.721, "Recall": 0.755, "mAP@50": 0.727, "mAP@50-95": 0.569},
    "Held-out Test": {"Precision": 0.795, "Recall": 0.691, "mAP@50": 0.713, "mAP@50-95": 0.561},
}


def _metric_bar(name, value):
    row = QVBoxLayout()
    row.setSpacing(3)
    top = QHBoxLayout()
    top.addWidget(label(name, 12, TEXT, True))
    top.addStretch(1)
    top.addWidget(label(f"{value*100:.1f}%", 12, MUTED, True))
    row.addLayout(top)
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(round(value * 100))
    bar.setTextVisible(False)
    bar.setFixedHeight(9)
    bar.setStyleSheet(f"""
        QProgressBar {{ background:#ECEBF2; border:none; border-radius:4px; }}
        QProgressBar::chunk {{ background:{PURPLE}; border-radius:4px; }}
    """)
    row.addWidget(bar)
    wrap = QWidget()
    wrap.setLayout(row)
    return wrap


class DatasetPage(PageShell):
    """Real model/training info (from training/train_yolo.md) plus dataset
    import - this used to be a placeholder progress bar with a made-up
    percentage; it now shows the actual documented baseline results."""

    def __init__(self):
        super().__init__(
            "Model & Training",
            "Model Performance",
            "Architecture, training configuration and validation metrics for the bundled model."
        )

        # ---- Model / training config card ----
        info_card = QFrame()
        info_card.setStyleSheet(f"background:white; border:1px solid {BORDER}; border-radius:10px;")
        il = QVBoxLayout(info_card)
        il.setContentsMargins(24, 20, 24, 20)
        il.setSpacing(10)
        il.addWidget(label("MODEL & TRAINING CONFIGURATION", 11, "#7779A1", True))

        grid = QGridLayout()
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        rows = [
            ("Architecture", MODEL_TRAINING_INFO["architecture"]),
            ("Image size", MODEL_TRAINING_INFO["image_size"]),
            ("Dataset", MODEL_TRAINING_INFO["dataset"]),
            ("Epochs", MODEL_TRAINING_INFO["epochs"]),
            ("Batch size", MODEL_TRAINING_INFO["batch_size"]),
            ("Training GPU", MODEL_TRAINING_INFO["gpu"]),
            ("Preprocessing", MODEL_TRAINING_INFO["preprocessing"]),
            ("Classes", "crab_pot, submarine_pipeline, shipwreck, ghost_net, mine_cylinder"),
        ]
        for i, (k, v) in enumerate(rows):
            r, c = divmod(i, 2)
            cell = QVBoxLayout()
            cell.setSpacing(1)
            cell.addWidget(label(k, 11, MUTED))
            val = label(v, 13, TEXT, True)
            val.setWordWrap(True)
            cell.addWidget(val)
            cell_w = QWidget()
            cell_w.setLayout(cell)
            grid.addWidget(cell_w, r, c)
        il.addLayout(grid)
        self.body.addWidget(info_card)

        # ---- Metrics card ----
        metrics_card = QFrame()
        metrics_card.setStyleSheet(f"background:white; border:1px solid {BORDER}; border-radius:10px;")
        ml = QVBoxLayout(metrics_card)
        ml.setContentsMargins(24, 20, 24, 20)
        ml.setSpacing(14)
        ml.addWidget(label("BASELINE RESULTS", 11, "#7779A1", True))

        cols = QHBoxLayout()
        cols.setSpacing(28)
        for split_name, metrics in MODEL_METRICS.items():
            col = QVBoxLayout()
            col.setSpacing(10)
            col.addWidget(label(split_name, 14, TEXT, True))
            for metric_name, value in metrics.items():
                col.addWidget(_metric_bar(metric_name, value))
            col_w = QWidget()
            col_w.setLayout(col)
            cols.addWidget(col_w, 1)
        ml.addLayout(cols)
        self.body.addWidget(metrics_card)

        # ---- Dataset import ----
        import_card = QFrame()
        import_card.setStyleSheet(f"background:white; border:1px solid {BORDER}; border-radius:10px;")
        cl = QVBoxLayout(import_card)
        cl.setContentsMargins(24, 20, 24, 20)
        cl.setSpacing(10)
        cl.addWidget(label("DATASET LIBRARY", 11, "#7779A1", True))
        cl.addWidget(label(
            "Import additional annotated sonar imagery to extend the training set.",
            13, MUTED
        ))
        import_btn = make_button("Import Dataset", "secondary", 40)
        import_btn.clicked.connect(self.import_dataset)
        cl.addWidget(import_btn, 0, Qt.AlignLeft)
        self.body.addWidget(import_card)

    def import_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Dataset", "",
            "Dataset Files (*.zip *.csv *.json *.txt);;All Files (*)"
        )
        if path:
            QMessageBox.information(
                self, "Dataset Imported",
                f"Selected dataset:\n{path}"
            )


class ConfigPage(PageShell):
    def __init__(self):
        super().__init__(
            "System Preferences",
            "Configuration",
            "Configure inference, compute device and reporting preferences."
        )

        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{ background:transparent; border:none; }}
            QComboBox, QLineEdit {{
                background:#FAFAFC; border:1px solid {BORDER};
                border-radius:6px; padding:8px 10px; color:{TEXT};
            }}
            QCheckBox {{ color:{TEXT}; }}
        """)
        grid = QGridLayout(card)
        grid.setContentsMargins(28, 26, 28, 26)
        grid.setHorizontalSpacing(30)
        grid.setVerticalSpacing(18)

        grid.addWidget(label("MODEL", 11, "#7779A1", True), 0, 0)
        grid.addWidget(label("COMPUTE", 11, "#7779A1", True), 0, 1)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv8 Marine Debris", "YOLOv8 Anomaly"])
        self.model_combo.setMinimumHeight(40)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CPU", "CUDA / GPU"])
        self.device_combo.setMinimumHeight(40)

        grid.addWidget(self.model_combo, 1, 0)
        grid.addWidget(self.device_combo, 1, 1)

        grid.addWidget(label("CONFIDENCE THRESHOLD", 11, "#7779A1", True), 2, 0)
        grid.addWidget(label("REPORT FORMAT", 11, "#7779A1", True), 2, 1)

        self.threshold_edit = QLineEdit("0.50")
        self.threshold_edit.setMinimumHeight(40)

        self.output_combo = QComboBox()
        self.output_combo.addItems(["CSV + JSON", "CSV", "JSON"])
        self.output_combo.setMinimumHeight(40)

        grid.addWidget(self.threshold_edit, 3, 0)
        grid.addWidget(self.output_combo, 3, 1)

        self.save_btn = make_button("Save Configuration", "primary", 44)
        self.save_btn.clicked.connect(self.save)
        grid.addWidget(self.save_btn, 4, 0, 1, 2, Qt.AlignLeft)

        self.body.addWidget(card)

    def save(self):
        QMessageBox.information(
            self, "Configuration Saved",
            f"Model: {self.model_combo.currentText()}\n"
            f"Device: {self.device_combo.currentText()}\n"
            f"Confidence: {self.threshold_edit.text()}\n"
            f"Report: {self.output_combo.currentText()}"
        )


class MarineVisionWindow(QMainWindow):
    """Main desktop shell with a collapsible left navigation rail."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MarineVision")
        self.resize(1280, 860)
        self.setMinimumSize(1100, 720)
        self.sidebar_expanded = True

        central = QWidget()
        central.setStyleSheet(f"background:{BG};")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---------------- LEFT SIDEBAR ----------------
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(190)
        self.sidebar.setStyleSheet(f"""
            QFrame#sidebar {{
                background:{NAVY};
                border-right:1px solid #262454;
            }}
        """)
        sl = QVBoxLayout(self.sidebar)
        sl.setContentsMargins(10, 14, 10, 14)
        sl.setSpacing(6)

        # Functional desktop control instead of duplicate branding.
        self.collapse_btn = QPushButton("☰  Collapse")
        self.collapse_btn.setCursor(Qt.PointingHandCursor)
        self.collapse_btn.setFixedHeight(40)
        self.collapse_btn.setStyleSheet("""
            QPushButton {
                background:transparent; color:#BDBBD9; border:none;
                border-radius:7px; text-align:left; padding:0 12px;
                font-size:11px; font-weight:600;
            }
            QPushButton:hover { background:rgba(255,255,255,14); color:white; }
        """)
        self.collapse_btn.clicked.connect(self.toggle_sidebar)
        sl.addWidget(self.collapse_btn)
        sl.addSpacing(10)

        self.nav_buttons = {}
        nav_items = [
            ("home", "⌂", "Home"),
            ("detect", "◎", "Analyze"),
            ("map", "⌖", "Map"),
            ("reports", "▤", "Reports"),
            ("dataset", "▧", "Model"),
            ("config", "⚙", "Config"),
        ]
        for key, icon, text in nav_items:
            b = QPushButton(f"{icon}   {text}")
            b.setProperty("nav_key", key)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(44)
            b.setToolTip(text)
            b.setStyleSheet("""
                QPushButton {
                    background:transparent; color:#BDBBD9; border:none;
                    border-radius:7px; text-align:left; padding:0 12px;
                    font-size:13px; font-weight:600;
                }
                QPushButton:hover { color:white; background:rgba(255,255,255,14); }
                QPushButton[active="true"] {
                    color:white; background:#302C75;
                    border-left:3px solid #F58A27;
                    padding-left:9px;
                }
            """)
            b.clicked.connect(lambda checked=False, k=key: self.go(k))
            self.nav_buttons[key] = b
            sl.addWidget(b)

        sl.addStretch(1)
        root.addWidget(self.sidebar)

        # ---------------- WORKSPACE ----------------
        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        # Compact desktop toolbar: page context + useful controls.
        self.toolbar = QFrame()
        self.toolbar.setFixedHeight(64)
        self.toolbar.setStyleSheet(f"""
            QFrame#desktopToolbar {{
                background:white;
                border-bottom:1px solid {BORDER};
            }}
        """)
        self.toolbar.setObjectName("desktopToolbar")
        tl = QHBoxLayout(self.toolbar)
        tl.setContentsMargins(22, 10, 22, 10)
        tl.setSpacing(8)

        self.page_title = label("Home", 16, TEXT, True)
        tl.addWidget(self.page_title)
        tl.addSpacing(18)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search missions, files or detections…")
        self.search.setFixedHeight(38)
        self.search.setMinimumWidth(280)
        self.search.setStyleSheet(f"""
            QLineEdit {{
                background:{BG}; border:1px solid {BORDER}; border-radius:7px;
                padding:0 12px; color:{TEXT};
            }}
            QLineEdit:focus {{ border-color:#B8B5DE; background:white; }}
        """)
        tl.addWidget(self.search)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Missions", "Completed", "In Progress", "Training"])
        self.filter_combo.setFixedHeight(38)
        self.filter_combo.setMinimumWidth(125)
        tl.addWidget(self.filter_combo)

        tl.addStretch(1)

        self.refresh_btn = QPushButton("↻  Refresh")
        self.refresh_btn.setFixedHeight(38)
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{ background:white; color:{TEXT}; border:1px solid {BORDER};
                border-radius:7px; padding:0 13px; font-weight:600; }}
            QPushButton:hover {{ background:{PURPLE_LIGHT}; }}
        """)
        self.refresh_btn.clicked.connect(self.refresh_current_page)
        tl.addWidget(self.refresh_btn)

        self.analyze_toolbar_btn = QPushButton("+  New Analysis")
        self.analyze_toolbar_btn.setFixedHeight(38)
        self.analyze_toolbar_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_toolbar_btn.setStyleSheet(f"""
            QPushButton {{ background:{PURPLE}; color:white; border:none;
                border-radius:7px; padding:0 14px; font-weight:700; }}
            QPushButton:hover {{ background:{NAVY_3}; }}
        """)
        self.analyze_toolbar_btn.clicked.connect(lambda: self.go("detect"))
        tl.addWidget(self.analyze_toolbar_btn)

        workspace_layout.addWidget(self.toolbar)

        # ---------------- PAGES ----------------
        self.pages = QStackedWidget()
        self.pages.setStyleSheet("background:transparent;")
        workspace_layout.addWidget(self.pages, 1)

        root.addWidget(workspace, 1)

        self.page_map = {}
        pages = [
            ("home", HomePage(self.go)),
            ("detect", AnalyzePage()),
            ("map", MapPage()),
            ("reports", ReportsPage()),
            ("dataset", DatasetPage()),
            ("config", ConfigPage()),
        ]
        for key, page in pages:
            self.page_map[key] = self.pages.count()
            self.pages.addWidget(page)

        self.go("home")

    def toggle_sidebar(self):
        self.sidebar_expanded = not self.sidebar_expanded
        self.sidebar.setFixedWidth(64 if not self.sidebar_expanded else 190)
        for key, button in self.nav_buttons.items():
            text = button.property("nav_text") or {
                "home":"Home", "detect":"Analyze", "map":"Map",
                "reports":"Reports", "dataset":"Model", "config":"Config"
            }[key]
            icon = {"home":"⌂", "detect":"◎", "map":"⌖", "reports":"▤", "dataset":"▧", "config":"⚙"}[key]
            button.setText(f"{icon}   {text}" if self.sidebar_expanded else icon)
            button.setStyleSheet(button.styleSheet().replace(
                "text-align:left; padding:0 12px;", 
                "text-align:center; padding:0;" if not self.sidebar_expanded else "text-align:left; padding:0 12px;"
            ).replace(
                "padding-left:9px;", "padding-left:9px;" if self.sidebar_expanded else "padding-left:0;"
            ))
        self.collapse_btn.setText("☰  Collapse" if self.sidebar_expanded else "☰")
        self.collapse_btn.setStyleSheet(self.collapse_btn.styleSheet().replace(
            "text-align:left; padding:0 12px;",
            "text-align:center; padding:0;" if not self.sidebar_expanded else "text-align:left; padding:0 12px;"
        ))

    def go(self, key):
        if key not in self.page_map:
            return
        self.pages.setCurrentIndex(self.page_map[key])
        if key == "map":
            self.pages.widget(self.page_map[key]).refresh()
        titles = {
            "home": "Home", "detect": "Analyze", "map": "Map",
            "reports": "Reports", "dataset": "Model", "config": "Config"
        }
        self.page_title.setText(titles.get(key, "MarineVision"))

        for name, button in self.nav_buttons.items():
            button.setProperty("active", name == key)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def refresh_current_page(self):
        # Re-apply the current page so the toolbar action remains useful
        # without introducing page-specific destructive behaviour.
        current = self.pages.currentIndex()
        self.pages.widget(current).update()

def apply_app_style(app):
    app.setStyle("Fusion")
    app.setFont(QFont(FONT_FAMILY, 10))
    app.setStyleSheet(f"""
        QWidget {{
            font-family:"{FONT_FAMILY}", "Segoe UI", Arial, sans-serif;
            color:{TEXT};
        }}
        QScrollArea {{
            border:none;
            background:{BG};
        }}
        QScrollBar:vertical {{
            width:8px;
            background:transparent;
        }}
        QScrollBar::handle:vertical {{
            background:#C9C7D0;
            border-radius:4px;
            min-height:30px;
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height:0;
        }}
        QToolTip {{
            background:#181741;
            color:white;
            border:1px solid #39366F;
            padding:6px;
        }}
    """)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MarineVision")
    apply_app_style(app)

    window = MarineVisionWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
