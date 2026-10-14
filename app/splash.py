
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsOpacityEffect


class SplashScreen(QWidget):
    """Animated MarineVision startup splash.

    Replace DUMMY_LOGO and APP_NAME below later with your real logo/name.
    """

    APP_NAME = "MarineVision"
    TAGLINE = "AI-Powered Sonar Detection & Mapping"
    DUMMY_LOGO = "◎"
    
    def __init__(self, duration=2200):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.duration = duration
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(620, 390)

        self._pulse = 0
        self._logo_opacity = QGraphicsOpacityEffect()
        self._name_opacity = QGraphicsOpacityEffect()
        self._tag_opacity = QGraphicsOpacityEffect()
        self._loading_opacity = QGraphicsOpacityEffect()

        self.setStyleSheet("""
            QWidget#splash {
                background: #100F3F;
                border: 1px solid #302C75;
                border-radius: 18px;
            }
            QLabel {
                background: transparent;
            }
        """)
        self.setObjectName("splash")

        root = QVBoxLayout(self)
        root.setContentsMargins(45, 35, 45, 30)
        root.setSpacing(0)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.logo = QLabel(self.DUMMY_LOGO)
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo.setFixedHeight(100)
        self.logo.setStyleSheet("""
            color: #F58A27;
            font-family: "Segoe UI";
            font-size: 72px;
            font-weight: 300;
        """)
        self.logo.setGraphicsEffect(self._logo_opacity)

        self.name = QLabel(self.APP_NAME)
        self.name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name.setStyleSheet("""
            color: white;
            font-family: "Segoe UI";
            font-size: 32px;
            font-weight: 700;
            letter-spacing: 1px;
        """)
        self.name.setGraphicsEffect(self._name_opacity)

        self.tagline = QLabel(self.TAGLINE)
        self.tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tagline.setStyleSheet("""
            color: #B9B7D2;
            font-family: "Segoe UI";
            font-size: 13px;
        """)
        self.tagline.setGraphicsEffect(self._tag_opacity)

        self.loading = QLabel("INITIALIZING MARINE INTELLIGENCE  •  •  •")
        self.loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading.setStyleSheet("""
            color: #8F8CB7;
            font-family: "Segoe UI";
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 1px;
        """)
        self.loading.setGraphicsEffect(self._loading_opacity)

        root.addWidget(self.logo)
        root.addSpacing(8)
        root.addWidget(self.name)
        root.addSpacing(7)
        root.addWidget(self.tagline)
        root.addSpacing(35)
        root.addWidget(self.loading)

        # Opacity starts at zero and fades in independently.
        for effect in (
            self._logo_opacity,
            self._name_opacity,
            self._tag_opacity,
            self._loading_opacity,
        ):
            effect.setOpacity(0.0)

        self._animations = []
        self._start_animations()



        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._advance_pulse)
        self._pulse_timer.start(45)




    def _fade(self, effect, start, end, duration, delay=0):
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setDuration(duration)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        if delay:
            QTimer.singleShot(delay, anim.start)
        else:
            anim.start()

        self._animations.append(anim)

    def _start_animations(self):
        self._fade(self._logo_opacity, 0.0, 1.0, 500, 0)
        self._fade(self._name_opacity, 0.0, 1.0, 500, 260)
        self._fade(self._tag_opacity, 0.0, 1.0, 450, 500)
        self._fade(self._loading_opacity, 0.0, 1.0, 400, 850)

    def _advance_pulse(self):
        self._pulse = (self._pulse + 5) % 180
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Main rounded splash surface.
        p.setPen(QPen(QColor("#302C75"), 1))
        p.setBrush(QColor("#100F3F"))
        p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 18, 18)

        # Decorative sonar rings behind the logo.
        center_x = self.width() / 2
        center_y = 92

        for base_radius, alpha in ((45, 36), (72, 28), (99, 21), (126, 15)):
            radius = base_radius + self._pulse * 0.08
            p.setPen(QPen(QColor(245, 138, 39, alpha), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(
                QRectF(center_x - radius, center_y - radius,
                       radius * 2, radius * 2)
            )

        # Tiny sonar sweep line.
        p.setPen(QPen(QColor(245, 138, 39, 70), 1))
        p.drawLine(center_x, center_y, center_x + 125, center_y)

        p.end()


def show_splash(app):
    splash = SplashScreen()
    screen = app.primaryScreen().availableGeometry()
    x = screen.center().x() - splash.width() // 2
    y = screen.center().y() - splash.height() // 2
    splash.move(x, y)
    splash.show()
    app.processEvents()
    return splash
