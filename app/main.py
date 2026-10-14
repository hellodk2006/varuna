import sys

from PySide6.QtCore import QTimer, Qt, QCoreApplication

# QtWebEngine (used by the new Mission Map's Leaflet view, geo_map.py)
# requires this to be set BEFORE QApplication is constructed, or
# Chromium's process can fail to initialize - usually silently, with no
# Python traceback, just an immediate exit. Must happen before
# QApplication(...) below.
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

from PySide6.QtWidgets import QApplication

from marine_app import MarineVisionWindow, apply_app_style
from splash import show_splash


def main():
    app = QApplication(sys.argv)

    app.setApplicationName("MarineVision")
    apply_app_style(app)

    splash = show_splash(app)

    # Create main window BEFORE splash ends
    try:
        window = MarineVisionWindow()
    except Exception as e:
        print("\n========== MAIN WINDOW ERROR ==========")
        print(type(e).__name__, ":", e)
        import traceback
        traceback.print_exc()
        print("=======================================\n")
        splash.close()
        raise

    def open_main_window():
        try:
            window.show()
            window.raise_()
            window.activateWindow()

            splash.close()
            splash.deleteLater()

        except Exception as e:
            print("\n========== WINDOW SHOW ERROR ==========")
            print(type(e).__name__, ":", e)
            import traceback
            traceback.print_exc()
            print("=======================================\n")

    QTimer.singleShot(splash.duration, open_main_window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()