
import sys
import time
import json
import cv2

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox,
    QFrame, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView,
    QProgressBar, QGroupBox
)

from core.mediapipe_engine import MediaPipeEngine
from core.biomechanics import BiomechanicsEngine
from core.joint_selector import JointSelector
from core.temporal_filter import TemporalFilter
from core.squat_counter import SquatCounter
from core.correction_engine import CorrectionEngine
from core.session_manager import SessionManager
from core.report_generator import ReportGenerator


# ==========================================
# CAMERA WORKER
# ==========================================

class CameraWorker(QThread):

    frame_ready = Signal(QImage)
    metrics_ready = Signal(dict)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.session = None
        self.counter = None

    def run(self):

        self.running = True
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            self.error.emit("Camera not found")
            return

        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        engine = None

        try:
            engine = MediaPipeEngine()
            biomechanics = BiomechanicsEngine()
            selector = JointSelector()
            smoothing = TemporalFilter(5)
            self.counter = SquatCounter()
            self.session = SessionManager(
                exercise="Squat",
                model="MediaPipe Pose"
            )

            previous_time = time.perf_counter()
            previous_side = None

            while self.running:

                success, frame = cap.read()
                if not success:
                    self.error.emit("Camera frame error")
                    break

                frame, result = engine.process(frame)

                angles = {}
                landmarks = None

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    angles = biomechanics.analyze(
                        landmarks, frame.shape
                    )

                selected = selector.update(
                    landmarks, angles
                )

                side = selected["side"]

                if side != previous_side:
                    smoothing.reset()
                    previous_side = side

                knee = smoothing.update(
                    selected["angle"]
                )

                now = time.perf_counter()
                status = self.counter.update(
                    knee, timestamp=now
                )

                elapsed = now - previous_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                previous_time = now

                self.session.update(fps)

                # TECHNIQUE ASSESSMENT
                selected_angles = angles.copy()
                selected_angles["left_knee"] = None
                selected_angles["right_knee"] = None

                if side is not None:
                    selected_angles[
                        f"{side}_knee"
                    ] = knee

                assessment = CorrectionEngine().analyze(
                    selected_angles,
                    status["stage"]
                )

                # IMAGE
                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )
                h, w, ch = rgb.shape

                image = QImage(
                    rgb.data, w, h,
                    ch * w, QImage.Format_RGB888
                ).copy()

                self.frame_ready.emit(image)

                history = status["rep_history"]
                depth_pass = sum(
                    1 for rep in history
                    if rep["status"] == "Correct"
                )

                self.metrics_ready.emit({
                    "reps": status["reps"],
                    "fps": round(fps, 1),
                    "knee": knee,
                    "stage": status["stage"],
                    "side": side,
                    "quality": selected["quality"],
                    "depth_pass": depth_pass,
                    "angles": angles.copy(),
                    "assessment": assessment,
                    "history": [r.copy() for r in history]
                })

        except Exception as exc:
            self.error.emit(str(exc))

        finally:
            if engine is not None:
                engine.close()
            cap.release()

    def stop(self):
        self.running = False


# ==========================================
# METRIC CARD
# ==========================================

class MetricCard(QFrame):

    def __init__(self, label, value="--"):
        super().__init__()
        self.setObjectName("metricCard")

        layout = QVBoxLayout(self)
        title = QLabel(label)
        title.setObjectName("metricTitle")

        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")

        layout.addWidget(title)
        layout.addWidget(self.value_label)

    def set_value(self, value):
        self.value_label.setText(str(value))


# ==========================================
# MAIN WINDOW
# ==========================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "ExerciseAI Studio - Research Dashboard"
        )
        self.resize(1450, 900)

        self.worker = None
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)

        header = QLabel("EXERCISE AI STUDIO")
        header.setObjectName("header")
        root.addWidget(header)

        subtitle = QLabel(
            "Real-Time Biomechanics & Technique Assessment"
        )
        subtitle.setObjectName("subtitle")
        root.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(18)
        root.addLayout(content, 1)

        # LEFT
        left = QVBoxLayout()
        content.addLayout(left, 3)

        self.camera_label = QLabel(
            "Camera preview will appear here"
        )
        self.camera_label.setObjectName("camera")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(600, 400)
        left.addWidget(self.camera_label, 1)

        controls = QHBoxLayout()

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")

        self.start_button.clicked.connect(
            self.start_camera
        )
        self.stop_button.clicked.connect(
            self.stop_camera
        )
        self.stop_button.setEnabled(False)

        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        left.addLayout(controls)

        # BIOMECHANICS
        bio_group = QGroupBox("Live Biomechanics")
        bio_layout = QVBoxLayout(bio_group)

        self.angle_bars = {}

        for name in [
            "left_knee",
            "right_knee",
            "left_hip",
            "right_hip",
            "torso_inclination"
        ]:
            row = QHBoxLayout()

            label = QLabel(name.replace("_", " ").title())
            label.setMinimumWidth(130)

            bar = QProgressBar()
            bar.setRange(0, 180)
            bar.setValue(0)
            bar.setFormat("N/A")

            row.addWidget(label)
            row.addWidget(bar)

            bio_layout.addLayout(row)
            self.angle_bars[name] = bar

        left.addWidget(bio_group)

        # RIGHT
        right = QVBoxLayout()
        content.addLayout(right, 2)

        right.addWidget(QLabel("POSE MODEL"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["MediaPipe Pose"])
        right.addWidget(self.model_combo)

        right.addWidget(QLabel("EXERCISE"))
        self.exercise_combo = QComboBox()
        self.exercise_combo.addItems(["Squat"])
        right.addWidget(self.exercise_combo)

        grid = QGridLayout()
        grid.setSpacing(10)

        self.cards = {
            "reps": MetricCard("REPETITIONS", "0"),
            "fps": MetricCard("FPS", "0"),
            "knee": MetricCard("KNEE ANGLE", "--"),
            "depth_pass": MetricCard("DEPTH PASS", "0"),
            "stage": MetricCard("STAGE", "--"),
            "quality": MetricCard("TRACK QUALITY", "--")
        }

        positions = [
            ("reps", 0, 0),
            ("fps", 0, 1),
            ("knee", 1, 0),
            ("depth_pass", 1, 1),
            ("stage", 2, 0),
            ("quality", 2, 1)
        ]

        for key, row, col in positions:
            grid.addWidget(self.cards[key], row, col)

        right.addLayout(grid)

        # TECHNIQUE FEEDBACK
        feedback_group = QGroupBox("Technique Feedback")
        feedback_layout = QVBoxLayout(feedback_group)

        self.feedback_label = QLabel(
            "Start camera to analyze technique"
        )
        self.feedback_label.setWordWrap(True)
        feedback_layout.addWidget(self.feedback_label)

        right.addWidget(feedback_group)

        # REP HISTORY
        history_group = QGroupBox("Repetition History")
        history_layout = QVBoxLayout(history_group)

        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels([
            "Rep", "Duration", "Min Knee", "Score"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.history_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )

        history_layout.addWidget(self.history_table)
        right.addWidget(history_group, 1)

        self.save_button = QPushButton(
            "Save Session & Generate PDF"
        )
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(
            self.save_report
        )
        right.addWidget(self.save_button)

        self.statusBar().showMessage("Ready")

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #101827;
                color: #E8F0F8;
                font-family: Segoe UI;
                font-size: 13px;
            }

            QLabel#header {
                font-size: 27px;
                font-weight: bold;
                color: #34E0B1;
            }

            QLabel#subtitle {
                color: #91A5BA;
            }

            QLabel#camera {
                background: #080F1B;
                border: 1px solid #304359;
                border-radius: 12px;
            }

            QFrame#metricCard {
                background: #1B293B;
                border: 1px solid #2B3E53;
                border-radius: 10px;
            }

            QLabel#metricTitle {
                color: #91A5BA;
                font-size: 11px;
            }

            QLabel#metricValue {
                color: white;
                font-size: 23px;
                font-weight: bold;
            }

            QGroupBox {
                border: 1px solid #304359;
                border-radius: 9px;
                margin-top: 12px;
                padding: 12px;
                font-weight: bold;
            }

            QGroupBox::title {
                color: #34E0B1;
                subcontrol-origin: margin;
                left: 12px;
            }

            QComboBox {
                background: #24364B;
                padding: 9px;
                border-radius: 7px;
            }

            QPushButton {
                background: #147D6A;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
            }

            QPushButton:disabled {
                background: #34445C;
                color: #8192A5;
            }

            QProgressBar {
                background: #24364B;
                border: none;
                border-radius: 5px;
                text-align: center;
            }

            QProgressBar::chunk {
                background: #34E0B1;
                border-radius: 5px;
            }

            QTableWidget {
                background: #152235;
                gridline-color: #304359;
                border: none;
            }
        """)

    def start_camera(self):
        if self.worker is not None and self.worker.isRunning():
            return

        self.worker = CameraWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.metrics_ready.connect(self.update_metrics)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.camera_finished)

        self.worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)

        self.history_table.setRowCount(0)
        self.statusBar().showMessage("Camera running...")

    def stop_camera(self):
        if self.worker is not None:
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.statusBar().showMessage("Stopping camera...")

    def camera_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if self.worker and self.worker.session:
            self.save_button.setEnabled(True)

        self.statusBar().showMessage(
            "Camera stopped. Session ready to save."
        )

    def update_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def update_metrics(self, data):
        self.cards["reps"].set_value(data["reps"])
        self.cards["fps"].set_value(data["fps"])

        knee = data["knee"]
        self.cards["knee"].set_value(
            f"{knee:.1f} deg" if knee is not None else "N/A"
        )

        self.cards["depth_pass"].set_value(data["depth_pass"])
        self.cards["stage"].set_value(data["stage"].upper())
        self.cards["quality"].set_value(
            f"{data['quality']:.2f}"
        )

        # LIVE ANGLES
        for name, bar in self.angle_bars.items():
            angle = data["angles"].get(name)

            if angle is not None:
                bar.setValue(int(max(0, min(180, angle))))
                bar.setFormat(f"{angle:.1f} deg")
            else:
                bar.setValue(0)
                bar.setFormat("N/A")

        # FEEDBACK
        assessment = data["assessment"]

        if isinstance(assessment, dict):
            feedback = assessment.get("feedback", [])
            if isinstance(feedback, list):
                self.feedback_label.setText(
                    "\n".join(map(str, feedback))
                    if feedback else "No feedback available"
                )
            else:
                self.feedback_label.setText(str(feedback))
        else:
            self.feedback_label.setText(str(assessment))

        # REPETITION HISTORY
        history = data["history"]

        if self.history_table.rowCount() != len(history):
            self.history_table.setRowCount(len(history))

            for row, rep in enumerate(history):
                values = [
                    rep["rep"],
                    f"{rep['duration']:.2f} s",
                    f"{rep['min_knee_angle']:.1f} deg",
                    f"{rep['score']}/100"
                ]

                for col, value in enumerate(values):
                    self.history_table.setItem(
                        row, col,
                        QTableWidgetItem(str(value))
                    )

    def save_report(self):
        if self.worker is None or self.worker.isRunning():
            return

        try:
            session = self.worker.session
            counter = self.worker.counter

            if session is None or counter is None:
                return

            folder = session.save(
                list(counter.rep_history)
            )

            with open(
                folder / "session.json",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

            pdf_path = ReportGenerator().generate(
                data,
                list(session.fps_history),
                folder
            )

            QMessageBox.information(
                self,
                "Report Saved",
                f"PDF report saved:\n{pdf_path}"
            )

        except Exception as exc:
            self.show_error(str(exc))

    def show_error(self, message):
        QMessageBox.critical(
            self, "ExerciseAI Studio Error", message
        )

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()

            if not self.worker.wait(5000):
                event.ignore()
                QMessageBox.warning(
                    self,
                    "Camera Busy",
                    "Camera is still stopping."
                )
                return

        event.accept()


def launch_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
