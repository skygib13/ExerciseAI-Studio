# ExerciseAI Studio

A Python desktop application for real-time exercise monitoring using computer vision, pose estimation and biomechanical measurements.

## Features

- Webcam pose tracking with MediaPipe Pose Landmarker
- Squat repetition detection and exercise feedback
- Live PySide6 desktop dashboard
- Session logs and PDF report generation
- Optional YOLOv8 pose experiments in separate scripts

## Project structure

```text
ExerciseAI_Studio/
├── main.py                  # Desktop app entry point
├── core/                    # Pose, biomechanics, feedback, sessions and reports
├── ui/main_window.py        # PySide6 interface
├── weights/                  # MediaPipe model asset
├── yolov8n-pose.pt          # YOLO pose model (experimental scripts)
├── test_pose_adapter.py
├── test_report.py
├── test_yolo_camera.py
└── requirements.txt
```

## Installation

Recommended: Python 3.10 on Windows (other versions may work depending on dependency compatibility).

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

The desktop application requires a working webcam. On Windows, it opens camera index 0 with the DirectShow backend. Make sure the camera is available to desktop apps.

## Model assets

The supplied project includes `weights/pose_landmarker_lite.task` and `yolov8n-pose.pt`. Verify model license terms before redistributing these third-party weights.

## Privacy and reproducibility

Generated session reports and participant measurements in `reports/` are intentionally excluded from GitHub. The application may create this directory locally when recording a session. No benchmark accuracy claims are made here.

## Notes

Dependency versions are not pinned; consider locking versions after verifying a working installation on your machine. This repository packaging has not been runtime-tested with a camera.
