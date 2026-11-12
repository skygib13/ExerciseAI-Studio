
import csv
import json
import time

from datetime import datetime
from pathlib import Path


class SessionManager:

    def __init__(
        self,
        exercise="Squat",
        model="MediaPipe Pose"
    ):
        self.exercise = exercise
        self.model = model

        self.started_at = datetime.now()
        self.start_time = time.perf_counter()

        self.fps_history = []

    def update(self, fps):
        if fps > 0:
            self.fps_history.append({
                "elapsed_seconds": round(
                    time.perf_counter() - self.start_time,
                    3
                ),
                "fps": round(fps, 2)
            })

    def save(self, repetitions):

        ended_at = datetime.now()

        session_id = self.started_at.strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        root = (
            Path(__file__).resolve().parent.parent
            / "reports"
            / f"session_{session_id}"
        )

        root.mkdir(
            parents=True,
            exist_ok=True
        )

        duration = round(
            time.perf_counter() - self.start_time,
            2
        )

        fps_values = [
            item["fps"]
            for item in self.fps_history
        ]

        average_fps = (
            round(sum(fps_values) / len(fps_values), 2)
            if fps_values else None
        )

        depth_pass_reps = sum(
            1 for rep in repetitions
            if rep["status"] == "Correct"
        )

        summary = {
            "session_id": session_id,
            "exercise": self.exercise,
            "model": self.model,
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": duration,
            "total_repetitions": len(repetitions),
            "depth_pass_repetitions": depth_pass_reps,
            "average_fps": average_fps,
            "assessment_type": "experimental_depth_only",
            "repetitions": repetitions
        }

        # JSON report
        with open(
            root / "session.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False
            )

        # Repetition CSV
        with open(
            root / "repetitions.csv",
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "rep",
                    "duration",
                    "min_knee_angle",
                    "score",
                    "status"
                ]
            )

            writer.writeheader()
            writer.writerows(repetitions)

        # Performance CSV
        with open(
            root / "performance.csv",
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "elapsed_seconds",
                    "fps"
                ]
            )

            writer.writeheader()
            writer.writerows(self.fps_history)

        return root
