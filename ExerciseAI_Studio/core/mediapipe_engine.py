import mediapipe as mp
import cv2

from pathlib import Path


class MediaPipeEngine:

    def __init__(self):

        model_path = (
            Path(__file__).resolve().parent.parent
            / "weights"
            / "pose_landmarker_lite.task"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = (
            mp.tasks.vision.PoseLandmarkerOptions
        )
        RunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=RunningMode.VIDEO,
            num_poses=1
        )

        self.landmarker = (
            PoseLandmarker.create_from_options(options)
        )

        self.timestamp_ms = 0

    def process(self, frame):

        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.timestamp_ms += 33

        result = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp_ms
        )

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            h, w = frame.shape[:2]

            for landmark in landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

            connections = (
                mp.tasks.vision.PoseLandmarksConnections
                .POSE_LANDMARKS
            )

            for connection in connections:

                start = landmarks[
                    connection.start
                ]

                end = landmarks[
                    connection.end
                ]

                p1 = (
                    int(start.x * w),
                    int(start.y * h)
                )

                p2 = (
                    int(end.x * w),
                    int(end.y * h)
                )

                cv2.line(
                    frame,
                    p1,
                    p2,
                    (255, 255, 0),
                    2
                )

        return frame, result

    def close(self):
        self.landmarker.close()