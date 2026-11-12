
import time
import cv2

from core.yolo_engine import YOLOPoseEngine
from core.pose_adapter import PoseAdapter


def calculate_knee(pose, side, width, height):
    return PoseAdapter.get_angle(
        pose,
        f"{side}_hip",
        f"{side}_knee",
        f"{side}_ankle",
        frame_width=width,
        frame_height=height,
    )


def main():
    engine = YOLOPoseEngine(
        model_path="yolov8n-pose.pt",
        confidence=0.5,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Camera not found")
        return

    previous_time = time.perf_counter()
    fps_smooth = None

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Camera frame error")
                break

            frame, pose = engine.process(frame)

            height, width = frame.shape[:2]

            left_knee = calculate_knee(
                pose, "left", width, height
            )

            right_knee = calculate_knee(
                pose, "right", width, height
            )

            now = time.perf_counter()
            elapsed = max(
                now - previous_time, 1e-6
            )
            previous_time = now

            instant_fps = 1.0 / elapsed

            if fps_smooth is None:
                fps_smooth = instant_fps
            else:
                fps_smooth = (
                    0.85 * fps_smooth
                    + 0.15 * instant_fps
                )

            cv2.rectangle(
                frame,
                (10, 10),
                (350, 155),
                (20, 20, 20),
                -1,
            )

            cv2.putText(
                frame,
                "EXERCISE AI STUDIO",
                (20, 37),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"YOLO FPS: {fps_smooth:.1f}",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            left_text = (
                f"{left_knee:.1f} deg"
                if left_knee is not None
                else "N/A"
            )

            right_text = (
                f"{right_knee:.1f} deg"
                if right_knee is not None
                else "N/A"
            )

            cv2.putText(
                frame,
                f"Left Knee: {left_text}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Right Knee: {right_text}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "ExerciseAI Studio - YOLOv8 Pose",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        engine.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
