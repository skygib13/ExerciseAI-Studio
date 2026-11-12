
import time
import cv2

from core.yolo_engine import YOLOPoseEngine
from core.pose_adapter import PoseAdapter


def main():

    engine = YOLOPoseEngine(
        model_path="yolov8n-pose.pt",
        confidence=0.5
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    previous_time = time.perf_counter()

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            frame, pose = engine.process(frame)

            now = time.perf_counter()
            fps = 1.0 / max(
                now - previous_time, 1e-6
            )
            previous_time = now

            knee_angle = None

            if pose is not None:
                knee_angle = PoseAdapter.get_angle(
                    pose,
                    "right_hip",
                    "right_knee",
                    "right_ankle"
                )

            cv2.putText(
                frame,
                f"YOLO FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if knee_angle is not None:
                cv2.putText(
                    frame,
                    f"Right Knee: {knee_angle:.1f} deg",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            cv2.imshow(
                "ExerciseAI Studio - YOLOv8 Pose",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        engine.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
