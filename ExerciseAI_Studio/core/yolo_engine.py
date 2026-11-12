
import cv2
from ultralytics import YOLO

from core.pose_adapter import PoseAdapter


class YOLOPoseEngine:

    def __init__(
        self,
        model_path="yolov8n-pose.pt",
        confidence=0.5,
        device=None
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def process(self, frame):
        """
        Returns:
            annotated_frame
            UnifiedPose or None
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()

        if (
            result.keypoints is None
            or len(result.keypoints) == 0
        ):
            return annotated_frame, None

        # Select the highest-confidence person
        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):
            person_index = int(
                result.boxes.conf.argmax().item()
            )
        else:
            person_index = 0

        keypoints_xy = (
            result.keypoints.xy[person_index]
            .cpu().numpy()
        )

        keypoints_conf = None

        if result.keypoints.conf is not None:
            keypoints_conf = (
                result.keypoints.conf[person_index]
                .cpu().numpy()
            )

        height, width = frame.shape[:2]

        pose = PoseAdapter.from_yolo(
            keypoints_xy,
            keypoints_conf,
            frame_width=width,
            frame_height=height
        )

        return annotated_frame, pose

    def close(self):
        pass
