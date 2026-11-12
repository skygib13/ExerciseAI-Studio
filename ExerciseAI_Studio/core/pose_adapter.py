
from dataclasses import dataclass
from typing import Dict, Optional
import math


@dataclass
class Joint:
    x: float
    y: float
    confidence: float = 1.0
    z: Optional[float] = None


@dataclass
class UnifiedPose:
    model: str
    joints: Dict[str, Joint]
    detected: bool = False


class PoseAdapter:
    """
    Unified 12-joint representation for
    MediaPipe Pose and YOLOv8-Pose.

    Coordinates:
        x, y: normalized image coordinates
        confidence: 0..1
        z: optional, model-specific depth
    """

    MEDIAPIPE_INDICES = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }

    YOLO_INDICES = {
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    JOINT_NAMES = list(MEDIAPIPE_INDICES.keys())

    @staticmethod
    def _valid(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def from_mediapipe(cls, landmarks):
        if landmarks is None:
            return UnifiedPose("MediaPipe Pose", {}, False)

        points = getattr(landmarks, "landmark", landmarks)
        joints = {}

        for name, index in cls.MEDIAPIPE_INDICES.items():
            if index >= len(points):
                continue

            point = points[index]
            x = getattr(point, "x", None)
            y = getattr(point, "y", None)

            if not cls._valid(x) or not cls._valid(y):
                continue

            visibility = getattr(
                point, "visibility", 1.0
            )
            if not cls._valid(visibility):
                visibility = 0.0

            z = getattr(point, "z", None)

            joints[name] = Joint(
                x=float(x),
                y=float(y),
                confidence=max(
                    0.0, min(1.0, float(visibility))
                ),
                z=float(z) if cls._valid(z) else None,
            )

        return UnifiedPose(
            model="MediaPipe Pose",
            joints=joints,
            detected=bool(joints),
        )

    @classmethod
    def from_yolo(
        cls,
        keypoints_xy,
        keypoints_conf,
        frame_width,
        frame_height,
    ):
        if keypoints_xy is None:
            return UnifiedPose("YOLOv8-Pose", {}, False)

        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(
                "Frame dimensions must be positive"
            )

        joints = {}

        for name, index in cls.YOLO_INDICES.items():
            if index >= len(keypoints_xy):
                continue

            x, y = keypoints_xy[index]

            if not cls._valid(x) or not cls._valid(y):
                continue

            confidence = 1.0

            if (
                keypoints_conf is not None
                and index < len(keypoints_conf)
            ):
                confidence = keypoints_conf[index]

            if not cls._valid(confidence):
                confidence = 0.0

            confidence = max(
                0.0, min(1.0, float(confidence))
            )

            if confidence < 0.3:
                continue

            joints[name] = Joint(
                x=float(x) / frame_width,
                y=float(y) / frame_height,
                confidence=confidence,
                z=None,
            )

        return UnifiedPose(
            model="YOLOv8-Pose",
            joints=joints,
            detected=bool(joints),
        )

    @staticmethod
    def get_joint(pose, name):
        if pose is None:
            return None
        return pose.joints.get(name)

    @staticmethod
    def get_angle(
        pose,
        first,
        middle,
        last,
        frame_width=640,
        frame_height=480,
    ):
        """
        Calculate 2D joint angle using
        pixel-scaled coordinate differences.
        """

        if pose is None:
            return None

        if frame_width <= 0 or frame_height <= 0:
            return None

        a = pose.joints.get(first)
        b = pose.joints.get(middle)
        c = pose.joints.get(last)

        if any(p is None for p in (a, b, c)):
            return None

        if min(
            a.confidence,
            b.confidence,
            c.confidence
        ) < 0.3:
            return None

        v1 = (
            (a.x - b.x) * frame_width,
            (a.y - b.y) * frame_height,
        )

        v2 = (
            (c.x - b.x) * frame_width,
            (c.y - b.y) * frame_height,
        )

        norm1 = math.hypot(*v1)
        norm2 = math.hypot(*v2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return None

        cosine = (
            v1[0] * v2[0] + v1[1] * v2[1]
        ) / (norm1 * norm2)

        cosine = max(-1.0, min(1.0, cosine))

        return round(
            math.degrees(math.acos(cosine)),
            2
        )
