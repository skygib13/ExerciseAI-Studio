
import numpy as np


class BiomechanicsEngine:

    def __init__(self, visibility_threshold=0.6):
        self.visibility_threshold = visibility_threshold

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculate the 2D joint angle at point B.

        A ---- B ---- C

        Returns angle in degrees or None.
        """

        if a is None or b is None or c is None:
            return None

        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        c = np.array(c, dtype=float)

        ba = a - b
        bc = c - b

        norm = (
            np.linalg.norm(ba)
            * np.linalg.norm(bc)
        )

        if norm < 1e-8:
            return None

        cosine = np.dot(ba, bc) / norm
        cosine = np.clip(cosine, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine))

        if not np.isfinite(angle):
            return None

        return round(float(angle), 1)

    def analyze(self, landmarks, frame_shape):
        """
        Analyze human biomechanics using
        MediaPipe Pose landmarks.

        Features:
        - Visibility filtering
        - Joint angle calculation
        - Torso inclination
        - Missing landmark handling
        """

        if landmarks is None or len(landmarks) < 33:
            return {}

        height, width = frame_shape[:2]

        # =====================================
        # 1. LANDMARK VALIDATION
        # =====================================

        def point(index):
            lm = landmarks[index]

            visibility = getattr(
                lm, "visibility", 0.0
            )

            if visibility < self.visibility_threshold:
                return None

            x = lm.x * width
            y = lm.y * height

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                return None

            # Reject points outside image
            if not (0 <= x < width and 0 <= y < height):
                return None

            return (x, y)

        # =====================================
        # 2. JOINT DEFINITIONS
        # =====================================

        joints = {
            "left_knee": (23, 25, 27),
            "right_knee": (24, 26, 28),

            "left_hip": (11, 23, 25),
            "right_hip": (12, 24, 26),

            "left_elbow": (11, 13, 15),
            "right_elbow": (12, 14, 16),

            "left_shoulder": (13, 11, 23),
            "right_shoulder": (14, 12, 24),
        }

        results = {}

        # =====================================
        # 3. CALCULATE JOINT ANGLES
        # =====================================

        for name, indices in joints.items():
            a, b, c = indices

            pa = point(a)
            pb = point(b)
            pc = point(c)

            results[name] = self.calculate_angle(
                pa, pb, pc
            )

        # =====================================
        # 4. TORSO INCLINATION
        # =====================================

        left_shoulder = point(11)
        right_shoulder = point(12)

        left_hip = point(23)
        right_hip = point(24)

        # Prefer the side with valid landmarks
        if left_shoulder and left_hip:
            shoulder = left_shoulder
            hip = left_hip

        elif right_shoulder and right_hip:
            shoulder = right_shoulder
            hip = right_hip

        else:
            shoulder = None
            hip = None

        if shoulder is not None and hip is not None:

            vertical = (
                hip[0],
                hip[1] - 100
            )

            results["torso_inclination"] = (
                self.calculate_angle(
                    shoulder,
                    hip,
                    vertical
                )
            )

        else:
            results["torso_inclination"] = None

        return results
