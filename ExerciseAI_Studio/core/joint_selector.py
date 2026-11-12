
import math
from collections import deque
from statistics import median


class JointSelector:
    """
    Reliable knee selection for real-time
    squat monitoring.

    Uses:
    - Landmark visibility
    - Temporal stability
    - Side switching hysteresis
    """

    LEFT = (23, 25, 27)
    RIGHT = (24, 26, 28)

    def __init__(
        self,
        visibility_threshold=0.6,
        history_size=5,
        switch_margin=0.15,
        switch_frames=5
    ):
        self.visibility_threshold = visibility_threshold
        self.switch_margin = switch_margin
        self.switch_frames = switch_frames

        self.selected_side = None
        self.pending_side = None
        self.pending_count = 0

        self.history = {
            "left": deque(maxlen=history_size),
            "right": deque(maxlen=history_size)
        }

    @staticmethod
    def _visibility(landmarks, indices):
        values = []

        for index in indices:
            value = getattr(
                landmarks[index],
                "visibility",
                0.0
            )

            if not math.isfinite(value):
                return 0.0

            values.append(value)

        return min(values)

    def update(self, landmarks, angles):
        candidates = {}

        if landmarks is None or len(landmarks) < 33:
            return self._empty_result()

        for side, indices in (
            ("left", self.LEFT),
            ("right", self.RIGHT)
        ):
            angle = angles.get(f"{side}_knee")

            if angle is None or not math.isfinite(angle):
                continue

            visibility = self._visibility(
                landmarks, indices
            )

            if visibility < self.visibility_threshold:
                continue

            history = self.history[side]

            if history:
                difference = abs(
                    angle - median(history)
                )

                stability = max(
                    0.0,
                    1.0 - difference / 60.0
                )
            else:
                stability = 1.0

            quality = (
                0.8 * visibility
                + 0.2 * stability
            )

            candidates[side] = {
                "angle": angle,
                "quality": quality,
                "visibility": visibility
            }

        if not candidates:
            return self._empty_result()

        best_side = max(
            candidates,
            key=lambda s: candidates[s]["quality"]
        )

        if self.selected_side not in candidates:
            self.selected_side = best_side
            self.pending_side = None
            self.pending_count = 0

        elif best_side != self.selected_side:
            improvement = (
                candidates[best_side]["quality"]
                - candidates[self.selected_side]["quality"]
            )

            if improvement > self.switch_margin:
                if self.pending_side == best_side:
                    self.pending_count += 1
                else:
                    self.pending_side = best_side
                    self.pending_count = 1

                if self.pending_count >= self.switch_frames:
                    self.selected_side = best_side
                    self.pending_side = None
                    self.pending_count = 0
            else:
                self.pending_side = None
                self.pending_count = 0

        else:
            self.pending_side = None
            self.pending_count = 0

        for side, data in candidates.items():
            self.history[side].append(data["angle"])

        selected = candidates[self.selected_side]

        return {
            "angle": selected["angle"],
            "side": self.selected_side,
            "quality": round(selected["quality"], 3),
            "visibility": round(
                selected["visibility"], 3
            )
        }

    def _empty_result(self):
        self.pending_side = None
        self.pending_count = 0

        return {
            "angle": None,
            "side": None,
            "quality": 0.0,
            "visibility": 0.0
        }
