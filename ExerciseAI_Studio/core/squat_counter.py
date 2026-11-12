
import time


class SquatCounter:

    def __init__(
        self,
        up_threshold=160,
        down_threshold=105,
        min_duration=0.6,
        max_duration=10.0
    ):
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

        self.min_duration = min_duration
        self.max_duration = max_duration

        self.reps = 0
        self.stage = "unknown"

        self.start_time = None
        self.min_angle = None

        self.rep_history = []

    def update(self, knee_angle, timestamp=None):

        if timestamp is None:
            timestamp = time.perf_counter()

        if knee_angle is None:
            return self.get_status()

        # =====================================
        # INITIAL STANDING POSITION
        # =====================================

        if self.stage == "unknown":
            if knee_angle >= self.up_threshold:
                self.stage = "up"

            return self.get_status()

        # =====================================
        # START DESCENDING
        # =====================================

        if self.stage == "up":
            if knee_angle < self.up_threshold - 10:
                self.start_time = timestamp
                self.min_angle = knee_angle
                self.stage = "descending"

        # =====================================
        # DESCENDING MOVEMENT
        # =====================================

        elif self.stage == "descending":

            self.min_angle = min(
                self.min_angle,
                knee_angle
            )

            if knee_angle <= self.down_threshold:
                self.stage = "down"

            elif knee_angle >= self.up_threshold:
                self._reset_cycle()

        # =====================================
        # BOTTOM POSITION
        # =====================================

        elif self.stage == "down":

            self.min_angle = min(
                self.min_angle,
                knee_angle
            )

            if knee_angle > self.down_threshold + 10:
                self.stage = "ascending"

        # =====================================
        # ASCENDING MOVEMENT
        # =====================================

        elif self.stage == "ascending":

            self.min_angle = min(
                self.min_angle,
                knee_angle
            )

            if knee_angle >= self.up_threshold:

                duration = (
                    timestamp - self.start_time
                )

                if (
                    self.min_duration
                    <= duration
                    <= self.max_duration
                ):
                    self.reps += 1

                    score = self._calculate_score(
                        self.min_angle
                    )

                    self.rep_history.append({
                        "rep": self.reps,
                        "duration": round(duration, 2),
                        "min_knee_angle": round(
                            self.min_angle, 1
                        ),
                        "score": score,
                        "status": (
                            "Correct"
                            if score >= 80
                            else "Needs review"
                        )
                    })

                self._reset_cycle()

        return self.get_status()

    def _calculate_score(self, min_angle):
        """
        Experimental depth-only score.
        Not a complete technique assessment.
        """

        if min_angle <= 105:
            return 100

        elif min_angle <= 120:
            return 80

        elif min_angle <= 140:
            return 60

        return 40

    def _reset_cycle(self):
        self.stage = "up"
        self.start_time = None
        self.min_angle = None

    def get_status(self):
        return {
            "reps": self.reps,
            "stage": self.stage,
            "min_angle": self.min_angle,
            "rep_history": self.rep_history
        }

    def reset(self):
        self.__init__(
            up_threshold=self.up_threshold,
            down_threshold=self.down_threshold,
            min_duration=self.min_duration,
            max_duration=self.max_duration
        )
