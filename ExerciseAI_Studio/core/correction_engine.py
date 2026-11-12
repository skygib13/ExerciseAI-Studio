
class CorrectionEngine:
    """
    Rule-based squat technique assessment.

    Thresholds are initial experimental values,
    not clinically validated standards.
    """

    def __init__(self):
        self.depth_threshold = 105.0
        self.lean_threshold = 45.0

    def analyze(self, angles, stage):
        knee_values = [
            angles.get("left_knee"),
            angles.get("right_knee")
        ]

        valid_knees = [
            value for value in knee_values
            if value is not None
        ]

        knee = (
            sum(valid_knees) / len(valid_knees)
            if valid_knees else None
        )

        torso = angles.get("torso_inclination")

        feedback = []
        warnings = []

        if knee is None:
            return {
                "score": None,
                "feedback": ["Knee landmarks unavailable"],
                "warnings": ["Insufficient pose data"]
            }

        score = 100

        if stage == "down":
            if knee > self.depth_threshold:
                warnings.append("Insufficient squat depth")
                score -= 20
            else:
                feedback.append("Squat depth reached")

        if torso is not None:
            if torso > self.lean_threshold:
                warnings.append("Excessive forward lean")
                score -= 20
            else:
                feedback.append("Torso angle within threshold")
        else:
            warnings.append("Torso angle unavailable")

        score = max(0, score)

        return {
            "score": score,
            "feedback": feedback,
            "warnings": warnings
        }
