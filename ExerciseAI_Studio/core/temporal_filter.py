
from collections import deque
from statistics import median


class TemporalFilter:
    """
    Causal median filter for real-time
    joint angle smoothing.
    """

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, value):
        if value is None:
            self.history.clear()
            return None

        self.history.append(float(value))

        return round(
            median(self.history),
            1
        )

    def reset(self):
        self.history.clear()
