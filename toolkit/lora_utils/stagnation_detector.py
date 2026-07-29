import numpy as np
from collections import deque


class StagnationDetector:
    def __init__(self, window_size=100, threshold=0.001, epsilon=1e-10):
        """
        :param window_size: Analysis window size (larger window reduces noise)
        :param threshold: Maximum allowed coefficient of variation (0.001 = 0.1% deviation)
        :param epsilon: Numerical stability constant to avoid division by zero
        """
        self.history = deque(maxlen=window_size)
        self.threshold = threshold
        self.epsilon = epsilon

    def check(self, current_rms):
        """
        Returns:
        - is_stagnant (bool): True if stagnation is detected
        - cv (float): Current coefficient of variation value (useful for logging)
        """
        self.history.append(float(current_rms))

        if len(self.history) < self.history.maxlen:
            return False, 0.0

        history_array = np.array(self.history, dtype=np.float64)
        mean_val = np.mean(history_array)
        std_val = np.std(history_array)

        # Coefficient of variation (Relative Standard Deviation)
        cv = std_val / (abs(mean_val) + self.epsilon)
        is_stagnant = cv < self.threshold
        return is_stagnant, float(cv)

    def state_dict(self):
        """Serialize history (oldest → newest). Threshold/window stay on the instance."""
        return {"history": list(self.history)}

    def load_state_dict(self, d):
        """Restore history; truncate to current maxlen (keep newest). Does not change threshold/epsilon/window."""
        hist = d.get("history", [])
        maxlen = self.history.maxlen
        if maxlen is not None and len(hist) > maxlen:
            hist = hist[-maxlen:]
        self.history = deque(hist, maxlen=maxlen)
