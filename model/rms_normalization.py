import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.asarray(x, dtype=float)
        gamma = np.asarray(gamma, dtype=float)

        x_hat = x / np.sqrt(np.mean(x**2) + eps)

        return np.round(gamma * x_hat, 4)
        pass
