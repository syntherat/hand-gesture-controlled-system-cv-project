from collections import deque
from typing import Deque, Optional, Tuple


class ExponentialSmoother:
    """Simple exponential moving average smoother for scalar values."""

    def __init__(self, alpha: float = 0.2, initial_value: Optional[float] = None) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0, 1].")
        self.alpha = alpha
        self.value = initial_value

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = float(new_value)
        else:
            self.value = (self.alpha * float(new_value)) + ((1.0 - self.alpha) * self.value)
        return self.value


class RunningAverage:
    """Sliding-window average smoother for scalar values."""

    def __init__(self, window_size: int = 5) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        self.window: Deque[float] = deque(maxlen=window_size)

    def update(self, new_value: float) -> float:
        self.window.append(float(new_value))
        return sum(self.window) / len(self.window)


class PointSmoother:
    """Smooths 2D points independently on x and y axes."""

    def __init__(self, alpha: float = 0.25) -> None:
        self.x_smoother = ExponentialSmoother(alpha=alpha)
        self.y_smoother = ExponentialSmoother(alpha=alpha)

    def update(self, point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = point
        return self.x_smoother.update(x), self.y_smoother.update(y)
