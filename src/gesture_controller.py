import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pyautogui


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.smoothing import ExponentialSmoother, PointSmoother, RunningAverage


@dataclass
class VolumeConfig:
    min_distance: float = 20.0
    max_distance: float = 220.0
    smoothing_alpha: float = 0.25
    update_interval_seconds: float = 0.08
    max_key_presses_per_tick: int = 3


@dataclass
class CursorConfig:
    frame_margin: int = 120
    smoothing_alpha: float = 0.25


class GestureController:
    """Maps hand gestures to system controls."""

    def __init__(
        self,
        frame_size: Tuple[int, int],
        volume_config: VolumeConfig | None = None,
        cursor_config: CursorConfig | None = None,
    ) -> None:
        self.frame_width, self.frame_height = frame_size
        self.volume_config = volume_config or VolumeConfig()
        self.cursor_config = cursor_config or CursorConfig()

        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0

        self.volume_avg_filter = RunningAverage(window_size=5)
        self.volume_ema_filter = ExponentialSmoother(alpha=self.volume_config.smoothing_alpha)
        self.cursor_smoother = PointSmoother(alpha=self.cursor_config.smoothing_alpha)

        self.current_volume = 50.0
        self.target_volume = 50.0
        self.last_volume_update_time = 0.0
        self.last_click_time = 0.0
        self.click_cooldown_seconds = 0.35

    def map_distance_to_volume(self, distance: float) -> float:
        clipped = float(np.clip(distance, self.volume_config.min_distance, self.volume_config.max_distance))
        return float(np.interp(clipped, [self.volume_config.min_distance, self.volume_config.max_distance], [0, 100]))

    def update_volume(self, distance: float, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        raw_target = self.map_distance_to_volume(distance)
        averaged = self.volume_avg_filter.update(raw_target)
        self.target_volume = self.volume_ema_filter.update(averaged)

        if now - self.last_volume_update_time < self.volume_config.update_interval_seconds:
            return self.current_volume

        delta = self.target_volume - self.current_volume
        if abs(delta) < 2.0:
            return self.current_volume

        direction_key = "volumeup" if delta > 0 else "volumedown"
        presses = int(min(self.volume_config.max_key_presses_per_tick, max(1, abs(delta) // 8)))

        for _ in range(presses):
            pyautogui.press(direction_key)

        self.current_volume += presses * 2.5 if delta > 0 else -presses * 2.5
        self.current_volume = float(np.clip(self.current_volume, 0, 100))
        self.last_volume_update_time = now
        return self.current_volume

    def move_cursor(self, index_point: Tuple[int, int]) -> Tuple[float, float]:
        x, y = index_point
        m = self.cursor_config.frame_margin

        x_clamped = int(np.clip(x, m, self.frame_width - m))
        y_clamped = int(np.clip(y, m, self.frame_height - m))

        screen_x = np.interp(x_clamped, [m, self.frame_width - m], [0, self.screen_width])
        screen_y = np.interp(y_clamped, [m, self.frame_height - m], [0, self.screen_height])

        smooth_x, smooth_y = self.cursor_smoother.update((screen_x, screen_y))
        pyautogui.moveTo(smooth_x, smooth_y, duration=0)
        return smooth_x, smooth_y

    def pinch_click(self, pinch_distance: float, threshold: float = 35.0, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        if pinch_distance < threshold and (now - self.last_click_time) > self.click_cooldown_seconds:
            pyautogui.click()
            self.last_click_time = now
            return True
        return False

    def draw_volume_indicator(
        self,
        frame,
        origin: Tuple[int, int] = (50, 120),
        bar_size: Tuple[int, int] = (30, 220),
    ):
        x, y = origin
        w, h = bar_size

        cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 2)

        fill_height = int(np.interp(self.current_volume, [0, 100], [0, h]))
        top_y = y + (h - fill_height)
        cv2.rectangle(frame, (x, top_y), (x + w, y + h), (0, 220, 120), cv2.FILLED)

        label = f"VOL {int(self.current_volume)}%"
        cv2.putText(frame, label, (x - 10, y + h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
