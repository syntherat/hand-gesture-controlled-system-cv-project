from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """MediaPipe based hand tracker wrapper."""

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )

    def find_hands(self, frame: np.ndarray, draw: bool = True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

        return frame, results.multi_hand_landmarks, results.multi_handedness

    @staticmethod
    def get_landmarks(
        frame: np.ndarray,
        hand_landmarks,
    ) -> Dict[int, Tuple[int, int, float]]:
        h, w, _ = frame.shape
        landmark_points: Dict[int, Tuple[int, int, float]] = {}

        for idx, landmark in enumerate(hand_landmarks.landmark):
            x_px, y_px = int(landmark.x * w), int(landmark.y * h)
            landmark_points[idx] = (x_px, y_px, landmark.z)

        return landmark_points

    @staticmethod
    def distance(
        landmarks: Dict[int, Tuple[int, int, float]],
        point_a: int,
        point_b: int,
    ) -> float:
        x1, y1, _ = landmarks[point_a]
        x2, y2, _ = landmarks[point_b]
        return float(np.hypot(x2 - x1, y2 - y1))

    @staticmethod
    def fingers_up(
        landmarks: Dict[int, Tuple[int, int, float]],
        handed_label: str = "Right",
    ) -> List[bool]:
        # Tip and lower-joint ids for thumb, index, middle, ring, pinky.
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [2, 6, 10, 14, 18]

        fingers_state: List[bool] = []

        thumb_tip_x = landmarks[tip_ids[0]][0]
        thumb_joint_x = landmarks[pip_ids[0]][0]

        if handed_label == "Right":
            fingers_state.append(thumb_tip_x > thumb_joint_x)
        else:
            fingers_state.append(thumb_tip_x < thumb_joint_x)

        for tip_id, pip_id in zip(tip_ids[1:], pip_ids[1:]):
            finger_tip_y = landmarks[tip_id][1]
            finger_joint_y = landmarks[pip_id][1]
            fingers_state.append(finger_tip_y < finger_joint_y)

        return fingers_state

    def release(self) -> None:
        self.hands.close()
