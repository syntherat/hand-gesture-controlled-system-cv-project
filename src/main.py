import time
from dataclasses import dataclass

import cv2

from gesture_controller import GestureController
from hand_tracker import HandTracker


@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    click_pinch_threshold: float = 34.0


def calculate_fps(previous_time: float, current_time: float) -> tuple[float, float]:
    delta = max(current_time - previous_time, 1e-6)
    fps = 1.0 / delta
    return fps, current_time


def main() -> None:
    config = AppConfig()

    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check camera index and permissions.")

    tracker = HandTracker(max_num_hands=1, detection_confidence=0.7, tracking_confidence=0.7)

    controller = GestureController(frame_size=(config.frame_width, config.frame_height))

    prev_time = time.time()
    active_mode = "Idle"

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame, hand_landmarks_list, handedness_list = tracker.find_hands(frame, draw=True)
            active_mode = "Idle"

            if hand_landmarks_list:
                hand_landmarks = hand_landmarks_list[0]
                handed_label = "Right"
                if handedness_list:
                    handed_label = handedness_list[0].classification[0].label

                landmarks = tracker.get_landmarks(frame, hand_landmarks)
                fingers = tracker.fingers_up(landmarks, handed_label=handed_label)

                thumb_index_distance = tracker.distance(landmarks, 4, 8)

                x_thumb, y_thumb, _ = landmarks[4]
                x_index, y_index, _ = landmarks[8]
                cv2.line(frame, (x_thumb, y_thumb), (x_index, y_index), (0, 180, 255), 2)
                cv2.circle(frame, (x_thumb, y_thumb), 7, (0, 200, 255), cv2.FILLED)
                cv2.circle(frame, (x_index, y_index), 7, (0, 200, 255), cv2.FILLED)

                only_index_up = fingers[1] and not fingers[2] and not fingers[3] and not fingers[4] and not fingers[0]
                if only_index_up:
                    ix, iy, _ = landmarks[8]
                    controller.move_cursor((ix, iy))
                    active_mode = "Cursor Move"

                    cv2.circle(frame, (ix, iy), 12, (50, 255, 120), 2)
                    margin = controller.cursor_config.frame_margin
                    cv2.rectangle(
                        frame,
                        (margin, margin),
                        (config.frame_width - margin, config.frame_height - margin),
                        (60, 120, 255),
                        2,
                    )

                thumb_and_index_up = fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]
                if thumb_and_index_up:
                    controller.update_volume(thumb_index_distance, now=time.time())
                    active_mode = "Volume Control"

                clicked = controller.pinch_click(
                    thumb_index_distance,
                    threshold=config.click_pinch_threshold,
                    now=time.time(),
                )
                if clicked:
                    active_mode = "Click"
                    cv2.circle(frame, (x_index, y_index), 16, (0, 0, 255), 3)

            controller.draw_volume_indicator(frame)

            curr_time = time.time()
            fps, prev_time = calculate_fps(prev_time, curr_time)

            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"Mode: {active_mode}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Hand Gesture Controlled System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
