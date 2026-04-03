# Hand Gesture Controlled System

A complete real-time computer vision project that uses a webcam to detect hand gestures and control desktop functions such as volume and mouse interactions.

## Problem Statement
Traditional keyboard-and-mouse input works well, but in touchless scenarios (presentations, accessibility use cases, smart displays, and interactive demos), users need a natural and intuitive control mechanism.

The challenge is to build a low-latency, stable, and practical hand-gesture interface that can:
- Track hand landmarks in real time
- Interpret gestures reliably
- Control system-level actions with minimal jitter

## Solution Overview
This project uses MediaPipe Hands for landmark detection and OpenCV for live video processing. Gesture logic is mapped to system actions via PyAutoGUI.

Core pipeline:
1. Capture webcam frames
2. Detect hand landmarks
3. Extract fingertip states and distances
4. Recognize gestures
5. Trigger control actions (volume/mouse)
6. Render overlays (landmarks, FPS, mode, volume)

## Features
- Real-time hand tracking with MediaPipe
- Landmark rendering on video stream
- Gesture recognition for:
  - Volume control (thumb-index distance)
  - Cursor movement (index finger only)
  - Click action (pinch gesture)
- Smoothing filters to reduce jitter in:
  - Volume updates
  - Cursor movement
- FPS counter overlay
- Volume level indicator overlay
- Configurable calibration constants

## Tech Stack
- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy

## Project Structure
```text
project/
|-- src/
|   |-- main.py
|   |-- hand_tracker.py
|   |-- gesture_controller.py
|-- utils/
|   |-- smoothing.py
|-- requirements.txt
|-- README.md
```

## Installation
### 1) Clone or copy the project folder
Place the project folder on your machine.

### 2) Create and activate a virtual environment
Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

Windows (CMD):
python -m venv .venv
.venv\Scripts\activate.bat

### 3) Install dependencies
pip install -r requirements.txt

## How to Run
From the project root:
python src/main.py

Press:
- q to quit
- Esc to quit

## Gesture Controls
1. Volume Control
- Gesture: Thumb up + index up, other fingers down
- Action: Distance between thumb tip and index tip maps to 0-100 volume level
- Smoothing: Running average + exponential smoothing to reduce jitter

2. Cursor Movement
- Gesture: Only index finger up
- Action: Index fingertip controls mouse position
- Smoothing: Exponential smoothing for stable pointer motion

3. Click (Pinch)
- Gesture: Thumb and index pinch close together
- Action: Single left click
- Debounce: Cooldown prevents rapid repeated clicks

## Calibration Guide
Tune these values based on your camera and hand distance:
- In src/gesture_controller.py
  - VolumeConfig.min_distance
  - VolumeConfig.max_distance
  - CursorConfig.frame_margin
  - smoothing_alpha values
- In src/main.py
  - AppConfig.click_pinch_threshold
  - frame_width and frame_height

Tips:
- If volume changes too aggressively, increase update interval or reduce key presses per tick.
- If cursor feels shaky, reduce smoothing alpha slightly.
- If click triggers accidentally, lower pinch threshold.

## Future Improvements
- Multi-hand gesture support (mode switching hand + control hand)
- Gesture-based drag and drop
- Custom gesture training profile per user
- On-screen calibration wizard
- Audio feedback for gesture state transitions
- Platform-specific direct volume APIs for finer control

## Notes
- PyAutoGUI controls your real mouse and volume keys.
- Keep your hand clearly visible to the camera for best tracking.
- Lighting significantly impacts detection accuracy.