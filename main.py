
from pathlib import Path
import cv2
import mediapipe as mp
import time
import pygame

# -----------------------------
# 1. File paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "hand_landmarker.task")
SOUNDS_DIR = BASE_DIR / "sounds_swaras"

# -----------------------------
# 2. MediaPipe setup
# -----------------------------
mp_vision = mp.tasks.vision

base_options = mp.tasks.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

pygame.mixer.init()

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# -----------------------------
# 3. Webcam setup
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    landmarker.close()
    raise RuntimeError("Cannot open webcam")

start_time = time.monotonic()
last_timestamp_ms = -1

sounds_mapping = {
    1: SOUNDS_DIR / "1_Sa.mp3",
    2: SOUNDS_DIR / "2_Re.mp3",
    3: SOUNDS_DIR / "3_Ga.mp3",
    4: SOUNDS_DIR / "4_Ma.mp3",
    5: SOUNDS_DIR / "5_Pa.mp3",
    6: SOUNDS_DIR / "6_Dha.mp3",
    7: SOUNDS_DIR / "7_Ni.mp3",
    8: SOUNDS_DIR / "8_Sa_upper.mp3"
}

previous_count = -1

# -----------------------------
# 4. Main webcam loop
# -----------------------------
try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Generate timestamp
        timestamp_ms = max(
            int((time.monotonic() - start_time) * 1000),
            last_timestamp_ms + 1
        )
        last_timestamp_ms = timestamp_ms

        # Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Reset finger count for each frame
        total_fingers = 0

        height, width, _ = frame.shape

        # Process each detected hand
        for hand_landmarks in result.hand_landmarks:

            # Draw hand connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]

                start_point = (
                    int(start.x * width),
                    int(start.y * height)
                )
                end_point = (
                    int(end.x * width),
                    int(end.y * height)
                )

                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    (0, 255, 0),
                    2
                )

            # Draw all 21 landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 0, 255),
                    -1
                )

            # Count non-thumb fingers
            finger_pairs = [
                (8, 6),    # Index
                (12, 10),  # Middle
                (16, 14),  # Ring
                (20, 18)   # Pinky
            ]

            for tip, pip in finger_pairs:
                if hand_landmarks[tip].y < hand_landmarks[pip].y:
                    total_fingers += 1

        # Display finger count
        cv2.putText(
            frame,
            f"Fingers: {total_fingers}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        # Display number of detected hands
        cv2.putText(
            frame,
            f"Hands: {len(result.hand_landmarks)}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

        # Play sound only when the finger count changes
        if total_fingers != previous_count:

            if total_fingers == 0:
                pygame.mixer.music.stop()

            elif total_fingers in sounds_mapping:
                sound_path = sounds_mapping[total_fingers]

                if sound_path.is_file():
                    pygame.mixer.music.load(str(sound_path))
                    pygame.mixer.music.play()
                else:
                    print(f"Sound file missing: {sound_path}")

        previous_count = total_fingers

        # Display webcam
        cv2.imshow("Gesture Music Player", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    landmarker.close()
    pygame.mixer.quit()
    cv2.destroyAllWindows()
