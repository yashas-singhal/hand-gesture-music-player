
from pathlib import Path
import time

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# File paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
SOUNDS_DIR = BASE_DIR / "sounds_swaras"

# Connections between the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20)
]

# Finger count to audio file mapping
SOUND_FILES = {
    1: "1_Sa.mp3",
    2: "2_Re.mp3",
    3: "3_Ga.mp3",
    4: "4_Ma.mp3",
    5: "5_Pa.mp3",
    6: "6_Dha.mp3",
    7: "7_Ni.mp3",
    8: "8_Sa_upper.mp3"
}

SWARA_NAMES = {
    1: "Sa",
    2: "Re",
    3: "Ga",
    4: "Ma",
    5: "Pa",
    6: "Dha",
    7: "Ni",
    8: "Sa (Upper)"
}


# Draw hand skeleton
def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape

    points = [
        (int(lm.x * w), int(lm.y * h))
        for lm in hand_landmarks
    ]

    # Draw green connecting lines
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame, points[start], points[end],
            (0, 255, 0), 2
        )

    # Draw red landmark dots
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


# Count raised fingers of one hand
def count_hand_fingers(hand_landmarks, label):
    count = 0

    # Thumb: compare horizontal coordinates
    if label == "Right":
        if hand_landmarks[4].x < hand_landmarks[3].x:
            count += 1

    elif label == "Left":
        if hand_landmarks[4].x > hand_landmarks[3].x:
            count += 1

    # Index, middle, ring and pinky
    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]

    for tip, pip in finger_pairs:
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            count += 1

    return count


def main():

    # Verify model file
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # Create sound mapping
    sounds_mapping = {
        count: str(SOUNDS_DIR / filename)
        for count, filename in SOUND_FILES.items()
    }

    # Verify sound files
    for count, path in sounds_mapping.items():
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Sound file missing for {count} fingers: {path}"
            )

    # Initialize Pygame
    pygame.mixer.init()

    # Create MediaPipe HandLandmarker
    base_options = mp_python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2
    )

    landmarker = None
    cap = None

    try:
        landmarker = mp_vision.HandLandmarker.create_from_options(
            options
        )

        # Open webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        start_time = time.time()
        last_timestamp_ms = -1
        previous_count = None

        while True:

            # Read frame
            success, frame = cap.read()

            if not success:
                break

            # Mirror frame
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            # Convert to MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp in milliseconds
            timestamp_ms = max(
                last_timestamp_ms + 1,
                int((time.time() - start_time) * 1000)
            )
            last_timestamp_ms = timestamp_ms

            # Detect hands
            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            # Reset finger count every frame
            count = 0

            # Process each detected hand
            for i, hand_landmarks in enumerate(
                result.hand_landmarks
            ):

                label = result.handedness[i][0].category_name

                count += count_hand_fingers(
                    hand_landmarks, label
                )

                draw_landmarks(frame, hand_landmarks)

            # Play sound only when finger count changes
            if count != previous_count:

                if count in sounds_mapping:
                    pygame.mixer.music.load(
                        sounds_mapping[count]
                    )
                    pygame.mixer.music.play()

                else:
                    # Silence for 0, 9 and 10 fingers
                    pygame.mixer.music.stop()

                previous_count = count

            # Display finger count
            cv2.putText(
                frame,
                f"Fingers: {count}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Display swara name
            swara = SWARA_NAMES.get(count, "Silence")

            cv2.putText(
                frame,
                f"Swara: {swara}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # Show webcam frame
            cv2.imshow(
                "Hand Gesture Swara Player",
                frame
            )

            # Quit on pressing q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Release resources
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        if landmarker is not None:
            landmarker.close()

        pygame.mixer.quit()


if __name__ == "__main__":
    main()
