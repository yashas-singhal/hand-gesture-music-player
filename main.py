
from pathlib import Path
import time

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ------------------------------
# 1. FILE PATHS AND SETTINGS
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "hand_landmarker.task"
SOUNDS_DIR = BASE_DIR / "sounds_swaras"

STABLE_FRAMES = 5
FLASH_SECONDS = 0.25
NAME_SECONDS = 0.60


# Connections between the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (0, 17), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20)
]


# Required swara files
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


# BONUS: Optional additional sounds
EXTRA_SOUND_FILES = {
    9: "9_High_Re.mp3",
    10: "10_High_Ga.mp3"
}


SWARA_NAMES = {
    1: "Sa",
    2: "Re",
    3: "Ga",
    4: "Ma",
    5: "Pa",
    6: "Dha",
    7: "Ni",
    8: "Sa (Upper)",
    9: "High Re",
    10: "High Ga"
}


# ------------------------------
# 2. DRAW HAND LANDMARKS
# ------------------------------

def draw_landmarks(frame, hand_landmarks):

    height, width = frame.shape[:2]
    points = []

    # Convert fractional coordinates into pixels
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw green connecting lines
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    # Draw red dots
    for point in points:
        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1
        )


# ------------------------------
# 3. COUNT RAISED FINGERS
# ------------------------------

def count_hand_fingers(hand, label):

    count = 0

    # Thumb: compare x coordinates
    if label == "Right":
        if hand[4].x < hand[3].x:
            count += 1

    elif label == "Left":
        if hand[4].x > hand[3].x:
            count += 1

    # Remaining four fingers: compare y coordinates
    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]

    for tip, pip in finger_pairs:
        if hand[tip].y < hand[pip].y:
            count += 1

    return count


# ------------------------------
# 4. BONUS: THUMBS-UP STOP GESTURE
# ------------------------------

def is_thumbs_up(hand):

    # Thumb should point upwards
    thumb_up = hand[4].y < hand[3].y < hand[2].y

    # Other four fingers should be folded
    folded_fingers = True

    for tip, pip in [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]:
        if hand[tip].y < hand[pip].y:
            folded_fingers = False

    return thumb_up and folded_fingers


# ------------------------------
# 5. MAIN PROGRAM
# ------------------------------

def main():

    # Check MediaPipe model file
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}"
        )

    # Check required sound files
    for filename in SOUND_FILES.values():
        path = SOUNDS_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing sound file: {path}"
            )

    # Initialize Pygame audio
    pygame.mixer.init()

    # BONUS: Load all sounds once using Sound
    sounds = {}

    for count, filename in SOUND_FILES.items():
        path = SOUNDS_DIR / filename
        sounds[count] = pygame.mixer.Sound(str(path))

    # BONUS: Load optional 9 and 10 finger sounds
    for count, filename in EXTRA_SOUND_FILES.items():
        path = SOUNDS_DIR / filename

        if path.is_file():
            sounds[count] = pygame.mixer.Sound(str(path))
        else:
            print(f"Optional sound not found: {filename}")

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
        last_timestamp = -1

        # BONUS: Stable gesture detection variables
        candidate_gesture = None
        candidate_frames = 0
        previous_gesture = None

        # BONUS: Mute and visual effects
        muted = False
        flash_until = 0
        name_until = 0
        last_played_name = ""

        while True:

            # Read webcam frame
            success, frame = cap.read()

            if not success:
                break

            # Mirror image
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Calculate timestamp
            timestamp = max(
                last_timestamp + 1,
                int((time.time() - start_time) * 1000)
            )

            last_timestamp = timestamp

            # Detect hands
            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            # Reset count for every frame
            count = 0
            stop_gesture = False

            # Process both hands
            for i, hand in enumerate(result.hand_landmarks):

                label = result.handedness[i][0].category_name

                count += count_hand_fingers(hand, label)

                draw_landmarks(frame, hand)

                # Check thumbs-up stop gesture
                if is_thumbs_up(hand):
                    stop_gesture = True

            # Decide current gesture
            if stop_gesture:
                gesture = "STOP"
            else:
                gesture = count

            # ------------------------------
            # BONUS: GESTURE STABILIZATION
            # ------------------------------

            if gesture == candidate_gesture:
                candidate_frames += 1

            else:
                candidate_gesture = gesture
                candidate_frames = 1

            # Play only after gesture is stable
            if candidate_frames >= STABLE_FRAMES:

                # Play only when gesture changes
                if gesture != previous_gesture:

                    if (
                        gesture == "STOP"
                        or gesture == 0
                        or gesture not in sounds
                    ):

                        # Stop all playing sounds
                        pygame.mixer.stop()

                        last_played_name = ""
                        name_until = 0

                    elif not muted:

                        # BONUS: Overlapping sounds
                        sounds[gesture].play()

                        # Remember swara name
                        last_played_name = SWARA_NAMES[gesture]

                        # BONUS: Start visual effects
                        flash_until = time.time() + FLASH_SECONDS
                        name_until = time.time() + NAME_SECONDS

                    previous_gesture = gesture

            # ------------------------------
            # 6. DISPLAY TEXT
            # ------------------------------

            if stop_gesture:
                swara_text = "STOP"

            elif count in sounds:
                swara_text = SWARA_NAMES[count]

            else:
                swara_text = "Silence"

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

            # Display swara
            cv2.putText(
                frame,
                f"Swara: {swara_text}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # BONUS: Display mute status
            if muted:
                status = "MUTED"
                status_color = (0, 0, 255)

            else:
                status = "Sound ON"
                status_color = (0, 255, 0)

            cv2.putText(
                frame,
                status,
                (20, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )

            cv2.putText(
                frame,
                "m: mute/unmute   q: quit",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # ------------------------------
            # BONUS: YELLOW VISUAL FLASH
            # ------------------------------

            if time.time() < flash_until:

                height, width = frame.shape[:2]

                cv2.rectangle(
                    frame,
                    (3, 3),
                    (width - 4, height - 4),
                    (0, 255, 255),
                    8
                )

            # ------------------------------
            # BONUS: LARGE SWARA NAME
            # ------------------------------

            if time.time() < name_until and last_played_name:

                cv2.putText(
                    frame,
                    last_played_name,
                    (130, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.8,
                    (0, 255, 255),
                    4
                )

            # Display webcam
            cv2.imshow(
                "Hand Gesture Swara Player",
                frame
            )

            # ------------------------------
            # BONUS: MUTE / UNMUTE
            # ------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("m"):

                muted = not muted

                if muted:
                    pygame.mixer.stop()
                    last_played_name = ""
                    flash_until = 0
                    name_until = 0

                else:
                    previous_gesture = None

    # ------------------------------
    # 7. CLEANUP
    # ------------------------------

    finally:

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        if landmarker is not None:
            landmarker.close()

        pygame.mixer.quit()


if __name__ == "__main__":
    main()
