# Hand Gesture Swara Player: Student Guide

**Computer Vision Club, BITS Goa \| Week 4 Project**

Wave your hands at the webcam and make music. Hold up some fingers, and your program counts them and plays the matching **swara** (Sa Re Ga Ma Pa Dha Ni Sa).

This guide walks you through **what to build, why each step is needed, and how to debug it**. It deliberately does *not* contain the finished code. The snippets are small hints, not a solution. Use the Week 4 notes (`Week4_Notes_CV.pdf`) alongside it, especially Chapters 2 to 6\.

---

Note:- If you open this file in VSCode, press Ctrl+Shift+V for readible interface.

---

## 1\. What You Will Build

```
Webcam frame ──► Detect hand landmarks ──► Count raised fingers ──► Look up a sound ──► Play it
                      (MediaPipe)               (your logic)          (dictionary)       (Pygame)
```

&nbsp;

| Fingers shown (both hands total) | Result |
| :---- | :---- |
| 0 | Silence |
| 1 | Sa |
| 2 | Re |
| 3 | Ga |
| 4 | Ma |
| 5 | Pa |
| 6 | Dha |
| 7 | Ni |
| 8 | Sa (upper octave) |
| 9 to 10 | Silence (no sound assigned) |

Your finished program should:

- Show a mirrored live webcam window with the hand skeleton drawn on it  
- Display the current finger count on screen  
- Play the matching swara when a gesture is shown, and show its name on screen  
- Quit cleanly when you press **q**

---

## 2\. Setup

### Requirements

- Python 3.10  
- OpenCV (`opencv-python`)  
- MediaPipe (`mediapipe`)  
- Pygame (`pygame`)

python 3.10.11

mediapipe==0.10.14

opencv-python==5.0.0.93

pygame==2.6.1

numpy==2.2.6

```sh
pip install opencv-python mediapipe pygame
```

### Suggested folder layout

```
hand-gesture-music-player/
├── main.py          # your code
├── README.md        # this guide
└── sounds_swaras/   # the 8 swara audio files
    ├── 1_Sa.mp3
    ├── 2_Re.mp3
    ├── 3_Ga.mp3
    ├── 4_Ma.mp3
    ├── 5_Pa.mp3
    ├── 6_Dha.mp3
    ├── 7_Ni.mp3
    └── 8_Sa_upper.mp3
```

### Get your sounds

You are given **8 short audio clips** (`.mp3`), one for each swara: Sa, Re, Ga, Ma, Pa, Dha, Ni and the upper Sa. Each clip is about 1.5 seconds long. Put them all in a folder called `sounds_swaras/`.

The file names start with the finger count they belong to (`1_Sa.mp3` for 1 finger, `2_Re.mp3` for 2 fingers, and so on). Keep the names exactly as they are, because your dictionary in Milestone 6 must match them.

> **Tip:** Check that each file plays in a normal media player before blaming your code.

---

## 3\. Concepts You Need

### 3.1 Hand landmarks (Notes, Ch. 2\)

MediaPipe finds **21 numbered points** on every hand. Memorise this layout, because your finger-counting logic is built on it.

| Part | Landmark indices | Names (base → tip) |
| :---- | :---- | :---- |
| Wrist | 0 |  |
| Thumb | 1, 2, 3, **4** | CMC, MCP, IP, **TIP** |
| Index | 5, 6, 7, **8** | MCP, PIP, DIP, **TIP** |
| Middle | 9, 10, 11, **12** | MCP, PIP, DIP, **TIP** |
| Ring | 13, 14, 15, **16** | MCP, PIP, DIP, **TIP** |
| Pinky | 17, 18, 19, **20** | MCP, PIP, DIP, **TIP** |

### 3.2 Coordinates are fractions, not pixels

Each landmark has `x` and `y` between 0 and 1\. For *comparing* two landmarks you can use these directly. For *drawing*, you must convert to pixels:

```py
h, w, _ = frame.shape
px, py = int(lm.x * w), int(lm.y * h)
```

### 3.3 The y-axis points DOWN

In images, `y = 0` is the **top** of the frame. So a point that is physically *higher* has a **smaller** `y` value. This one fact is the key to detecting a raised finger.

### 3.4 Mirroring and handedness

- `cv2.flip(frame, 1)` mirrors the image so it feels like a mirror. Do this **before** detection.  
- MediaPipe also tells you whether each detected hand is `"Left"` or `"Right"`. You will need this for the thumb (see Step 5).

---

## 4\. Step-by-Step Implementation

Build in small milestones and **test after each one**. Do not write everything and then run it.

### Milestone 1: Webcam works

**Goal:** A window shows your mirrored webcam feed, and `q` closes it.

**Functions:** `cv2.VideoCapture(0)`, `cap.read()`, `cv2.flip()`, `cv2.imshow()`, `cv2.waitKey(1)`, `cap.release()`, `cv2.destroyAllWindows()`

**Checkpoint:** Move your hand and confirm the image is mirrored.

---

### Milestone 2: Create the detector (Notes, Ch. 3\)

**Goal:** Build a `HandLandmarker` **once, before the loop**.

The pattern is three lines: *where is the model → how should it run → build it.*

```py
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,                      # two hands -> counts up to 10
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)
```

**Why `num_hands=2`?** To reach counts of 6 to 10, you must track both hands.

**Why create it outside the loop?** It loads an AI model from disk. Doing that every frame will freeze your program.

**Remember:** call `landmarker.close()` at the end.

---

### Milestone 3: Detect hands in every frame (Notes, Ch. 4\)

**Goal:** Get `result.hand_landmarks` each frame.

Inside the loop, in this order:

1. Convert BGR to RGB: `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`  
2. Wrap it: `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)`  
3. Detect: `landmarker.detect_for_video(mp_image, timestamp_ms)`

Where `timestamp_ms` comes from the time since your program started:

```py
start_time = time.time()                              # before the loop
timestamp_ms = int((time.time() - start_time) * 1000) # inside the loop
```

**Checkpoint:** `print(len(result.hand_landmarks))` each frame. It should print `0` with no hand, `1` with one hand, and `2` with two.

> Use the **original BGR frame** for drawing and display. Only the RGB copy goes to MediaPipe.

---

### Milestone 4: Draw the skeleton (Notes, Ch. 5\)

**Goal:** See green lines and red dots on your hands.

Loop over each detected hand and call the drawing helper from your boilerplate:

```py
for hand_landmarks in result.hand_landmarks:
    draw_landmarks(frame, hand_landmarks)
```

**Checkpoint:** The skeleton should stick to your hand as you move. If it sits in the top-left corner, you forgot to convert fractions to pixels.

---

### Milestone 5: Count the fingers

This is the heart of the project. **Reset the count to 0 at the start of every frame**, then add each hand's fingers to it. With two hands you get a total from 0 to 10\.

#### The four fingers (index, middle, ring, pinky)

A finger is **up** when its tip is *above* the joint two steps below it. Remember that "above" means a smaller `y`.

Worked example for the index finger (tip \= 8, PIP joint \= 6):

```py
if hand_landmarks[8].y < hand_landmarks[6].y:
    count += 1
```

Now work out the other three yourself. Look at the table in 3.1 and spot the pattern: for each finger the joint you compare against is the tip index minus 2\.

#### The thumb (the tricky one)

The thumb moves **sideways**, not up and down, so a `y` comparison will not work.

Hints:

- Compare the **x** of the thumb tip (4) with the x of the joint below it (3).  
- Which direction means "thumb is out" **depends on whether it is a left or right hand**. You need handedness here.  
- You can read each hand's label like this (the lists line up, so index `i` matches the same hand):

```py
label = result.handedness[i][0].category_name   # "Left" or "Right"
```

- Because you loop over hands, use `enumerate(result.hand_landmarks)` to get `i`.

> **Stuck on the thumb?** Hold one hand up and `print(label, thumb_tip_x, thumb_joint_x)` with the thumb out, then tucked in. Watch which way the numbers move for each hand. If the thumb is counted backwards, swap your comparison direction. If *both* hands are backwards, check that you flipped the frame.

#### Pseudocode for this milestone

```
count = 0
for each detected hand (with its index i):
    label = handedness of hand i
    if thumb is out (depends on label):   count += 1
    for each of the 4 other fingers:
        if tip is above its lower joint:  count += 1
    draw the hand
draw the count on the frame with cv2.putText
```

**Checkpoint:** Show 0 to 5 fingers with one hand, then 6 to 10 with two hands. The number on screen must match every time. Do not move on until it is reliable.

---

### Milestone 6: Map counts to sounds

**Goal:** A dictionary that turns a finger count into a file path.

```py
sounds_mapping = {
    1: "sounds_swaras/1_Sa.mp3",
    # ... up to 8
}
```

Then a simple lookup: `if count in sounds_mapping:`. Count 0, 9 and 10 are simply not in the dictionary, so no sound plays for them.

**Design question:** The swaras go in order, so 1 finger is Sa and 8 fingers is the upper Sa. Why does this ordering feel natural? What happens as you raise your fingers one by one from a fist to an open hand? (Hint: your count passes through 1, 2, 3 and 4 on the way to 5\. Think about whether each of those should play a note.)

---

### Milestone 7: Play audio with Pygame

**Test audio on its own first**, in a separate tiny script, before connecting it to the camera:

```py
import pygame, time
pygame.mixer.init()                        # once, at the top
pygame.mixer.music.load("sounds_swaras/1_Sa.mp3")
pygame.mixer.music.play()
time.sleep(2)                              # keep the script alive long enough to hear it
```

Key functions: `pygame.mixer.init()`, `pygame.mixer.music.load()`, `.play()`, `.stop()`

Once that works, call it from your gesture lookup.

#### The trigger problem

Swara notes are about 1.5 seconds long, so they need to **ring out**. Do not call `.stop()` right after `.play()`, or you will only hear a tiny click.

The loop runs about 30 times per second. If you call `play()` every frame while a gesture is held, the sound restarts constantly and you hear stuttering noise. Decide how to handle this. Three common approaches:

| Approach | Idea | Trade-off |
| :---- | :---- | :---- |
| **Short delay** | Pause briefly after playing | Simple, but a blocking delay **freezes the video**, and a delay shorter than the note cuts it off |
| **Cooldown timer** | Only play if enough time has passed since the last sound (`time.time()`) | Does not block; needs one extra variable |
| **Trigger on change** | Only play when the count *differs* from the previous frame | Feels like a real "hit"; you must store `previous_count` |

> **Extra thought:** `pygame.mixer.music` plays one track at a time and loads from disk each time. If you want faster response or overlapping sounds, look at `pygame.mixer.Sound`, which you can load once into a dictionary before the loop.

---

### Milestone 8: Put it all together

Your final loop should follow this skeleton:

```
setup: init pygame, load sound mapping, create landmarker, open camera, record start_time

while camera is open:
    read frame (if it fails, skip this frame)
    flip frame
    convert to RGB -> wrap as mp.Image -> detect_for_video
    count = 0
    for each detected hand:
        count its fingers, add to total
        draw it
    if count has a sound (and trigger rules allow it):
        play it
    draw count on frame
    show frame
    if 'q' pressed: break

cleanup: release camera, destroy windows, close landmarker
```

---

## 5\. Debugging Guide

| Symptom | Likely cause | What to try |
| :---- | :---- | :---- |
| Camera window never opens | Another app (Zoom, Teams, Camera) is using the webcam | Close it. Try `VideoCapture(1)` if you have multiple cameras |
| Program is extremely slow or freezes | `HandLandmarker` created inside the loop | Move creation above the loop |
| Error about timestamps | Timestamps not increasing | Compute from `time.time() - start_time` every frame, never reuse a value |
| Skeleton in the top-left corner | Using raw 0 to 1 values for drawing | Multiply `x` by width and `y` by height, then `int()` |
| Landmarks never appear | Passing the BGR frame, or not wrapping in `mp.Image` | Re-check Milestone 3 order |
| Thumb counted wrong | Comparison direction flipped | Print thumb x values and the label, then fix the direction |
| Left and right hands both inverted | Frame not mirrored before detection | Flip first, detect second |
| Count flickers between numbers | Fingers half-bent, or the hand is turned sideways | Keep palm facing the camera; test your comparisons by printing values |
| Count is off by one or two | Wrong landmark index for a finger | Re-read the table in 3.1 |
| No sound at all | Wrong file path, or `mixer.init()` missing | Run the standalone audio test from Milestone 7 |
| `FileNotFoundError` for a sound | Path is relative to where you *run* the script | Run from the project folder, or check spelling and extension |
| Sound stutters or repeats | Calling `play()` every frame | Add a trigger rule (see Milestone 7\) |
| Note is cut off after a split second | `stop()` or a short delay right after `play()` | Let the note ring out; use trigger-on-change instead |
| Wrong notes play while raising fingers | The count passes through lower numbers on the way up | Only trigger once the count has stayed the same for a few frames |
| Video freezes briefly on each sound | A blocking delay in the loop | Switch to a cooldown timer or trigger-on-change |

**General debugging habits**

- `print()` your intermediate values (landmark counts, labels, finger count) before guessing.  
- Test one piece at a time: camera, then detection, then drawing, then counting, then audio.  
- Use `cv2.putText` to show debug values directly on the frame.

---

## 6\. Stretch Goals

Finished early? Try one of these:

- Show the **name of the sound** on screen next to the count  
- Add a **visual flash** or colour change when a sound triggers  
- Use `pygame.mixer.Sound` so two sounds can overlap  
- Add a key (for example `m`) to mute and unmute  
- Make the count more stable by requiring the same value for several frames before triggering  
- Detect a different gesture (fist, thumbs-up) and map it to **stop all sounds**  
- Give 9 and 10 fingers their own sounds (for example, higher-octave notes) by adding more clips  
- Show the swara name large on screen when it plays, and highlight it for a moment

---

## 7\. Submission Checklist

- [ ] Program runs with `python main.py` from a clean terminal  
- [ ] Webcam feed is mirrored and the skeleton is drawn on all detected hands  
- [ ] Finger count is displayed and correct for 0 to 10  
- [ ] Each count from 1 to 8 plays the correct swara (1 \= Sa ... 8 \= upper Sa), and 9 and 10 stay silent  
- [ ] Sounds do not stutter or freeze the video  
- [ ] Pressing `q` closes everything cleanly (camera released, windows closed, landmarker closed)  
- [ ] `sounds_swaras/` folder and file paths work on another machine  
- [ ] Code is commented where the logic is not obvious

---

## 8\. References

- Week 4 Notes: `Week4_Notes_CV.pdf` (HandLandmarker setup, detection, drawing, cheat-sheet)  
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)  
- [OpenCV Python docs](https://docs.opencv.org/)  
- [Pygame mixer docs](https://www.pygame.org/docs/ref/mixer.html)

**\#GO ERC\!\!\!\!\!**