"""
air_guitar_chords.py — Air Guitar with chord support

Usage:
- Place optional samples in 'guitar_samples/string1.wav' ... 'string6.wav'.
- Run: python air_guitar_chords.py
- Pluck strings with any combination of fingers (thumb/index/middle/ring/pinky).
- Simultaneous plucks across fingers produce chords.
"""

import os
import time
from collections import deque, defaultdict

import cv2
import mediapipe as mp
import numpy as np
import pygame

# ---------------- Config ----------------
SAMPLES_DIR = "guitar_samples"
NUM_STRINGS = 6
# Velocity threshold (normalized units/sec). Increase if too sensitive.
VELOCITY_THRESHOLD = 0.55
MIN_TIME_BETWEEN_PLUCKS = 0.12   # per-string cooldown (s)
SAMPLE_VOLUME = 0.8
SYNTH_DURATION = 2.0
SR = 44100
MAX_HISTORY = 6                   # frames of history to compute velocity
FINGER_LANDMARKS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
# ----------------------------------------

# MediaPipe init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ---------------- Pygame mixer (robust) ----------------
try:
    pygame.mixer.init(frequency=SR, size=-16, channels=1)
except Exception:
    pygame.mixer.init(frequency=SR)

mixer_info = pygame.mixer.get_init()
if mixer_info is None:
    pygame.mixer.init(frequency=SR)
    mixer_info = pygame.mixer.get_init()

mixer_channels = mixer_info[2]
print("pygame.mixer initialized:", mixer_info)
pygame.mixer.set_num_channels(32)
# ------------------------------------------------------

def load_string_samples(folder, num_strings):
    samples = []
    have_files = False
    for i in range(1, num_strings + 1):
        fname = os.path.join(folder, f"string{i}.wav")
        if os.path.exists(fname):
            try:
                s = pygame.mixer.Sound(fname)
                s.set_volume(SAMPLE_VOLUME)
                samples.append(s)
                have_files = True
            except Exception as e:
                print(f"Warning: couldn't load {fname}: {e}")
                samples.append(None)
        else:
            samples.append(None)
    return samples, have_files

def synthesize_pluck(freq_hz, duration_s=SYNTH_DURATION, sr=SR, mixer_ch=mixer_channels):
    N = int(sr * duration_s)
    t = np.linspace(0, duration_s, N, endpoint=False)
    env = np.exp(-6.0 * t)
    wave = np.sin(2.0 * np.pi * freq_hz * t) * env
    arr = (wave * 32767).astype(np.int16)
    if mixer_ch == 2:
        arr = np.column_stack((arr, arr))
    sound = pygame.sndarray.make_sound(arr)
    sound.set_volume(SAMPLE_VOLUME)
    return sound

# Load or synthesize samples
samples, have_files = load_string_samples(SAMPLES_DIR, NUM_STRINGS)
base_freqs = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]  # E2 A2 D3 G3 B3 E4

if not have_files:
    print(f"No sample files found in '{SAMPLES_DIR}'. Using synthesized tones.")
    for i in range(NUM_STRINGS):
        samples[i] = synthesize_pluck(base_freqs[i], duration_s=SYNTH_DURATION)
else:
    for i in range(NUM_STRINGS):
        if samples[i] is None:
            samples[i] = synthesize_pluck(base_freqs[i], duration_s=SYNTH_DURATION)

for s in samples:
    if s:
        try:
            s.set_volume(SAMPLE_VOLUME)
        except Exception:
            pass

# Per-string cooldown
last_pluck_time = [0.0] * NUM_STRINGS

# Per-finger history: landmark_id -> deque of (x_norm, y_norm, t)
finger_history = {fid: deque(maxlen=MAX_HISTORY) for fid in FINGER_LANDMARKS}

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def fingertip_to_string(x_norm):
    """ Map normalized x [0..1] to string index 0..NUM_STRINGS-1 """
    idx = int(x_norm * NUM_STRINGS)
    idx = max(0, min(NUM_STRINGS - 1, idx))
    return idx

def compute_velocity(hist):
    """ Return (vx, vy) from oldest to newest sample in history """
    if len(hist) < 2:
        return 0.0, 0.0
    x0, y0, t0 = hist[0]
    x1, y1, t1 = hist[-1]
    dt = t1 - t0
    if dt <= 1e-6:
        return 0.0, 0.0
    vx = (x1 - x0) / dt
    vy = (y1 - y0) / dt
    return vx, vy

print("Air Guitar (chords) running.")
print("Pluck with any combination of fingers. ESC/q to quit.")
print("Mixer channels:", mixer_channels, "| Samples folder:", SAMPLES_DIR)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    overlay = img.copy()
    now = time.time()

    # draw vertical string lines and labels
    for i in range(NUM_STRINGS):
        x = int((i + 0.5) * w / NUM_STRINGS)
        cv2.line(overlay, (x, 0), (x, h), (60, 60, 60), 2)
        cv2.putText(overlay, str(i + 1), (x - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # highlight the fingers area and detect plucks for each tracked finger
    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(overlay, hand, mp_hands.HAND_CONNECTIONS)
        lm = hand.landmark

        plucked_strings_this_frame = []  # accumulate for possible chord logging

        # update histories for each fingertip landmark we care about
        for fid in FINGER_LANDMARKS:
            # some landmarks might be missing if hand partial; safe-guard:
            try:
                fx = lm[fid].x
                fy = lm[fid].y
            except Exception:
                continue
            finger_history[fid].append((fx, fy, now))

            # compute velocity for this finger
            vx, vy = compute_velocity(finger_history[fid])

            # visualize fingertip
            cx, cy = int(fx * w), int(fy * h)
            cv2.circle(overlay, (cx, cy), 8, (0, 255, 200), -1)

            # map to string
            sidx = fingertip_to_string(fx)
            sx = int((sidx + 0.5) * w / NUM_STRINGS)
            # small rectangle showing which string this finger is over
            cv2.rectangle(overlay, (sx - 30, cy - 18), (sx + 30, cy + 18), (40, 40, 40), -1)
            cv2.putText(overlay, f"{sidx+1}", (sx - 10, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # detect downward pluck (vy positive because y increases downward)
            if vy > VELOCITY_THRESHOLD:
                # check per-string cooldown
                if now - last_pluck_time[sidx] > MIN_TIME_BETWEEN_PLUCKS:
                    try:
                        samples[sidx].play()
                    except Exception as e:
                        print("Play error:", e)
                    last_pluck_time[sidx] = now
                    plucked_strings_this_frame.append(sidx + 1)
                    # visual pluck marker
                    cv2.circle(overlay, (sx, int(h * 0.45)), 26, (0, 180, 255), 5)

        # If multiple strings were plucked this frame, print chord info
        if len(plucked_strings_this_frame) > 1:
            print("Chord:", plucked_strings_this_frame)

    else:
        # clear stale finger histories
        if any(len(hist) and now - hist[-1][2] > 0.35 for hist in finger_history.values()):
            for hist in finger_history.values():
                hist.clear()

    cv2.imshow("Air Guitar (chords)", overlay)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
