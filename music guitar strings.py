"""
air_guitar_fixed.py — Air Guitar (robust audio handling)

Requirements:
pip install opencv-python mediapipe pygame numpy

This version ensures generated audio arrays match the mixer channel count
(stereo vs mono) by checking pygame.mixer.get_init() after initialization.
"""

import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame

# ---------------- Config ----------------
SAMPLES_DIR = "guitar_samples"
NUM_STRINGS = 6
VELOCITY_THRESHOLD = 0.55
MIN_TIME_BETWEEN_PLUCKS = 0.12
SAMPLE_VOLUME = 0.8
SYNTH_DURATION = 2.0
SR = 44100
MAX_HISTORY = 6
# ----------------------------------------

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# --------- Pygame mixer init (robust) ----------
# Try to initialize; then read back actual mixer configuration.
try:
    pygame.mixer.init(frequency=SR, size=-16, channels=1)
except Exception:
    # fallback to default init
    pygame.mixer.init(frequency=SR)

mixer_info = pygame.mixer.get_init()  # returns (frequency, size, channels) or None
if mixer_info is None:
    # extremely rare: mixer not initialized; try default
    pygame.mixer.init()
    mixer_info = pygame.mixer.get_init()

mixer_channels = mixer_info[2]
print("pygame.mixer initialized:", mixer_info)  # e.g. (44100, -16, 2) => stereo

pygame.mixer.set_num_channels(16)
# ------------------------------------------------

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
    """
    Create an int16 numpy array shaped correctly for the mixer:
      - if mixer_ch == 1: shape (N,)  (mono)
      - if mixer_ch == 2: shape (N,2) (stereo)
    """
    N = int(sr * duration_s)
    t = np.linspace(0, duration_s, N, endpoint=False)
    env = np.exp(-6.0 * t)                 # quick decay for pluck-like tone
    wave = np.sin(2.0 * np.pi * freq_hz * t) * env
    arr = (wave * 32767).astype(np.int16)  # mono int16 shape (N,)

    if mixer_ch == 2:
        # duplicate to stereo -> shape (N,2)
        arr = np.column_stack((arr, arr))
    # else: keep 1-D array for mono

    sound = pygame.sndarray.make_sound(arr)
    sound.set_volume(SAMPLE_VOLUME)
    return sound

# Load or synth samples
samples, have_files = load_string_samples(SAMPLES_DIR, NUM_STRINGS)
if not have_files:
    print(f"No sample files found in '{SAMPLES_DIR}'. Using synthesized tones.")
    base_freqs = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
    for i in range(NUM_STRINGS):
        samples[i] = synthesize_pluck(base_freqs[i], duration_s=SYNTH_DURATION)
else:
    base_freqs = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
    for i in range(NUM_STRINGS):
        if samples[i] is None:
            samples[i] = synthesize_pluck(base_freqs[i], duration_s=SYNTH_DURATION)

for s in samples:
    if s:
        try:
            s.set_volume(SAMPLE_VOLUME)
        except Exception:
            pass

# Playback control trackers
last_pluck_time = [0.0] * NUM_STRINGS
history = deque(maxlen=MAX_HISTORY)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def fingertip_to_string(x_norm):
    idx = int(x_norm * NUM_STRINGS)
    idx = max(0, min(NUM_STRINGS - 1, idx))
    return idx

def compute_velocity(hist):
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

print("Air Guitar (fixed) running. Mixer channels:", mixer_channels)
print("Pluck by moving your index fingertip downward quickly. ESC/q to quit.")

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

    # draw string separators
    for i in range(NUM_STRINGS):
        x = int((i + 0.5) * w / NUM_STRINGS)
        cv2.line(overlay, (x, 0), (x, h), (60, 60, 60), 2)
        cv2.putText(overlay, str(i + 1), (x - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(overlay, hand, mp_hands.HAND_CONNECTIONS)
        lm = hand.landmark

        ixn = lm[8].x
        iyn = lm[8].y
        history.append((ixn, iyn, now))
        vx, vy = compute_velocity(history)

        cx, cy = int(ixn * w), int(iyn * h)
        cv2.circle(overlay, (cx, cy), 10, (0, 255, 255), -1)

        sidx = fingertip_to_string(ixn)
        sx = int((sidx + 0.5) * w / NUM_STRINGS)
        cv2.rectangle(overlay, (sx - 40, h - 60), (sx + 40, h - 10), (50, 50, 50), -1)
        cv2.putText(overlay, f"String {sidx+1}", (sx - 38, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if vy > VELOCITY_THRESHOLD:
            if now - last_pluck_time[sidx] > MIN_TIME_BETWEEN_PLUCKS:
                try:
                    samples[sidx].play()
                except Exception as e:
                    print("Play error:", e)
                last_pluck_time[sidx] = now
                cv2.circle(overlay, (sx, int(h * 0.45)), 30, (0, 180, 255), 6)
                print(f"Plucked string {sidx+1} (vy={vy:.2f})")
    else:
        if history and now - history[-1][2] > 0.35:
            history.clear()

    cv2.imshow("Air Guitar (fixed)", overlay)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
