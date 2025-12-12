"""
air_drums.py — Play drums with hand hits.

Place optional samples in 'drum_samples/':
 kick.wav, snare.wav, hihat.wav, tom.wav, crash.wav

Requirements:
 pip install opencv-python mediapipe pygame numpy
"""

import os, time
from collections import deque
import math

import cv2
import mediapipe as mp
import numpy as np
import pygame

# ---- Config ----
SAMPLES_DIR = "drum_samples"
PAD_NAMES = ["kick", "snare", "hihat", "tom", "crash"]
NUM_PADS = len(PAD_NAMES)
VELOCITY_THRESHOLD = 0.9      # tuning: bigger = less sensitive
PAD_COOLDOWN = 0.12           # seconds
SAMPLE_VOLUME = 0.9
SYNTH_SR = 44100
HISTORY_LEN = 6
# ----------------

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Pygame mixer (robust)
try:
    pygame.mixer.init(frequency=SYNTH_SR, size=-16, channels=1)
except Exception:
    pygame.mixer.init(frequency=SYNTH_SR)
mixer_info = pygame.mixer.get_init()
mixer_ch = mixer_info[2] if mixer_info else 2
pygame.mixer.set_num_channels(32)
print("mixer:", mixer_info)

# Load samples or synth
def load_drum_samples(folder):
    samples = {}
    have_files = False
    for name in PAD_NAMES:
        path = os.path.join(folder, f"{name}.wav")
        if os.path.exists(path):
            try:
                s = pygame.mixer.Sound(path)
                s.set_volume(SAMPLE_VOLUME)
                samples[name] = s
                have_files = True
            except Exception as e:
                print("Failed load", path, e)
                samples[name] = None
        else:
            samples[name] = None
    return samples, have_files

def synth_percussion(kind, dur=0.25, sr=SYNTH_SR, mixer_ch=mixer_ch):
    N = int(sr * dur)
    t = np.linspace(0, dur, N, endpoint=False)
    if kind == "kick":
        # low frequency exponential sweep
        freqs = np.linspace(100, 40, N)
        wave = np.sin(2*np.pi*freqs*t) * np.exp(-10*t)
    elif kind == "snare":
        noise = (np.random.randn(N) * np.exp(-15*t))
        wave = noise
    elif kind == "hihat":
        # filtered noise
        noise = np.random.randn(N) * (t*0 + 1.0)
        wave = noise * np.exp(-40*t)
    elif kind == "tom":
        freqs = np.linspace(200, 120, N)
        wave = np.sin(2*np.pi*freqs*t) * np.exp(-8*t)
    elif kind == "crash":
        noise = np.random.randn(N) * np.exp(-6*t)
        wave = noise * 0.6
    else:
        wave = np.random.randn(N) * np.exp(-10*t)

    arr = (wave / np.max(np.abs(wave)) * 32767).astype(np.int16)
    if mixer_ch == 2:
        arr = np.column_stack((arr, arr))
    s = pygame.sndarray.make_sound(arr)
    s.set_volume(SAMPLE_VOLUME)
    return s

samples, have = load_drum_samples(SAMPLES_DIR)
if not have:
    print("No drum sample files found — using synthesized percussion.")
    for name in PAD_NAMES:
        samples[name] = synth_percussion(name)

# pad cooldown tracker
last_hit = {name: 0.0 for name in PAD_NAMES}

# per-hand index-finger history: hand_index -> deque of (x,y,t)
hand_hist = {}  # will hold deques keyed by hand index (0 or 1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def pad_from_x(x_norm):
    idx = int(x_norm * NUM_PADS)
    idx = max(0, min(NUM_PADS-1, idx))
    return PAD_NAMES[idx], idx

print("Air Drums running. Hit downward with index finger to play pads.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    overlay = img.copy()
    now = time.time()

    # draw pads
    for i, name in enumerate(PAD_NAMES):
        x = int((i + 0.5) * w / NUM_PADS)
        cv2.rectangle(overlay, (int(i*w/NUM_PADS)+4, int(h*0.6)), (int((i+1)*w/NUM_PADS)-4, h-4), (40,40,40), -1)
        cv2.putText(overlay, name.upper(), (int(i*w/NUM_PADS)+10, int(h*0.6)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    if res.multi_hand_landmarks:
        for hid, hand_landmarks in enumerate(res.multi_hand_landmarks):
            # track per-hand history by index of the hand in list
            if hid not in hand_hist:
                hand_hist[hid] = deque(maxlen=HISTORY_LEN)
            lm = hand_landmarks.landmark
            ix, iy = lm[8].x, lm[8].y
            hand_hist[hid].append((ix, iy, now))

            # draw fingertip
            cx, cy = int(ix*w), int(iy*h)
            cv2.circle(overlay, (cx, cy), 8, (0,255,200), -1)
            # compute velocity
            hist = hand_hist[hid]
            if len(hist) >= 2:
                x0,y0,t0 = hist[0]
                x1,y1,t1 = hist[-1]
                dt = t1 - t0
                vy = (y1 - y0)/dt if dt>1e-6 else 0.0
                # detect downward hit
                if vy > VELOCITY_THRESHOLD:
                    pad_name, pad_idx = pad_from_x(ix)
                    if now - last_hit[pad_name] > PAD_COOLDOWN:
                        try:
                            samples[pad_name].play()
                        except Exception as e:
                            print("Play error:", e)
                        last_hit[pad_name] = now
                        cv2.circle(overlay, (int((pad_idx+0.5)*w/NUM_PADS), int(h*0.7)), 40, (0,180,255), 6)
                        print("Hit:", pad_name, "vy=", f"{vy:.2f}")

            mp_draw.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # clear histories occasionally
        to_clear = []
        for hid, hist in list(hand_hist.items()):
            if hist and now - hist[-1][2] > 0.35:
                to_clear.append(hid)
        for hid in to_clear:
            del hand_hist[hid]

    cv2.imshow("Air Drums", overlay)
    key = cv2.waitKey(1) & 0xFF
    if key==27 or key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
