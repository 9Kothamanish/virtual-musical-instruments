"""
air_piano_complete_filtered.py — Air Piano (filtered triggers)

Improvements:
- Only trigger when finger crosses the key-play line (from above -> into key area) AND downward velocity.
- Per-finger cooldown and last-key memory.
- More robust glissando (requires larger horizontal speed).
- Keeps sustain, octave shift, chords, black keys, full-screen.

Requirements:
 pip install opencv-python mediapipe pygame numpy
"""

import os
import time
from collections import deque, defaultdict

import cv2
import mediapipe as mp
import numpy as np
import pygame

# ---------------- CONFIG ----------------
SAMPLES_DIR = "piano_samples"
TOTAL_KEYS = 24
BASE_MIDI_NOTE = 48
SAMPLE_VOLUME = 0.8
SYNTH_SR = 44100
KEY_HISTORY_LEN = 6
FINGER_LANDMARKS = [4, 8, 12, 16, 20]

# Triggering thresholds (tweak to taste)
PRESS_VY_THRESHOLD = 0.45       # downward velocity (normalized units/sec)
GLISS_VX_THRESHOLD = 1.2        # horizontal speed for glissando (higher to avoid accidental)
SUSTAIN_TOGGLE_VY = 1.0
OCTAVE_VX_THRESHOLD = 1.0
TOGGLE_COOLDOWN = 0.6
KEY_COOLDOWN = 0.12             # per-key global cooldown
FINGER_COOLDOWN = 0.18          # per-finger cooldown (prevent same finger spamming)
PLAY_LINE_Y = 0.55              # normalized y line: play only when finger crosses from y < PLAY_LINE_Y -> y >= PLAY_LINE_Y

# visuals
GLOW_ALPHA = 0.7
WHITE_KEY_COLOR = (245, 245, 245)
BLACK_KEY_COLOR = (30, 30, 30)
GLOW_COLOR = (0, 200, 255)

# octave offset range
OCTAVE_OFFSET_MIN = -1
OCTAVE_OFFSET_MAX = 1
# ----------------------------------------

# MediaPipe init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Pygame mixer init
try:
    pygame.mixer.init(frequency=SYNTH_SR, size=-16, channels=1)
except Exception:
    pygame.mixer.init(frequency=SYNTH_SR)
mixer_info = pygame.mixer.get_init()
if mixer_info is None:
    pygame.mixer.init(frequency=SYNTH_SR)
    mixer_info = pygame.mixer.get_init()
MIXER_CHANNELS = mixer_info[2]
pygame.mixer.set_num_channels(128)
print("pygame.mixer init:", mixer_info)

# synth helpers (same as before)
def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

def synth_piano_note(midi_note, dur=2.5, sr=SYNTH_SR, mixer_ch=MIXER_CHANNELS):
    freq = midi_to_freq(midi_note)
    N = int(sr * dur)
    t = np.linspace(0, dur, N, endpoint=False)
    env = np.exp(-3.5 * t)
    wave = np.zeros_like(t)
    for h, amp in enumerate([1.0, 0.5, 0.25, 0.12], start=1):
        wave += (amp) * np.sin(2 * np.pi * freq * h * t)
    wave *= env
    arr = (wave / np.max(np.abs(wave)) * 32767).astype(np.int16)
    if mixer_ch == 2:
        arr = np.column_stack((arr, arr))
    return pygame.sndarray.make_sound(arr)

# load samples
def load_key_samples(folder, n_keys):
    samples = []
    got_files = False
    for i in range(1, n_keys + 1):
        fname = os.path.join(folder, f"key{i}.wav")
        if os.path.exists(fname):
            try:
                s = pygame.mixer.Sound(fname)
                s.set_volume(SAMPLE_VOLUME)
                samples.append(s)
                got_files = True
            except Exception as ex:
                print("Failed to load", fname, ex)
                samples.append(None)
        else:
            samples.append(None)
    return samples, got_files

samples, have_files = load_key_samples(SAMPLES_DIR, TOTAL_KEYS)
if not have_files:
    print("No piano samples found — synthesizing notes.")
    for k in range(TOTAL_KEYS):
        midi = BASE_MIDI_NOTE + k
        samples[k] = synth_piano_note(midi, dur=3.0)
else:
    for k in range(TOTAL_KEYS):
        if samples[k] is None:
            samples[k] = synth_piano_note(BASE_MIDI_NOTE + k, dur=3.0)

# layout helpers
BLACK_INDICES_IN_OCTAVE = {1, 3, 6, 8, 10}

# state
key_last_play = [0.0] * TOTAL_KEYS
finger_history = {fid: deque(maxlen=KEY_HISTORY_LEN) for fid in FINGER_LANDMARKS}
finger_last_trigger_time = {fid: 0.0 for fid in FINGER_LANDMARKS}
finger_last_key = {fid: None for fid in FINGER_LANDMARKS}
hand_side_cache = {}
sustain_on = False
octave_offset = 0
last_toggle_time = 0.0

# Webcam + fullscreen
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

cv2.namedWindow("Air Piano Filtered", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Air Piano Filtered", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Air Piano Filtered running — PLAY_LINE_Y:", PLAY_LINE_Y)

# helpers
def x_to_key_index(x_norm, octave_off=0):
    idx = int(x_norm * TOTAL_KEYS)
    idx = max(0, min(TOTAL_KEYS - 1, idx))
    idx = idx + octave_off * 12
    idx = max(0, min(TOTAL_KEYS - 1, idx))
    return idx

def compute_velocity_smooth(hist):
    # compute velocity as median of small diffs to reduce jitter
    if len(hist) < 2:
        return 0.0, 0.0
    xs = [p[0] for p in hist]
    ys = [p[1] for p in hist]
    ts = [p[2] for p in hist]
    # compute pairwise velocities and take median
    vxs = []
    vys = []
    for i in range(1, len(hist)):
        dt = ts[i] - ts[i-1]
        if dt <= 1e-6:
            continue
        vxs.append((xs[i] - xs[i-1]) / dt)
        vys.append((ys[i] - ys[i-1]) / dt)
    if not vxs:
        return 0.0, 0.0
    vx = np.median(vxs)
    vy = np.median(vys)
    return vx, vy

def play_key(k):
    now = time.time()
    if now - key_last_play[k] < KEY_COOLDOWN:
        return
    try:
        samples[k].play()
    except Exception as e:
        print("Play error:", e)
    key_last_play[k] = now

def draw_keyboard_overlay(base_img, pressed_keys_set, screen_w, screen_h):
    img = base_img.copy()
    key_w = screen_w // TOTAL_KEYS
    # draw white keys
    for k in range(TOTAL_KEYS):
        x0 = k * key_w
        x1 = x0 + key_w
        cv2.rectangle(img, (x0, int(screen_h * 0.55)), (x1, screen_h - 4), WHITE_KEY_COLOR, -1)
        cv2.rectangle(img, (x0, int(screen_h * 0.55)), (x1, screen_h - 4), (180, 180, 180), 1)
    # draw black keys
    for k in range(TOTAL_KEYS):
        semitone = (k % 12)
        if semitone in BLACK_INDICES_IN_OCTAVE:
            x0 = int(k * key_w - key_w * 0.35)
            x1 = int(k * key_w + key_w * 0.65)
            y0 = int(screen_h * 0.55)
            y1 = int(screen_h * 0.55 + (screen_h - 4 - int(screen_h * 0.55)) * 0.6)
            x0 = max(0, x0)
            x1 = min(screen_w - 1, x1)
            cv2.rectangle(img, (x0, y0), (x1, y1), BLACK_KEY_COLOR, -1)
    # glow pressed keys
    glow = img.copy()
    for k in pressed_keys_set:
        x0 = k * key_w
        x1 = x0 + key_w
        cv2.rectangle(glow, (x0, int(screen_h * 0.55)), (x1, screen_h - 4), GLOW_COLOR, -1)
    cv2.addWeighted(glow, GLOW_ALPHA, img, 1 - GLOW_ALPHA, 0, img)
    # status
    status_text = f"Sustain: {'ON' if sustain_on else 'OFF'}    Octave offset: {octave_offset}"
    cv2.putText(img, status_text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
    # draw play line
    pl_y = int(screen_h * PLAY_LINE_Y)
    cv2.line(img, (0, pl_y), (screen_w, pl_y), (180, 180, 0), 1)
    return img

# main loop
pressed_keys_visual = set()
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.flip(frame, 1)
    screen_h, screen_w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    now = time.time()
    pressed_keys_frame = set()

    # map handedness
    hand_side_cache.clear()
    if res.multi_handedness:
        for idx, hh in enumerate(res.multi_handedness):
            hand_side_cache[idx] = hh.classification[0].label

    if res.multi_hand_landmarks:
        for hid, hand_landmarks in enumerate(res.multi_hand_landmarks):
            label = hand_side_cache.get(hid, None)
            lm = hand_landmarks.landmark

            # for every finger tracked
            for fid in FINGER_LANDMARKS:
                try:
                    fx = lm[fid].x
                    fy = lm[fid].y
                except Exception:
                    continue

                # append history
                finger_history[fid].append((fx, fy, now))

                # compute smooth velocity
                vx, vy = compute_velocity_smooth(finger_history[fid])

                # determine which key the finger is over now
                key_idx = x_to_key_index(fx, octave_offset)

                # fingertip normalized y values for crossing tests
                prev_y = finger_history[fid][0][1] if len(finger_history[fid]) >= 2 else None
                curr_y = finger_history[fid][-1][1]

                # per-finger cooldown check
                if now - finger_last_trigger_time[fid] < FINGER_COOLDOWN:
                    continue

                # RIGHT hand behavior: play or glissando
                if label == "Right" or (label is None and hid == 0):
                    # GLISS: require strong horizontal speed and finger above play line to intentionally sweep
                    if abs(vx) > GLISS_VX_THRESHOLD and (curr_y < PLAY_LINE_Y):
                        # play current key if not too soon
                        if now - key_last_play[key_idx] > KEY_COOLDOWN:
                            play_key(key_idx)
                            pressed_keys_frame.add(key_idx)
                            finger_last_trigger_time[fid] = now
                            finger_last_key[fid] = key_idx
                    else:
                        # Normal press: require crossing the play line top->down AND vy sufficient
                        crossed = (prev_y is not None) and (prev_y < PLAY_LINE_Y) and (curr_y >= PLAY_LINE_Y)
                        if crossed and vy > PRESS_VY_THRESHOLD:
                            # avoid retriggering same key for same finger immediately
                            if finger_last_key[fid] != key_idx or (now - key_last_play[key_idx] > KEY_COOLDOWN):
                                play_key(key_idx)
                                pressed_keys_frame.add(key_idx)
                                finger_last_trigger_time[fid] = now
                                finger_last_key[fid] = key_idx

                # LEFT hand gestures: sustain toggle or octave shift
                elif label == "Left":
                    # sustain toggle
                    if vy > SUSTAIN_TOGGLE_VY and (now - last_toggle_time) > TOGGLE_COOLDOWN:
                        sustain_on = not sustain_on
                        last_toggle_time = now
                        print("Sustain toggled ->", sustain_on)
                    # octave swipe
                    if vx > OCTAVE_VX_THRESHOLD and (now - last_toggle_time) > TOGGLE_COOLDOWN:
                        if octave_offset < OCTAVE_OFFSET_MAX:
                            octave_offset += 1
                            last_toggle_time = now
                            print("Octave offset ->", octave_offset)
                    elif vx < -OCTAVE_VX_THRESHOLD and (now - last_toggle_time) > TOGGLE_COOLDOWN:
                        if octave_offset > OCTAVE_OFFSET_MIN:
                            octave_offset -= 1
                            last_toggle_time = now
                            print("Octave offset ->", octave_offset)

            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # clear stale histories
        for fid in finger_history:
            if finger_history[fid] and now - finger_history[fid][-1][2] > 0.35:
                finger_history[fid].clear()

    # build pressed visuals
    for k in range(TOTAL_KEYS):
        if now - key_last_play[k] < 0.28:
            pressed_keys_visual.add(k)
    pressed_keys_visual |= pressed_keys_frame

    out = draw_keyboard_overlay(img, pressed_keys_visual, screen_w, screen_h)
    cv2.imshow("Air Piano Filtered", out)

    # cleanup visuals older than threshold
    pressed_keys_visual = {k for k in pressed_keys_visual if now - key_last_play[k] < 0.6}

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
