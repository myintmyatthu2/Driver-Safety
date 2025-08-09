# utils.py
import os
import threading
import platform
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model as tf_load_model

import config

# Ensure directories exist
os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(config.ALARM_WAV), exist_ok=True) if os.path.dirname(config.ALARM_WAV) else None

# --- Model loader helper ---
def load_model(path=None):
    """
    Load Keras model from given path or default config.MODEL_PATH.
    """
    model_path = path if path else config.MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train first (see train_model.py).")
    model = tf_load_model(model_path)
    return model

# --- Eye crop helper ---
def crop_eye_from_face(face_color, eye_box, target_size=config.IMAGE_SIZE):
    """
    Crop the eye region from face_color using eye_box (ex, ey, ew, eh),
    add small padding and resize to target_size.
    Returns normalized float array (H,W,3).
    """
    ex, ey, ew, eh = eye_box
    if ew == 0 or eh == 0:
        return None

    pad_w = int(ew * 0.25)
    pad_h = int(eh * 0.25)
    x1 = max(ex - pad_w, 0)
    y1 = max(ey - pad_h, 0)
    x2 = min(ex + ew + pad_w, face_color.shape[1])
    y2 = min(ey + eh + pad_h, face_color.shape[0])

    eye = face_color[y1:y2, x1:x2]
    if eye.size == 0:
        return None

    eye = cv2.resize(eye, target_size)
    eye = eye.astype('float32') / 255.0
    return eye

# --- Non-blocking alarm play ---
def _play_wav_simpleaudio(path):
    """
    Play WAV via simpleaudio (non-blocking).
    """
    try:
        import simpleaudio as sa
    except Exception:
        return False

    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()  # returns immediately
        # optionally: play_obj.wait_done()  # blocking if you want
        return True
    except Exception:
        return False

def _play_winsound(duration_ms):
    """
    Use winsound.Beep on Windows (blocking for duration). Wrapped in thread.
    """
    try:
        import winsound
    except Exception:
        return False
    freq = 2500
    winsound.Beep(freq, duration_ms)
    return True

def _terminal_beep():
    """
    Try terminal bell as fallback.
    """
    print("\a")
    return True

def play_alarm_nonblocking(wav_path=config.ALARM_WAV, duration_ms=config.ALARM_DURATION_MS):
    """
    Play an alarm sound in a separate thread (non-blocking main loop).
    Prefer WAV via simpleaudio, then winsound on Windows, else terminal beep.
    """
    def _play():
        # 1) Try WAV via simpleaudio
        if wav_path and os.path.exists(wav_path):
            if _play_wav_simpleaudio(wav_path):
                return
        # 2) On Windows fallback to winsound
        if platform.system() == "Windows":
            try:
                _play_winsound(duration_ms)
                return
            except Exception:
                pass
        # 3) Terminal beep
        _terminal_beep()

    t = threading.Thread(target=_play, daemon=True)
    t.start()

# --- Logging helper ---
def vlog(msg):
    if config.VERBOSE:
        print(msg)
