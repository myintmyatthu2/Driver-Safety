# config.py
"""
Configuration for Driver Safety System
Adjust thresholds and paths here.
"""

# I/O paths
MODEL_DIR = "models"
MODEL_FILENAME = "eye_model.h5"
MODEL_PATH = f"{MODEL_DIR}/{MODEL_FILENAME}"

# Dataset settings (used in data_prep.py and train_model.py)
DATA_DIR = "Train_Dataset"         # should contain subfolders 'open' and 'closed'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Training settings (train_model.py)
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 4
LEARNING_RATE = 1e-4
FINE_TUNE_LR = 1e-5

# Detection settings (detect.py)
CAMERA_INDEX = 0                   # change to 1 if that works better
EYE_CLOSED_FRAMES_THRESHOLD = 1   # number of consecutive frames with closed eyes to trigger alarm
LOOK_AWAY_FRAMES_THRESHOLD = 12    # number of consecutive frames of looking away
NO_EYE_FRAMES_THRESHOLD = 30       # face present but no eyes detected
PREDICTION_THRESHOLD = 0.35       # sigmoid threshold for open/closed classification

# Alarm / audio
ALARM_WAV = r"D:\projects\driver_safety\video_2025-08-09_13-08-15.wav"
# config.py additions:


     # optional WAV file; if not present will use fallback beep
ALARM_DURATION_MS = 1000           # used for winsound fallback on Windows

# Misc
VERBOSE = True
