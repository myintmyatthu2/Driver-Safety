# Driver Safety System
[![Watch Demo Video](https://img.youtube.com/vi/VYiU78oYqcs/maxresdefault.jpg)](https://youtube.com/shorts/VYiU78oYqcs?si=GSAbw0e8mDhWi1kQ)

## Overview

This project implements a Driver Safety System using computer vision and deep learning techniques. The system monitors the driver’s face and eyes in real-time through a webcam, detects signs of drowsiness (closed eyes) and distraction (looking away), and plays an alarm to alert the driver. The goal is to reduce accidents caused by driver fatigue and inattentiveness.

## Features

- Real-time face and eye detection using Haar Cascades.
- Eye state classification (open/closed) using a trained deep learning model.
- Alerts when both eyes are closed (drowsiness detection).
- Alerts when the driver looks away for a sustained period.
- Alerts if no eyes are detected while face is present.
- Plays alarm sound only once per alert occurrence.
- Background music plays during driving.
- Configurable thresholds and paths in `config.py`.

## How It Works

- The system captures video from the webcam.
- Detects the largest face in each frame.
- Detects eyes within the face and crops eye regions.
- Uses a CNN model to classify each eye as open or closed.
- Maintains counters for consecutive frames with closed eyes or looking away.
- Triggers alarm if thresholds are met.
- Displays status and alerts on the screen.

## Requirements

- Python 3.x
- OpenCV
- TensorFlow / Keras
- NumPy
- Pygame (for audio playback)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/myintmyatthu2/Driver-Safety.git
   cd Driver-Safety
2. Install dependencies:
3. Make sure your webcam is connected.
4. Place your alarm sound file (WAV format) at the path specified in config.py.
5. python train_model.py if you don't not have trian model
6. Run the detection: python detect.py
Configuration
Adjust parameters in config.py, such as:

Camera index
Model path
Alarm sound path
Thresholds for eye closure, look-away, and no-eye detection

Developer Information
Developed by: Myint Myat Thu
Email: myintmyatthuoxygen200373mk@gmail.com
University: University of Computer Studies (Thaton)
Phone: +959988485929
