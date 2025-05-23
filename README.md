# Facial Emotion Recognition

This project detects human emotions in real-time using a webcam feed. It uses a Convolutional Neural Network (CNN) model trained on the FER-2013 dataset and OpenCV for face detection and live video processing.

## Features

- Real-time face detection
- Emotion recognition: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Webcam-based input
- Simple and easy-to-run Python script

## Requirements

- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy

Install the required packages using:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository.
2. Make sure the files facial_emotion_detector.json and facial_emotion_detector.h5 are in the project folder.
3. Run the script:

```bash
python real_time_detection.py
```

## Files

- facial_emotion_detector.json – Model architecture

- facial_emotion_detector.h5 – Trained weights

- real_time_detection.py – Main script for real-time detection

- requirements.txt – Python dependencies
