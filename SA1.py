import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load Emotion Model
model = load_model("emotion_model.h5")

emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']

print("Model Loaded Successfully")

