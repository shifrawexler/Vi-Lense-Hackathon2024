
import cv2
import keras
import tensorflow as tf
import mediapipe as mp
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

global label
global IMAGE_HEIGHT
global IMAGE_SIZE
global original_size
global counterNV
global counterV

# print("hello")
# # Load your pre-trained machine learning model
# model = keras.models.load_model("cnn_lstm_model_PRO.hdf5")
# print("model is loaded")

def detect_video(frames, model):  # Classifies the frames by the model and returns the label
    predicted_labels_probabilities = np.zeros(shape=(1, 2))
    predicted_labels_probabilities = model.predict(frames)
    print(predicted_labels_probabilities)
    if predicted_labels_probabilities[0][0] > 0.5:
        label = "Non Violence"
    else:
        label = "Violence"
    return label


def main(video, model):
    label = "Non Violence"  # Default label

    cap = cv2.VideoCapture(video)  # Opens the video. If video==0, uses the computer camera

    # print(model.input_shape)  # Number of dimensions in the matrix (None, 10, 224, 224, 3)

    IMAGE_SIZE = 64
    resized_frames = np.zeros(shape=(1, 10, IMAGE_SIZE, IMAGE_SIZE, 3),
                              dtype=np.float32)  # float32 to minimize using RAM
    # frames = np.zeros(shape=(10, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)  # float32 to minimize using RAM

    counterV = 0  # Counter for the "Violence" label
    counterNV = 0  # Counter for the "Non Violence" label

    while True:
        frames_list = []

        ret, frame = cap.read()  # Reads the next frame from the video

        if ret == True:
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))  # Resizes the frame to the size the model knows
            frame = frame / 255  # Normalizes the frame
            for j in range(10):
                frames_list.append(frame)
        else:
            break

        # if len(frames_list) < 10:  # Checks if it failed to collect 10 frames
        #     break

        frames = np.array(frames_list).reshape(1, 10, IMAGE_SIZE, IMAGE_SIZE, 3)

        print('Took 10 Frames Successfully')
        resized_frames[0][:] = frames
        print(resized_frames.shape)

        print("Start detecting...")

        label = detect_video(resized_frames, model)  # Classifies the frames using the model
        if label == "Violence":
            counterV = counterV + 1
        else:
            counterNV = counterNV + 1
        print("!!!!!!!!!!!!!!!!!!!")
        print(label)

    cap.release()  # Closes the video
    cv2.destroyAllWindows()  # Closes all OpenCV windows

    total_class = counterNV + counterV
    percent_Non_Violence = counterNV / total_class  # Percentage of frames classified as "Non Violence"
    percent_Violence = counterV / total_class  # Percentage of frames classified as "Violence"
    print(f"The number of violence frames {counterV}")
    print(f"The number of non violence frames {counterNV}")
    if counterNV > counterV:
        return "Non Violence"  # Returns "Non Violence" if there are more such frames
    if counterNV < counterV:
        return "Violence"  # Returns "Violence" if there are more such frames
    if counterNV == counterV:
        return "Non Violence"  # If tied, returns "Non Violence"


# print(main("youtube_vidoes/Violence in a hospital.mp4"))
