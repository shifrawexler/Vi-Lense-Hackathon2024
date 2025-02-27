#import threading
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
global frame_original
def draw_class_on_image(label, frame):  # Draw the label on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    if label == "Violence":
        fontColor = (0, 0, 255)  # Red
    else:
        fontColor = (0, 255, 0)  # green
    thickness = 2
    lineType = 2
    cv2.putText(frame, str(label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return frame


def detect(model, frames):  # Classifies the frames by the model and return the label
    predicted_labels_probabilities = np.zeros(shape=(1, 2))
    predicted_labels_probabilities = model.predict(frames)
    print('#################################')
    print(predicted_labels_probabilities)
    if predicted_labels_probabilities[0][0] > 0.5:
        label = "Non Violence"
    else:
        label = "Violence"

    return label
def on_close(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    cv2.destroyWindow("image")

def main(video):
    label = "Non Violence"  # Default label

    cap = cv2.VideoCapture(video)  # video==0

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_close)

    model = keras.models.load_model(r'cnn_lstm_model_PRO.hdf5')  # load our model from Colab
    print(model.input_shape)  # The number of dimensions in the matrix (None, 10, 224, 224, 3)

    # i = 0
    # warm_up_frames = 2
    IMAGE_SIZE = 64
    resized_frames = np.zeros(shape=(1, 10, IMAGE_SIZE, IMAGE_SIZE, 3),
                              dtype=np.float32)  # float32 to minimze using RAM
    frames = np.zeros(shape=(10, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)  # float32 to minimze using RAM

    counterV = 0
    counterNV = 0

    while True:
        frames_list = []
        if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
            break

        ret, frame = cap.read()

        if ret == True:
            frame_original = cv2.resize(frame, (500, 500))
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))  # Adjusts the frame to the size the model knows
            frame = frame / 255 # normalizing
            for j in range(10):
                frames_list.append(frame)
        else:
            break

        if len(frames_list) < 10:
            break

        frames = np.array(frames_list).reshape(1, 10, IMAGE_SIZE, IMAGE_SIZE, 3)

        print('Took 10 Frames Successfully')
        resized_frames[0][:] = frames
        print(resized_frames.shape)

        print("Start detecting...")

        label = detect(model, resized_frames)
        if label == "Violence":
            counterV = counterV + 1
        else:
            counterNV = counterNV + 1
        print("!!!!!!!!!!!!!!!!!!!")
        print(label)

        frame_original = draw_class_on_image(label, frame_original)
        cv2.imshow("image", frame_original)

        if cv2.waitKey(1) == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


    if counterNV > counterV:
        #print("Non Violence")
        return "Non Violence"
    if counterNV < counterV:
    #    #print("Violence")
        return "Violence"
    if counterNV == counterV:
       #print("counterNV = counterV")
       return "counterNV = counterV"


#main(0,"Models/cnn_lstm_model_PRO.hdf5")
main(0)