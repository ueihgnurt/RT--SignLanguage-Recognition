import numpy as np
import cv2
import mediapipe as mp
import pandas as pd 
from keras.models import load_model
#1247
model = load_model("./datacollection/model.h5")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)
num_class = pd.read_csv("./datacollection/ClassName.csv").to_numpy()
num_frame = []
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    image = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    results = holistic.process(image)
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.left_hand_landmarks:
        left = np.array([[landmark.x,landmark.y,landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten()
    else:
        left = np.zeros(21*3)
    if results.right_hand_landmarks:
        right = np.array([[landmark.x,landmark.y,landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten()
    else:
        right = np.zeros(21*3)
    if results.face_landmarks:
        face = np.array([[landmark.x,landmark.y,landmark.z] for landmark in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468*3)
    # classname = ["U"]
    row = list(left) + list(right) + list(face)
    num_frame.append(np.array(row))
    if len(num_frame) > 15:
        x = np.array(num_frame).reshape(1,16,1530)
        pred = model.predict(x)
        print(num_class[np.argmax(pred, axis=1)])
        num_frame.pop(0)
    frame = cv2.flip(frame,1)
    cv2.imshow('CV2',frame)
    # if count> 300:
    #     break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()             