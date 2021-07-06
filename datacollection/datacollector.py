import numpy as np
import time
import threading
import cv2
import csv
import mediapipe as mp
import tensorflow as tf
#1247
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
num_frame = []
def datasave(n,classname):
    for i in n:
        with open('J.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(i)
    with open('J_class.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(classname)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    image = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    results = holistic.process(image)
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACE_OVAL)
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
    classname = ["J"]
    row = list(left) + list(right) + list(face)
    num_frame.append(row)
    save = threading.Thread(target=datasave,args=(num_frame,classname))
    if len(num_frame) > 15:
        count+=1
        datasave(num_frame,classname)
        num_frame.pop(0)
    frame = cv2.flip(frame,1)
    cv2.imshow('CV2',frame)
    if count> 300:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()             