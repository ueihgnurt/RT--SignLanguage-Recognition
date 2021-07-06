from tkinter import Tk, RIGHT, LEFT, BOTH, X, Y, filedialog, StringVar, FLAT, SUNKEN, GROOVE, RIDGE, RAISED
from tkinter.ttk import Frame, Button, Style, Entry, Label
import tkinter.font as TkFont
from tkinter.font import *
import tkinter as tk, threading
from PIL import Image,ImageTk
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from keras.models import load_model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
class Window(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='#b3b3b3')
        self.master = master
        self.init_window()
        self.holistic = mp_holistic.Holistic(static_image_mode=False,
                                        min_detection_confidence = 0.5,
                                        min_tracking_confidence = 0.5)
        self.toggle = False
        self.model =  load_model("./datacollection/model.h5")
        self.num_class = pd.read_csv("./class_name.csv").to_numpy()
    def init_window(self):
        self.master.title("Viet Nam Sign Language Detector")
        self.font0 = TkFont.Font(self, size=12)
        self.font = TkFont.Font(self, size=14)
        self.style = Style()
        self.style.theme_use("clam")
        self.pack(fill = BOTH, expand = 1)
        ###########
        frame= tk.Frame(self)
        frame.pack(fill=X)
        exit_button=tk.Button(frame, text = "Exit", bg="#b3b3b3", font=self.font,command=self.exit)
        exit_button.grid(row=0, column=1)
        detect_button=tk.Button(frame, text = "Detect", bg="#b3b3b3", font=self.font,command=self.detect)
        detect_button.grid(row=0, column=2)
        frame2 = Frame(self, relief=GROOVE)
        frame2.pack(fill=BOTH)
        self.my_label = tk.Label(frame2)
        self.my_label.pack()
        self.sub = tk.Label(self.master,text = " ")
        self.sub.config(font=("Courier",70))
        self.sub.pack()
        self.video()

    def video(self):
        def stream(label):
            cap = cv2.VideoCapture(0)
            left_hands = []
            right_hands = []
            faces = []
            while cap:
                _,image = cap.read()
                image = cv2.resize(image, (1200, 650))
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                results = self.holistic.process(image)
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
                if self.toggle:
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                image = cv2.flip(image,1)
                frame_image = ImageTk.PhotoImage(Image.fromarray(image))
                left_hands.append(np.array(left))
                right_hands.append(np.array(right))
                faces.append(np.array(face))
                if len(left_hands)+len(right_hands)+len(faces) > 45:
                    L = np.array(left_hands).reshape(1,16,63)
                    R = np.array(right_hands).reshape(1,16,63)
                    F = np.array(faces).reshape(1,16,1404)
                    pred = self.model.predict([L,R,F])
                    left_hands.pop(0)
                    right_hands.pop(0)
                    faces.pop(0)
                    rs = self.num_class[np.argmax(pred, axis=1)][0][0]
                    if rs == "NONE":
                        self.sub['text']=" "
                    else:
                        self.sub['text'] = rs
                label.config(image=frame_image)
                label.image = frame_image
        thread = threading.Thread(target=stream, args=(self.my_label,))
        thread.daemon = 1
        thread.start()
    
    def detect(self):
        self.toggle= not (self.toggle)

    def exit(self):
        form.destroy()

if __name__ == '__main__':
    form = Tk()
    form.geometry("1200x800")
    app = Window(form)
    form.mainloop()