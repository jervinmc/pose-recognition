import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class MotionRecognitionGUI:
    def __init__(self, root, video_source=0):
        self.root = root
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.is_sleeping = False
        self.is_clapping = False
        self.label_action = tk.Label(self.root, text="Action: ")
        self.label_action.grid(row=0, column=0)
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.grid(row=1, columnspan=2)

        self.button_start = tk.Button(self.root, text="Start", command=self.start_capture)
        self.button_start.grid(row=2, column=0)
        self.button_stop = tk.Button(self.root, text="Stop", command=self.stop_capture)
        self.button_stop.grid(row=2, column=1)

    def start_capture(self):
        self.update_video()

    def stop_capture(self):
        self.cap.release()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            left_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HAND]
            right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HAND]
            if left_hand and right_hand:
                if left_hand.y < right_hand.y and abs(left_hand.x - right_hand.x) < 0.2:
                    self.is_clapping = True
                else:
                    self.is_clapping = False

            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER].y and results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].y:
                self.is_sleeping = True
            else:
                self.is_sleeping = False

            if self.is_sleeping:
                self.label_action.config(text="Action: Sleeping")
            elif self.is_clapping:
                self.label_action.config(text="Action: Clapping")
            else:
                self.label_action.config(text="Action: Idle")

            img = cv2.resize(frame, (640, 480))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.img_tk = img_tk 
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        self.root.after(10, self.update_video)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Motion Recognition")

    app = MotionRecognitionGUI(root)

    root.mainloop()