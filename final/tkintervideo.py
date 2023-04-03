import cv2
import tkinter as tk
from PIL import Image, ImageTk

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        
        self.vid = cv2.VideoCapture(video_source)
        
        self.canvas = tk.Canvas(window, width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        
        self.btn_snapshot=tk.Button(window, text="Snapshot", command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        
        self.delay = 15
        self.update()
        
        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.read()
        
        if ret:
            cv2.imwrite("snapshot.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        ret, frame = self.vid.read()
        
        if ret:
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        
        self.window.after(self.delay, self.update)

App(tk.Tk(), "Tkinter and OpenCV")