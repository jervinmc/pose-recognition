import tkinter as tk
import cv2
import mediapipe as mp
import threading
import speech_recognition as sr
r = sr.Recognizer()
mic = sr.Microphone()
mp_pose = mp.solutions.pose

detected = ''

def worker(num):
    global detected
    while True:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(f'detected {detected}')
            print(f"You said: ", text)
            if(detected==text):
                print('Correct!')
            else:
                print('Wrong!')

        except sr.UnknownValueError:
            print("Could not understand audio")

        except sr.RequestError as e:
            print("Error: {0}".format(e))

def start_detection():
    global detected
    t = threading.Thread(target=worker, args=('',))
    t.start()
    mp_pose = mp.solutions.pose
    clapping_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_WRIST, mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    hi_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, 
                    mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, 
                    mp.solutions.pose.PoseLandmark.LEFT_WRIST, mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    sleeping_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_EYE, mp.solutions.pose.PoseLandmark.RIGHT_EYE]
    pointing_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW, 
                      mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    stretch_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, 
                     mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    raising_arms_keypoints = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                          mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
                          mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]
    shaking_head_keypoints = [mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.LEFT_EAR, mp.solutions.pose.PoseLandmark.RIGHT_EAR]


    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:
        while True:
            ret, image = cap.read()
            if not ret:
                break
            image = cv2.flip(image, 1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose_detection.process(image_rgb)
            if results.pose_landmarks is not None:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.append(landmark)

                for i, landmark in enumerate(keypoints):
                    x, y, z = landmark.x, landmark.y, landmark.z
                    x, y = int(x * image.shape[1]), int(y * image.shape[0])
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                for connection in mp_pose.POSE_CONNECTIONS:
                    x0, y0 = int(keypoints[connection[0]].x * image.shape[1]), int(keypoints[connection[0]].y * image.shape[0])
                    x1, y1 = int(keypoints[connection[1]].x * image.shape[1]), int(keypoints[connection[1]].y * image.shape[0])
                    cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

                is_clapping = all([keypoints[k].visibility > 0.5 for k in clapping_keypoints])
                is_hi = all([keypoints[k].visibility > 0.5 for k in hi_keypoints])
                is_sleeping = all([keypoints[k].visibility > 0.5 for k in sleeping_keypoints])
                is_pointing = all([keypoints[k].visibility > 0.5 for k in pointing_keypoints])
                is_stretching = all([keypoints[k].visibility > 0.5 for k in stretch_keypoints])
                is_raising_arms = all([keypoints[k].visibility > 0.5 for k in raising_arms_keypoints])
                is_shaking_head = all([keypoints[k].visibility > 0.5 for k in shaking_head_keypoints])

                if is_clapping:
                    detected = 'clapping'
                    cv2.putText(image, 'Clapping', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif is_hi:
                    cv2.putText(image, 'Hi', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif is_stretching:
                    cv2.putText(image, 'Stretching', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif is_raising_arms:
                    cv2.putText(image, 'Raising Arms', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif is_pointing:
                    cv2.putText(image, 'Pointing', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif is_sleeping:
                    cv2.putText(image, 'Sleeping', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif is_shaking_head:
                    cv2.putText(image, 'Shaking Head', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Pose Detection', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()


def start_video_capture():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    cv2.namedWindow("Video Capture", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Video Capture", 100, 100)
    cv2.resizeWindow("Video Capture", 800, 600)

    while True:
        ret, frame = cap.read()
        cv2.imshow("Video Capture", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def quit_program():
    root.destroy()

root = tk.Tk()
root.title("Program Selector")
root.geometry("400x400") 
root.configure(bg="#0D6E9B") 
label = tk.Label(root, text="Choose a program to start:", fg="#FFFFFF", bg="#0D6E9B", font=("Arial", 24, "bold"))
label.pack(pady=20)
detection_button = tk.Button(root, text="Detection", command=start_detection, fg="#FFFFFF", bg="#000000", font=("Arial", 18, "bold"), padx=30, pady=15)
detection_button.pack(pady=20)

video_button = tk.Button(root, text="Video Capture", command=start_video_capture, fg="#FFFFFF", bg="#000000", font=("Arial", 18, "bold"), padx=30, pady=15)
video_button.pack(pady=20)

quit_button = tk.Button(root, text="Quit", command=quit_program, fg="#FFFFFF", bg="#000000", font=("Arial", 18, "bold"), padx=30, pady=15)
quit_button.pack(pady=20)
root.mainloop()