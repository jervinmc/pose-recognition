import cv2
import speech_recognition as sr
import threading

r = sr.Recognizer()

cap = cv2.VideoCapture(0)

frame_lock = threading.Lock()

def capture_frames():
    global frame_lock

    while True:
        ret, frame = cap.read()
        with frame_lock:
            cv2.imshow("Video Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def recognize_speech():
    while True:
        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                print("Speech Recognition:", text)
            except sr.UnknownValueError:
                print("Speech Recognition: Could not understand audio")
            except sr.RequestError as e:
                print("Speech Recognition: Could not request results; {0}".format(e))

# Create worker threads
capture_thread = threading.Thread(target=capture_frames)
speech_thread = threading.Thread(target=recognize_speech)

# Start worker threads
capture_thread.start()
speech_thread.start()

# Wait for worker threads to finish
capture_thread.join()
speech_thread.join()

# Release video stream and close windows
cap.release()
cv2.destroyAllWindows()