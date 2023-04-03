import cv2
import speech_recognition as sr
import threading

# Initialize speech recognition module
r = sr.Recognizer()

# Open video stream
cap = cv2.VideoCapture(0)

# Create a lock to synchronize access to the shared video frame
frame_lock = threading.Lock()

# Define a worker function that captures video frames
def capture_frames():
    global frame_lock

    while True:
        # Read frame from video stream
        ret, frame = cap.read()

        # Acquire lock to prevent other threads from accessing the shared frame
        with frame_lock:
            # Do something with the video frame
            # ...

            # Display frame
            cv2.imshow("Video Stream", frame)

        # Exit program on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Define a worker function that performs speech recognition
def recognize_speech():
    while True:
        # Perform speech recognition on audio from microphone
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