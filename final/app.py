import cv2
import mediapipe as mp
# import speech_recognition as sr

# r = sr.Recognizer()
# mic = sr.Microphone()
mp_pose = mp.solutions.pose
clapping_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_WRIST, mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
hi_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, 
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, 
                mp.solutions.pose.PoseLandmark.LEFT_WRIST, mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
sleeping_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_EYE, mp.solutions.pose.PoseLandmark.RIGHT_EYE]

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:
    while True:
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detection.process(image_rgb)
        # Extract the landmarks of the pose
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

            # Recognize the different poses based on the keypoints
            is_clapping = all([keypoints[k].visibility > 0.5 for k in clapping_keypoints])
            is_hi = all([keypoints[k].visibility > 0.5 for k in hi_keypoints])
            is_sleeping = all([keypoints[k].visibility > 0.5 for k in sleeping_keypoints])

            # Display the recognized pose
            if is_clapping:
                cv2.putText(image, 'Clapping', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_hi:
                cv2.putText(image, 'Hi', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_sleeping:
                cv2.putText(image, 'Sleeping', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Pose Detection', image)
        # with mic as source:
        #     r.adjust_for_ambient_noise(source)
        #     audio = r.listen(source)
        # try:
        #     text = r.recognize_google(audio)
        #     print("You said: ", text)

        # except sr.UnknownValueError:
        #     print("Could not understand audio")

        # except sr.RequestError as e:
        #     print("Error: {0}".format(e))
        
        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()

