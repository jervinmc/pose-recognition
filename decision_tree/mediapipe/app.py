import cv2
import mediapipe as mp

# Define the Mediapipe solutions
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define the video stream
cap = cv2.VideoCapture(0)

# Set up the Mediapipe pose detector
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detector:

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        
        # Convert the frame to RGB color space
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with Mediapipe
        results = pose_detector.process(frame)
        
        # Check if the person is clapping or standing
        if results.pose_landmarks:
            # Calculate the distance between the left and right hands
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            distance = ((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2) ** 0.5
            
            # Check if the hands are close enough to be clapping
            if distance < 0.1:
                print("Clapping detected!")
            else:
                print("Standing detected!")
        
        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Convert the frame back to BGR color space
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Mediapipe Clap Detection', frame)
        
        # Exit the program when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()