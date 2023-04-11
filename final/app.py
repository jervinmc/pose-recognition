import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
clapping_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_WRIST, mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
hi_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, 
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, 
                mp.solutions.pose.PoseLandmark.LEFT_WRIST, mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
sleeping_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_EYE, mp.solutions.pose.PoseLandmark.RIGHT_EYE]
pointing_keypoints = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW, 
                      mp.solutions.pose.PoseLandmark.LEFT_WRIST]

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

            if is_clapping:
                cv2.putText(image, 'Clapping', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_hi:
                cv2.putText(image, 'Hi', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_sleeping:
                cv2.putText(image, 'Sleeping', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Pose Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()

