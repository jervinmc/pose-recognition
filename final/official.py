import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    if results.pose_landmarks:
        left_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HAND]
        right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HAND]
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        distance = abs(left_hand.x - nose.x) + abs(right_hand.x - nose.x) + abs(left_hand.y - nose.y) + abs(right_hand.y - nose.y)
        if distance < 0.2:
            print("Sleeping!")
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        if left_wrist.y > left_elbow.y and right_wrist.y > right_elbow.y:
            print("Clapping!")
        cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
pose.close()
cap.release()
cv2.destroyAllWindows()