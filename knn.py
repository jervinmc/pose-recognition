import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Define the labels and the corresponding hand landmarks
labels = ["down", "left", "right", "up"]
landmarks = [[0, 5], [5, 9], [9, 13], [13, 17], [17, 21], [21, 25]]

# Create a MediaPipe Hands object
mp_hands = mp.solutions.hands.Hands()

# Initialize the features and labels lists
features = []
labels_list = []

# Load the data from the dataset folder
for label in labels:
    folder_path = f"dataset/{label}"
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)
        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in landmarks:
                    x = hand_landmarks.landmark[lm[0]].x
                    y = hand_landmarks.landmark[lm[0]].y
                    landmarks_list.append(x)
                    landmarks_list.append(y)
        if landmarks_list:
            features.append(landmarks_list)
            labels_list.append(label)
        print(labels_list)

# Encode the labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels_list)

print(labels_list)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Create a k-NN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the k-NN model
knn.fit(X_train, y_train)

# Start the video stream
cap = cv2.VideoCapture(0)

# Start the main loop
while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the hands in the frame
    results = mp_hands.process(rgb)

    # Extract the hand landmarks
    landmarks_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in landmarks:
                x = hand_landmarks.landmark[lm[0]].x
                y = hand_landmarks.landmark[lm[0]].y
                landmarks_list.append(x)
                landmarks_list.append(y)

    # Classify the pose using the k-NN model
    if landmarks_list:
        label_encoded = knn.predict([landmarks_list])[0]
        label = le.inverse_transform([label_encoded])[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (10, 50), font, 2, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Stop the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
mp_hands.close()