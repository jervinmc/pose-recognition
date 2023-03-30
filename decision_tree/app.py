import os
import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_and_test_image_recognition(image_folder_path, test_size=0.2, random_state=42):
    # Load images from folder and convert to grayscale
    images = []
    labels = []
    for filename in os.listdir(image_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img = Image.open(os.path.join(image_folder_path, filename)).convert('L')
            img = np.array(img).flatten()
            print(img)
            images.append(img)
            label = filename.split("_")[0]
            labels.append(label)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)

    # Train a decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    return clf, accuracy

# Path to folder containing images
image_folder_path = "images"

# Train and test the image recognition model
model, accuracy = train_and_test_image_recognition(image_folder_path)

# Print the accuracy of the model
print("Accuracy:", accuracy)
print("Model:",model)


img1 = Image.open("okay1.png").convert('L')
img1 = np.array(img1).flatten()
label = model.predict([img1])[0]

print(label)
