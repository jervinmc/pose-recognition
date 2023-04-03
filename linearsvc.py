import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Step 1: Collect dataset
# TODO: Collect dataset and label the postures

# Step 2: Feature Extraction
# TODO: Extract features from the images or videos

# Step 3: Data Preprocessing
# TODO: Scale and normalize the data

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Selecting the Model
svm_model = SVC(kernel='linear')

# Step 6: Training the Model
svm_model.fit(X_train, y_train)

# Step 7: Testing the Model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 score: {f1:.2f}")
