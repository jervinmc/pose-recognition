import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load dataset
X_train, y_train = np.load('train_images.npy'), np.load('train_labels.npy')
X_test, y_test = np.load('test_images.npy'), np.load('test_labels.npy')

# Initialize DBN model
dbn = pd.models.DBN(input_shape=X_train.shape[1:], output_shape=len(np.unique(y_train)))

# Train DBN model
dbn.fit(X_train, y_train)

# Test DBN model
accuracy = dbn.score(X_test, y_test)
print("Test accuracy: {:.2f}%".format(accuracy * 100))