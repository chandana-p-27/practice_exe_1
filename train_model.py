from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Initialize the logistic regression model
clf = LogisticRegression(max_iter=200)

# Train the model
clf.fit(X, y)

# Print success message
print("Model trained successfully!")
