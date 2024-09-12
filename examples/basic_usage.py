import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from dw_nca.knn_classifier import DWNCA_KNNClassifier
from dw_nca.metrics import PerformanceMetrics

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the DWNCA_KNNClassifier
classifier = DWNCA_KNNClassifier(n_neighbors=3, n_components=2)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Calculate and print performance metrics
metrics = PerformanceMetrics.all_metrics(y_test, y_pred)
print("Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
