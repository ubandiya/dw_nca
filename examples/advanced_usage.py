import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dw_nca.dw_nca import DistanceWeightedNCA
from dw_nca.knn_classifier import DWNCA_KNNClassifier
from dw_nca.metrics import cross_validation_metrics

# Load the wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create and train the DistanceWeightedNCA
dw_nca = DistanceWeightedNCA(n_components=5)
X_train_transformed = dw_nca.fit_transform(X_train, y_train)
X_test_transformed = dw_nca.transform(X_test)

# Create and train the DWNCA_KNNClassifier
classifier = DWNCA_KNNClassifier(n_neighbors=5)
classifier.fit(X_train_transformed, y_train)

# Perform cross-validation
cv_metrics = cross_validation_metrics(classifier, X_train_transformed, y_train, cv=5)

print("Cross-validation Metrics:")
for metric, value in cv_metrics.items():
    print(f"{metric}: {value:.4f}")

# Make predictions on the test set
y_pred = classifier.predict(X_test_transformed)
accuracy = classifier.score(X_test_transformed, y_test)
print(f"\nTest Set Accuracy: {accuracy:.4f}")
