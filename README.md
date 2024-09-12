# DW-NCA: Distance-Weighted Neighbourhood Component Analysis

DW-NCA is a Python library that implements the Distance-Weighted Neighbourhood Component Analysis algorithm for feature transformation and classification.

## Installation

You can install DW-NCA using pip:

```
pip install dw_nca
```

## Usage

Here's a basic example of how to use DW-NCA:

```python
from dw_nca.knn_classifier import DWNCA_KNNClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the DWNCA_KNNClassifier
classifier = DWNCA_KNNClassifier(n_neighbors=3, n_components=2)
classifier.fit(X_train, y_train)

# Make predictions
accuracy = classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

For more advanced usage and examples, please refer to the documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


# LICENSE:

MIT License

Copyright (c) 2024 UBANDIYA Najib Yusuf

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

requirements.txt:

```
numpy>=1.18.0
scipy>=1.4.0
scikit-learn>=0.22.0
```
