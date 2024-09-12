import unittest
import numpy as np
from dw_nca.knn_classifier import DWNCA_KNNClassifier

class TestDWNCA_KNNClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 0, 1, 1])
        self.classifier = DWNCA_KNNClassifier(n_neighbors=3)

    def test_fit_predict(self):
        self.classifier.fit(self.X, self.y)
        y_pred = self.classifier.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))

    def test_predict_proba(self):
        self.classifier.fit(self.X, self.y)
        y_proba = self.classifier.predict_proba(self.X)
        self.assertEqual(y_proba.shape, (len(self.X), len(np.unique(self.y))))

    def test_score(self):
        score = self.classifier.fit(self.X, self.y).score(self.X, self.y)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main()
