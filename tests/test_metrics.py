import unittest
import numpy as np
from dw_nca.metrics import PerformanceMetrics, cross_validation_metrics
from dw_nca.knn_classifier import DWNCA_KNNClassifier


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 1, 2, 0, 1, 2])
        self.y_pred = np.array([0, 2, 1, 0, 1, 1])

    def test_performance_metrics(self):
        metrics = PerformanceMetrics.all_metrics(self.y_true, self.y_pred)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)

    def test_cross_validation_metrics(self):
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)
        model = DWNCA_KNNClassifier()
        cv_metrics = cross_validation_metrics(model, X, y, cv=3)
        self.assertIn('accuracy', cv_metrics)
        self.assertIn('precision', cv_metrics)
        self.assertIn('recall', cv_metrics)
        self.assertIn('f1_score', cv_metrics)

if __name__ == '__main__':
    unittest.main()
