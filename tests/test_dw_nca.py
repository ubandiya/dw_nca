import unittest
import numpy as np
from dw_nca.dw_nca import DistanceWeightedNCA

class TestDistanceWeightedNCA(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randint(0, 3, 100)
        self.dw_nca = DistanceWeightedNCA(n_components=3, max_iter=1000, tol=1e-8)

    def test_fit_transform(self):
        print("Running test_fit_transform...")
        try:
            X_transformed = self.dw_nca.fit_transform(self.X, self.y)
            self.assertEqual(X_transformed.shape, (self.X.shape[0], 3))
            print("test_fit_transform passed!")
        except Exception as e:
            print(f"Error in test_fit_transform: {str(e)}")
            raise

    def test_fit_and_transform(self):
        print("Running test_fit_and_transform...")
        try:
            self.dw_nca.fit(self.X, self.y)
            X_transformed = self.dw_nca.transform(self.X)
            self.assertEqual(X_transformed.shape, (self.X.shape[0], 3))
            print("test_fit_and_transform passed!")
        except Exception as e:
            print(f"Error in test_fit_and_transform: {str(e)}")
            raise

    def test_transform_new_data(self):
        print("Running test_transform_new_data...")
        try:
            self.dw_nca.fit(self.X, self.y)
            X_new = np.random.randn(10, 5)
            X_transformed = self.dw_nca.transform(X_new)
            self.assertEqual(X_transformed.shape, (10, 3))
            print("test_transform_new_data passed!")
        except Exception as e:
            print(f"Error in test_transform_new_data: {str(e)}")
            raise

    def test_invalid_input(self):
        print("Running test_invalid_input...")
        with self.assertRaises(ValueError):
            self.dw_nca.fit(self.X, self.y[:50])
        print("test_invalid_input passed!")

if __name__ == '__main__':
    print("Starting tests...")
    unittest.main(verbosity=2)
