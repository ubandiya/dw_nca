import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.optimize

class DistanceWeightedNCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, max_iter=1000, tol=1e-8, learning_rate=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = self._validate_data(X)
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples.")
        
        # Set number of features and classes
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))

        # Set the number of components
        n_components = self.n_components or self.n_features_

        # Initialize transformation matrix
        initial_A = np.random.randn(n_components, self.n_features_)

        def objective(A):
            A = A.reshape(n_components, self.n_features_)
            return -self._dw_nca_objective(X, y, A)

        def gradient(A):
            A = A.reshape(n_components, self.n_features_)
            return -self._dw_nca_gradient(X, y, A).ravel()

        # Perform optimization
        result = scipy.optimize.minimize(
            objective, initial_A.ravel(), method='L-BFGS-B', jac=gradient,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tol,
                'gtol': 1e-6,
                'disp': False,
                'maxls': 50,  # Increase max line search steps
            }
        )

        if not result.success:
            print(f"Optimization warning: {result.message}")
        
        # Save the learned components
        self.components_ = result.x.reshape(n_components, self.n_features_)
        return self

    def transform(self, X):
        check_is_fitted(self, 'components_')
        X = self._validate_data(X, reset=False)
        return np.dot(X, self.components_.T)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _dw_nca_objective(self, X, y, A):
        X_transformed = np.dot(X, A.T)
        distances = self._pairwise_squared_distances(X_transformed)
        epsilon = 1e-10
        weights = 1 / (1 + distances + epsilon)
        max_distance = 700
        p = weights * np.exp(-np.clip(distances, 0, max_distance))
        np.fill_diagonal(p, 0)
        p /= np.sum(p, axis=1, keepdims=True) + epsilon
    
        mask = y[:, np.newaxis] == y[np.newaxis, :]
        return np.sum(p[mask])

    def _dw_nca_gradient(self, X, y, A):
        X_transformed = np.dot(X, A.T)
        distances = self._pairwise_squared_distances(X_transformed)
        epsilon = 1e-10
        weights = 1 / (1 + distances + epsilon)
        max_distance = 700
        p = weights * np.exp(-np.clip(distances, 0, max_distance))
        np.fill_diagonal(p, 0)
        p /= np.sum(p, axis=1, keepdims=True) + epsilon
    
        mask = y[:, np.newaxis] == y[np.newaxis, :]
        grad = np.zeros_like(A)
        for i in range(len(X)):
            dij = X_transformed[i] - X_transformed
            w_grad = -2 / (1 + distances[i] + epsilon)**2
            p_grad = p[i] * (w_grad - 2*weights[i])
            grad_i = np.dot(p_grad * mask[i] - p[i] * np.sum(p_grad * mask[i]), dij)
            grad += np.outer(grad_i, X[i])
        return grad

    def _pairwise_squared_distances(self, X):
        sum_X = np.sum(X**2, axis=1)
        return sum_X[:, np.newaxis] + sum_X - 2 * np.dot(X, X.T)
