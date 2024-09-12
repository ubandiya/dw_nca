from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .dw_nca import DistanceWeightedNCA
from .metrics import PerformanceMetrics, cross_validation_metrics

class DWNCA_KNNClassifier(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors Classifier with Distance-Weighted NCA transformation

    This classifier combines DW-NCA for feature transformation with
    K-Nearest Neighbors for classification.

    Parameters:
    -----------
    n_neighbors : int, optional (default=5)
        Number of neighbors to use for KNN.
    
    n_components : int, optional (default=None)
        Number of components for DW-NCA transformation.
        If None, use all features.
    
    max_iter : int, optional (default=100)
        Maximum number of iterations for DW-NCA optimization.
    
    tol : float, optional (default=1e-5)
        Tolerance for termination of the DW-NCA optimization algorithm.
    """

    def __init__(self, n_neighbors=5, n_components=None, max_iter=100, tol=1e-5):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """
        Fit the DW-NCA transformation and KNN classifier.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        
        self.dw_nca_ = DistanceWeightedNCA(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol
        )
        X_transformed = self.dw_nca_.fit_transform(X, y)
        
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn_.fit(X_transformed, y)
        
        self.classes_ = self.knn_.classes_
        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Class labels for each data sample.
        """
        check_is_fitted(self, ['dw_nca_', 'knn_'])
        X = check_array(X)
        X_transformed = self.dw_nca_.transform(X)
        return self.knn_.predict(X_transformed)

    def predict_proba(self, X):
        """
        Predict class probabilities for the provided data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        Returns:
        --------
        y_proba : array-like, shape (n_samples, n_classes)
            Probability of each class for each sample.
        """
        check_is_fitted(self, ['dw_nca_', 'knn_'])
        X = check_array(X)
        X_transformed = self.dw_nca_.transform(X)
        return self.knn_.predict_proba(X_transformed)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True labels for X.

        Returns:
        --------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        return super().score(X, y)
