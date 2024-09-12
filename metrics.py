from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class PerformanceMetrics:
    """
    Performance metrics class for evaluating classification models.

    Methods:
    --------
    accuracy(y_true, y_pred):
        Compute the accuracy of the model.
    
    precision(y_true, y_pred):
        Compute the precision of the model.
    
    recall(y_true, y_pred):
        Compute the recall of the model.
    
    f1(y_true, y_pred):
        Compute the F1 score of the model.
    
    all_metrics(y_true, y_pred):
        Compute all metrics and return them as a dictionary.
    """

    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def precision(y_true, y_pred):
        return precision_score(y_true, y_pred, average='weighted')
    
    @staticmethod
    def recall(y_true, y_pred):
        return recall_score(y_true, y_pred, average='weighted')
    
    @staticmethod
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')
    
    @staticmethod
    def all_metrics(y_true, y_pred):
        return {
            'accuracy': PerformanceMetrics.accuracy(y_true, y_pred),
            'precision': PerformanceMetrics.precision(y_true, y_pred),
            'recall': PerformanceMetrics.recall(y_true, y_pred),
            'f1_score': PerformanceMetrics.f1(y_true, y_pred)
        }

def cross_validation_metrics(model, X, y, cv=5):
    """
    Perform cross-validation and return performance metrics.

    Parameters:
    -----------
    model : estimator object
        The model to evaluate.
    
    X : array-like, shape (n_samples, n_features)
        The data to fit.
    
    y : array-like, shape (n_samples,)
        The target variable to try to predict in the case of supervised learning.
    
    cv : int, optional (default=5)
        Number of cross-validation folds.

    Returns:
    --------
    metrics : dict
        Cross-validated performance metrics including accuracy, precision,
        recall, and F1 score.
    """
    accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision_weighted')
    recall = cross_val_score(model, X, y, cv=cv, scoring='recall_weighted')
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

    return {
        'accuracy': np.mean(accuracy),
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1_score': np.mean(f1)
    }
