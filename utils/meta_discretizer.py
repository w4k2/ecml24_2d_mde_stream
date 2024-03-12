import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import KBinsDiscretizer


class Discretizer(BaseEstimator, ClassifierMixin):
    def __init__(self, clf):
        self.clf = clf
        self.dis = KBinsDiscretizer(n_bins=3)
        
    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        
        self.X_ = self.dis.fit_transform(self.X_).toarray().astype(int)
        self.clf.fit(self.X_, self.y_)
        
        return self
    
    def partial_fit(self, X, y, classes=None):
        self.X_ = X
        self.y_ = y
        
        self.X_ = self.dis.fit_transform(self.X_).toarray().astype(int)
        self.clf.partial_fit(self.X_, self.y_)
        
        return self
    
    def predict(self, X):
        X = self.dis.fit_transform(X).toarray().astype(int)
        pred = self.clf.predict(X)
        return pred