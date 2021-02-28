import numpy as np


class LinearRegression: # HML p 106
    def __init__(self):
        self.best_theta = None
        self.intercept_ = None
        self.coef_ = None

    def fit(self, features, targets):
        X = np.copy(features)
        y = np.copy(targets)
        X = np.c_[np.ones((len(X), 1)), X] # add x0 = 1 to each instance
        self.best_theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept_ = self.best_theta[0]
        self.coef_ = self.best_theta[1:]
       
    def predict(self, features):
        predictions = np.c_[np.ones((len(features), 1)), features] # add x0 = 1 to each instance
        return predictions.dot(self.best_theta)