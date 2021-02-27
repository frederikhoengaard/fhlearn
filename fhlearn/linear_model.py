import numpy as np


class LinearRegressionModel:
    def __init__(self):
        self.theta = None
        self.X = None
        self.y = None

    def fit(self,features,targets):
        self.X = np.copy(features)
        self.y = np.copy(targets)
        self.X = np.c_[np.ones((len(self.X), 1)), self.X] # add x0 = 1 to each instance
        self.theta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
       
    def predict(self,features):
        X_new = np.c_[np.ones((len(features), 1)), features] # add x0 = 1 to each instance
        return X_new.dot(self.theta)

    def show_theta(self):
        return self.theta