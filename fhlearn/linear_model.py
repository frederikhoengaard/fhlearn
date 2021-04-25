import numpy as np
from metrics import sum_squared_errors, sum_squared_residuals


class LinearRegression: # HML p 106
    def __init__(self):
        self.best_theta = None
        self.intercept_ = None
        self.coef_ = None
        self.is_fitted = False

    def fit(self, features: np.array, targets: np.array) -> None:
        X = np.copy(features)
        y = np.copy(targets)
        X = np.c_[np.ones((len(X), 1)), X] # add x0 = 1 to each instance
        self.best_theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept_ = self.best_theta[0]
        self.coef_ = self.best_theta[1:]
        self.is_fitted = True
       
    def predict(self, features: np.array) -> np.array:
        predictions = np.c_[np.ones((len(features), 1)), features] # add x0 = 1 to each instance
        return predictions.dot(self.best_theta)

    def score(self, features:np.array, targets: np.array) -> float:
        if not self.is_fitted:
            raise ValueError('Linear regression model must first be fitted!')
        predicted_targets = self.predict(features)
        return 1 - (sum_squared_residuals(targets,predicted_targets) / sum_squared_errors(targets,predicted_targets))

    
class LogisticRegression:
    def __init__(self):
        self.learning_rate: float = 0.2
        self.coef_ = None

    def _sigmoid_func(self,x):
        return 1 / (1 + np.exp(-x))

    def _hypothesis(self, sample, theta):
        return self._sigmoid_func(sample @ theta)

    def _cost(self):
        pass

    def fit(self):
        if not self.coef_: # initialise thetas
            np.random.seed(42) 
            self.coef_ = np.random.uniform(-1,1,m+1)

    def predict(self):
        pass


def main():


    pass

if __name__ == '__main__':
    main()