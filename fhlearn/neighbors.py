import numpy as np
from math import sqrt


class KNeighborsClassifier:
    def __init__(self, n_neighbors: int = 5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.train_features = None
        self.train_labels = None

    def _compute_distance(self,x,y):
        pass



    def fit(self,features: np.array, labels: np.array) -> None:
        if not np.size(features, axis=0) == np.size(labels, axis=0):
            raise ValueError('Given feature and label sets not containing equal number of samples')

        self.train_features = features
        self.train_labels = labels

    def _predict_sample(self, sample):
        print(sample)

        return

    def predict(self,features) -> np.array:
        predictions = []

        n_observations = np.size(features, axis=0)
        for i in range(n_observations):
            observation = features[i]
            predictions.append(self._predict_sample(observation))

        return predictions