import numpy as np
from math import sqrt
from operator import itemgetter
import random


class KNeighborsClassifier:
    def __init__(self, n_neighbors: int = 5, weights='uniform',random_state: int = None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.random_state = random_state
        self.train_features = None
        self.train_labels = None
        self.n_train_obs = None
        self.is_fitted = False

    def _compute_distance(self,x,y):
        if not len(x) == len(y):
            raise ValueError('Inconsistent number of dimensions in passed samples')
        return sqrt(sum([(x[i]-y[i]) ** 2 for i in range(len(x))]))

    def _is_series(self, data):
        return len(np.shape(data)) == 1

    def _get_majority_class(self, distances):
        if self.random_state:
            random.seed(self.random_state)
        votes = {class_:0 for (distance,class_) in distances}
        if self.weights == 'distance':
            for i in range(len(distances)):
                distances[i][0] = 1 / distances[i][0]
            for entry in distances:
                distance, class_ = entry
                votes[class_] += (1 * distance)
        else:
            for entry in distances:
                distance, class_ = entry
                votes[class_] += 1
        major_classes = []
        max_count = float('-inf')
        for class_,count in votes.items():
            if count > max_count:
                max_count = count
                major_classes = [class_]
            elif count == max_count:
                major_classes.append(class_)
        return random.choice(major_classes)

    def fit(self,features: np.array, labels: np.array) -> None:
        if not np.size(features, axis=0) == np.size(labels, axis=0):
            raise ValueError('Given feature and label sets not containing equal number of samples')
        self.train_features = features
        self.train_labels = labels
        self.is_fitted = True
        self.n_train_obs = np.size(features,axis=0)

    def _predict_sample(self, sample):
        distances = []
        for i in range(self.n_train_obs):
            distances.append((self._compute_distance(sample,self.train_features[i]),self.train_labels[i]))
        sorted_distances = sorted(distances,key=itemgetter(0))
        return self._get_majority_class(sorted_distances[:self.n_neighbors])

    def predict(self,features) -> np.array:
        predictions = []
        if self._is_series(features):
            features = np.reshape(features, (1,-1))
        n_observations = np.size(features, axis=0)
        for i in range(n_observations):
            observation = features[i]
            predictions.append(self._predict_sample(observation))
        return predictions

