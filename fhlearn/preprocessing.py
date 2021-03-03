import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, features: np.array) -> None:
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)

    def fit_transform(self, features: np.array) -> np.array:
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        features_copy = np.copy(features)
        return (features_copy - self.mean) / self.std

    def transform(self, features: np.array) -> np.array:
        features_copy = np.copy(features)
        return (features_copy - self.mean) / self.std


class LabelEncoder:
    def __init__(self, labels: np.array = None):
        self.labels = labels
        self.encoded_labels: dict = None
        self.has_encoded = False

    def encode(self,labels):
        unique_labels = sorted(np.unique(labels))
        self.encoded_labels = {unique_labels[i]:i for i in range(len(unique_labels))}
        self.has_encoded = True 

    def transform(self, labels):
        if not self.has_encoded:
            raise ValueError('LabelEncoder must be encoded before call to transform.')
        copy = np.copy(labels)
        for i in range(len(labels)):
            copy[i] = self.encoded_labels[copy[i]]
        return copy

    def encode_transform(self,labels):
        unique_labels = sorted(np.unique(labels))
        self.encoded_labels = {unique_labels[i]:i for i in range(len(unique_labels))}
        self.has_encoded = True
        copy = np.copy(labels)
        for i in range(len(labels)):
            copy[i] = self.encoded_labels[copy[i]]
        return copy