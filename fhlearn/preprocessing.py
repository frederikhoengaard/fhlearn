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
