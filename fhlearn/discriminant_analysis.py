# %%

import numpy as np
# %%
from model_selection import train_test_split
# load some data

data = np.loadtxt('data/wine.csv',delimiter=',')
#X_train,X_test,y_train,y_test = train_test_split(data)
X_train = data[:,:-1]
y_train = data[:,-1]
# %%

class LinearDiscriminantAnalysis:
    def __init__(self,solver='eigen'):
        self.solver = solver
        self.n_features: int = None
        self.n_classes: int = None

    def _get_n_classes(self, labels: np.array):
        return len(np.unique(labels))

    def _get_n_obs(self, data: np.array) -> int:
        return np.size(data, axis=0)

    def _class_feature_means(
            self, 
            features: np.array, 
            labels: np.array
        ) -> np.array:
        mean_vectors = []
        for class_ in range(self.n_classes):
            mean = np.mean(features[labels == class_], axis=0)
            mean_vectors.append(np.reshape(mean, (-1,1)))
        return np.concatenate(mean_vectors, axis=1)

    def _get_within_scatter_matrix(
            self,
            features: np.array,
            labels: np.array
        ) -> np.array:
        Sw = np.zeros((self.n_features,self.n_features))
        class_means = self._class_feature_means(features,labels)
        for class_ in range(self.n_classes):
            mean_vector = np.reshape(class_means[:,class_], (-1,1))
            x = features[labels == class_]
            for i in range(len(x)):
                x_i = np.reshape(x[i], (-1,1))
                Sw += (x_i-mean_vector) @ ((x_i-mean_vector).T)
        return Sw

    def _get_between_scatter_matrix(
            self,
            features: np.array,
            labels: np.array
        ) -> np.array:
        Sb = np.zeros((self.n_features,self.n_features))
        class_means = self._class_feature_means(features,labels)
        feature_means = np.mean(features, axis=0)

        for class_ in range(self.n_classes):
            data = features[labels == class_]
            n = self._get_n_obs(data)
            mc, m = np.reshape(class_means[:,class_], (-1,1)), np.reshape(feature_means, (-1,1))
            Sb += n * (mc - m) @ ((mc - m).T)

        return Sb

    def _get_eigen(
            self, 
            within_class_scatter_matrix: np.array, 
            between_class_scatter_matrix: np.array
        ):
        Sw_inv = np.linalg.inv(within_class_scatter_matrix)
        
        problem = Sw_inv @ between_class_scatter_matrix
        return np.linalg.eig(problem)

    def fit(self,features: np.array, labels: np.array):
        self.n_features = np.size(features, 1)
        self.n_classes = self._get_n_classes(labels)
        

        Sw = self._get_within_scatter_matrix(features,labels)
        Sb = self._get_between_scatter_matrix(features, labels)
        eigen_values, eigen_vectors = self._get_eigen(Sw,Sb)
        pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
        for pair in pairs:
            print(pair[0])


lda = LinearDiscriminantAnalysis()

lda.fit(X_train,y_train)

# %%
